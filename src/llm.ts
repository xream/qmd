/**
 * llm.ts - LLM abstraction layer for QMD using node-llama-cpp
 *
 * Provides embeddings, text generation, and reranking using local GGUF models.
 */

import {
  getLlama,
  resolveModelFile,
  LlamaChatSession,
  LlamaLogLevel,
  type Llama,
  type LlamaModel,
  type LlamaEmbeddingContext,
  type Token as LlamaToken,
} from "node-llama-cpp";
import { homedir } from "os";
import { join } from "path";
import { existsSync, mkdirSync, statSync, unlinkSync, readdirSync, readFileSync, writeFileSync } from "fs";
import { loadConfig, type ApiModelConfig as ConfigApiModelConfig } from "./collections.js";

// =============================================================================
// Embedding Formatting Functions
// =============================================================================

/**
 * Format a query for embedding.
 * Uses nomic-style task prefix format for embeddinggemma.
 */
export function formatQueryForEmbedding(query: string): string {
  return `task: search result | query: ${query}`;
}

/**
 * Format a document for embedding.
 * Uses nomic-style format with title and text fields.
 */
export function formatDocForEmbedding(text: string, title?: string): string {
  return `title: ${title || "none"} | text: ${text}`;
}

// =============================================================================
// Types
// =============================================================================

/**
 * Token with log probability
 */
export type TokenLogProb = {
  token: string;
  logprob: number;
};

/**
 * Embedding result
 */
export type EmbeddingResult = {
  embedding: number[];
  model: string;
};

/**
 * Generation result with optional logprobs
 */
export type GenerateResult = {
  text: string;
  model: string;
  logprobs?: TokenLogProb[];
  done: boolean;
};

/**
 * Rerank result for a single document
 */
export type RerankDocumentResult = {
  file: string;
  score: number;
  index: number;
};

/**
 * Batch rerank result
 */
export type RerankResult = {
  results: RerankDocumentResult[];
  model: string;
};

/**
 * Model info
 */
export type ModelInfo = {
  name: string;
  exists: boolean;
  path?: string;
};

/**
 * Options for embedding
 */
export type EmbedOptions = {
  model?: string;
  isQuery?: boolean;
  title?: string;
};

/**
 * Options for text generation
 */
export type GenerateOptions = {
  model?: string;
  maxTokens?: number;
  temperature?: number;
};

/**
 * Options for reranking
 */
export type RerankOptions = {
  model?: string;
};

/**
 * Options for LLM sessions
 */
export type LLMSessionOptions = {
  /** Max session duration in ms (default: 10 minutes) */
  maxDuration?: number;
  /** External abort signal */
  signal?: AbortSignal;
  /** Debug name for logging */
  name?: string;
};

/**
 * Session interface for scoped LLM access with lifecycle guarantees
 */
export interface ILLMSession {
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null>;
  embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]>;
  expandQuery(query: string, options?: { context?: string; includeLexical?: boolean }): Promise<Queryable[]>;
  rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult>;
  /** Whether this session is still valid (not released or aborted) */
  readonly isValid: boolean;
  /** Abort signal for this session (aborts on release or maxDuration) */
  readonly signal: AbortSignal;
}

/**
 * Supported query types for different search backends
 */
export type QueryType = 'lex' | 'vec' | 'hyde';

/**
 * A single query and its target backend type
 */
export type Queryable = {
  type: QueryType;
  text: string;
};

/**
 * Document to rerank
 */
export type RerankDocument = {
  file: string;
  text: string;
  title?: string;
};

export type ApiModelRuntimeConfig = {
  baseUrl: string;
  key: string;
  model: string;
};

// =============================================================================
// Model Configuration
// =============================================================================

// HuggingFace model URIs for node-llama-cpp
// Format: hf:<user>/<repo>/<file>
const DEFAULT_EMBED_MODEL = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
const DEFAULT_RERANK_MODEL = "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";
// const DEFAULT_GENERATE_MODEL = "hf:ggml-org/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf";
const DEFAULT_GENERATE_MODEL = "hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf";

export const DEFAULT_EMBED_MODEL_URI = DEFAULT_EMBED_MODEL;
export const DEFAULT_RERANK_MODEL_URI = DEFAULT_RERANK_MODEL;
export const DEFAULT_GENERATE_MODEL_URI = DEFAULT_GENERATE_MODEL;

// Local model cache directory
const MODEL_CACHE_DIR = join(homedir(), ".cache", "qmd", "models");
export const DEFAULT_MODEL_CACHE_DIR = MODEL_CACHE_DIR;

export type PullResult = {
  model: string;
  path: string;
  sizeBytes: number;
  refreshed: boolean;
};

type ResolvedApiModelConfigs = {
  embedApi: ApiModelRuntimeConfig | null;
  queryApi: ApiModelRuntimeConfig | null;
  rerankApi: ApiModelRuntimeConfig | null;
};

function normalizeApiModelConfig(config?: ConfigApiModelConfig): ApiModelRuntimeConfig | null {
  const baseUrl = config?.base_url?.trim();
  const key = config?.key?.trim();
  const model = config?.model?.trim();
  if (!baseUrl || !key || !model) {
    return null;
  }
  return {
    baseUrl: baseUrl.replace(/\/+$/, ""),
    key,
    model,
  };
}

function resolveApiModelConfigsFromYaml(): ResolvedApiModelConfigs {
  const config = loadConfig();
  return {
    embedApi: normalizeApiModelConfig(config.models?.embed),
    queryApi: normalizeApiModelConfig(config.models?.query),
    rerankApi: normalizeApiModelConfig(config.models?.rerank),
  };
}

type HfRef = {
  repo: string;
  file: string;
};

function parseHfUri(model: string): HfRef | null {
  if (!model.startsWith("hf:")) return null;
  const without = model.slice(3);
  const parts = without.split("/");
  if (parts.length < 3) return null;
  const repo = parts.slice(0, 2).join("/");
  const file = parts.slice(2).join("/");
  return { repo, file };
}

async function getRemoteEtag(ref: HfRef): Promise<string | null> {
  const url = `https://huggingface.co/${ref.repo}/resolve/main/${ref.file}`;
  try {
    const resp = await fetch(url, { method: "HEAD" });
    if (!resp.ok) return null;
    const etag = resp.headers.get("etag");
    return etag || null;
  } catch {
    return null;
  }
}

export async function pullModels(
  models: string[],
  options: { refresh?: boolean; cacheDir?: string } = {}
): Promise<PullResult[]> {
  const cacheDir = options.cacheDir || MODEL_CACHE_DIR;
  if (!existsSync(cacheDir)) {
    mkdirSync(cacheDir, { recursive: true });
  }

  const results: PullResult[] = [];
  for (const model of models) {
    let refreshed = false;
    const hfRef = parseHfUri(model);
    const filename = model.split("/").pop();
    const entries = readdirSync(cacheDir, { withFileTypes: true });
    const cached = filename
      ? entries
          .filter((entry) => entry.isFile() && entry.name.includes(filename))
          .map((entry) => join(cacheDir, entry.name))
      : [];

    if (hfRef && filename) {
      const etagPath = join(cacheDir, `${filename}.etag`);
      const remoteEtag = await getRemoteEtag(hfRef);
      const localEtag = existsSync(etagPath)
        ? readFileSync(etagPath, "utf-8").trim()
        : null;
      const shouldRefresh =
        options.refresh || !remoteEtag || remoteEtag !== localEtag || cached.length === 0;

      if (shouldRefresh) {
        for (const candidate of cached) {
          if (existsSync(candidate)) unlinkSync(candidate);
        }
        if (existsSync(etagPath)) unlinkSync(etagPath);
        refreshed = cached.length > 0;
      }
    } else if (options.refresh && filename) {
      for (const candidate of cached) {
        if (existsSync(candidate)) unlinkSync(candidate);
        refreshed = true;
      }
    }

    const path = await resolveModelFile(model, cacheDir);
    const sizeBytes = existsSync(path) ? statSync(path).size : 0;
    if (hfRef && filename) {
      const remoteEtag = await getRemoteEtag(hfRef);
      if (remoteEtag) {
        const etagPath = join(cacheDir, `${filename}.etag`);
        writeFileSync(etagPath, remoteEtag + "\n", "utf-8");
      }
    }
    results.push({ model, path, sizeBytes, refreshed });
  }
  return results;
}

// =============================================================================
// LLM Interface
// =============================================================================

/**
 * Abstract LLM interface - implement this for different backends
 */
export interface LLM {
  /**
   * Get embeddings for text
   */
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null>;

  /**
   * Generate text completion
   */
  generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null>;

  /**
   * Check if a model exists/is available
   */
  modelExists(model: string): Promise<ModelInfo>;

  /**
   * Expand a search query into multiple variations for different backends.
   * Returns a list of Queryable objects.
   */
  expandQuery(query: string, options?: { context?: string, includeLexical?: boolean }): Promise<Queryable[]>;

  /**
   * Rerank documents by relevance to a query
   * Returns list of documents with relevance scores (higher = more relevant)
   */
  rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult>;

  /**
   * Dispose of resources
   */
  dispose(): Promise<void>;
}

// =============================================================================
// node-llama-cpp Implementation
// =============================================================================

export type LlamaCppConfig = {
  embedModel?: string;
  generateModel?: string;
  rerankModel?: string;
  embedApi?: ApiModelRuntimeConfig | null;
  queryApi?: ApiModelRuntimeConfig | null;
  rerankApi?: ApiModelRuntimeConfig | null;
  modelCacheDir?: string;
  /**
   * Inactivity timeout in ms before unloading contexts (default: 2 minutes, 0 to disable).
   *
   * Per node-llama-cpp lifecycle guidance, we prefer keeping models loaded and only disposing
   * contexts when idle, since contexts (and their sequences) are the heavy per-session objects.
   * @see https://node-llama-cpp.withcat.ai/guide/objects-lifecycle
   */
  inactivityTimeoutMs?: number;
  /**
   * Whether to dispose models on inactivity (default: false).
   *
   * Keeping models loaded avoids repeated VRAM thrash; set to true only if you need aggressive
   * memory reclaim.
   */
  disposeModelsOnInactivity?: boolean;
};

/**
 * LLM implementation using node-llama-cpp
 */
// Default inactivity timeout: 5 minutes (keep models warm during typical search sessions)
const DEFAULT_INACTIVITY_TIMEOUT_MS = 5 * 60 * 1000;

export class LlamaCpp implements LLM {
  private llama: Llama | null = null;
  private embedModel: LlamaModel | null = null;
  private embedContext: LlamaEmbeddingContext | null = null;
  private generateModel: LlamaModel | null = null;
  private rerankModel: LlamaModel | null = null;
  private rerankContext: Awaited<ReturnType<LlamaModel["createRankingContext"]>> | null = null;

  private embedModelUri: string;
  private generateModelUri: string;
  private rerankModelUri: string;
  private modelCacheDir: string;
  private embedApi: ApiModelRuntimeConfig | null;
  private queryApi: ApiModelRuntimeConfig | null;
  private rerankApi: ApiModelRuntimeConfig | null;

  // Ensure we don't load the same model/context concurrently (which can allocate duplicate VRAM).
  private embedModelLoadPromise: Promise<LlamaModel> | null = null;
  private embedContextCreatePromise: Promise<LlamaEmbeddingContext> | null = null;
  private generateModelLoadPromise: Promise<LlamaModel> | null = null;
  private rerankModelLoadPromise: Promise<LlamaModel> | null = null;

  // Inactivity timer for auto-unloading models
  private inactivityTimer: ReturnType<typeof setTimeout> | null = null;
  private inactivityTimeoutMs: number;
  private disposeModelsOnInactivity: boolean;

  // Track disposal state to prevent double-dispose
  private disposed = false;


  constructor(config: LlamaCppConfig = {}) {
    this.embedModelUri = config.embedModel || DEFAULT_EMBED_MODEL;
    this.generateModelUri = config.generateModel || DEFAULT_GENERATE_MODEL;
    this.rerankModelUri = config.rerankModel || DEFAULT_RERANK_MODEL;
    this.embedApi = config.embedApi || null;
    this.queryApi = config.queryApi || null;
    this.rerankApi = config.rerankApi || null;
    this.modelCacheDir = config.modelCacheDir || MODEL_CACHE_DIR;
    this.inactivityTimeoutMs = config.inactivityTimeoutMs ?? DEFAULT_INACTIVITY_TIMEOUT_MS;
    this.disposeModelsOnInactivity = config.disposeModelsOnInactivity ?? false;
  }

  /**
   * Reset the inactivity timer. Called after each model operation.
   * When timer fires, models are unloaded to free memory (if no active sessions).
   */
  private touchActivity(): void {
    // Clear existing timer
    if (this.inactivityTimer) {
      clearTimeout(this.inactivityTimer);
      this.inactivityTimer = null;
    }

    // Only set timer if we have disposable contexts and timeout is enabled
    if (this.inactivityTimeoutMs > 0 && this.hasLoadedContexts()) {
      this.inactivityTimer = setTimeout(() => {
        // Check if session manager allows unloading
        // canUnloadLLM is defined later in this file - it checks the session manager
        // We use dynamic import pattern to avoid circular dependency issues
        if (typeof canUnloadLLM === 'function' && !canUnloadLLM()) {
          // Active sessions/operations - reschedule timer
          this.touchActivity();
          return;
        }
        this.unloadIdleResources().catch(err => {
          console.error("Error unloading idle resources:", err);
        });
      }, this.inactivityTimeoutMs);
      // Don't keep process alive just for this timer
      this.inactivityTimer.unref();
    }
  }

  /**
   * Check if any contexts are currently loaded (and therefore worth unloading on inactivity).
   */
  private hasLoadedContexts(): boolean {
    return !!(this.embedContext || this.rerankContext);
  }

  /**
   * Unload idle resources but keep the instance alive for future use.
   *
   * By default, this disposes contexts (and their dependent sequences), while keeping models loaded.
   * This matches the intended lifecycle: model → context → sequence, where contexts are per-session.
   */
  async unloadIdleResources(): Promise<void> {
    // Don't unload if already disposed
    if (this.disposed) {
      return;
    }

    // Clear timer
    if (this.inactivityTimer) {
      clearTimeout(this.inactivityTimer);
      this.inactivityTimer = null;
    }

    // Dispose contexts first
    if (this.embedContext) {
      await this.embedContext.dispose();
      this.embedContext = null;
    }
    if (this.rerankContext) {
      await this.rerankContext.dispose();
      this.rerankContext = null;
    }

    // Optionally dispose models too (opt-in)
    if (this.disposeModelsOnInactivity) {
      if (this.embedModel) {
        await this.embedModel.dispose();
        this.embedModel = null;
      }
      if (this.generateModel) {
        await this.generateModel.dispose();
        this.generateModel = null;
      }
      if (this.rerankModel) {
        await this.rerankModel.dispose();
        this.rerankModel = null;
      }
      // Reset load promises so models can be reloaded later
      this.embedModelLoadPromise = null;
      this.generateModelLoadPromise = null;
      this.rerankModelLoadPromise = null;
    }

    // Note: We keep llama instance alive - it's lightweight
  }

  /**
   * Ensure model cache directory exists
   */
  private ensureModelCacheDir(): void {
    if (!existsSync(this.modelCacheDir)) {
      mkdirSync(this.modelCacheDir, { recursive: true });
    }
  }

  /**
   * Initialize the llama instance (lazy)
   */
  private async ensureLlama(): Promise<Llama> {
    if (!this.llama) {
      this.llama = await getLlama({ logLevel: LlamaLogLevel.error });
    }
    return this.llama;
  }

  /**
   * Resolve a model URI to a local path, downloading if needed
   */
  private async resolveModel(modelUri: string): Promise<string> {
    this.ensureModelCacheDir();
    // resolveModelFile handles HF URIs and downloads to the cache dir
    return await resolveModelFile(modelUri, this.modelCacheDir);
  }

  /**
   * Load embedding model (lazy)
   */
  private async ensureEmbedModel(): Promise<LlamaModel> {
    if (this.embedModel) {
      return this.embedModel;
    }
    if (this.embedModelLoadPromise) {
      return await this.embedModelLoadPromise;
    }

    this.embedModelLoadPromise = (async () => {
      const llama = await this.ensureLlama();
      const modelPath = await this.resolveModel(this.embedModelUri);
      const model = await llama.loadModel({ modelPath });
      this.embedModel = model;
      // Model loading counts as activity - ping to keep alive
      this.touchActivity();
      return model;
    })();

    try {
      return await this.embedModelLoadPromise;
    } finally {
      // Keep the resolved model cached; clear only the in-flight promise.
      this.embedModelLoadPromise = null;
    }
  }

  /**
   * Load embedding context (lazy). Context can be disposed and recreated without reloading the model.
   * Uses promise guard to prevent concurrent context creation race condition.
   */
  private async ensureEmbedContext(): Promise<LlamaEmbeddingContext> {
    if (!this.embedContext) {
      // If context creation is already in progress, wait for it
      if (this.embedContextCreatePromise) {
        return await this.embedContextCreatePromise;
      }

      // Start context creation and store promise so concurrent calls wait
      this.embedContextCreatePromise = (async () => {
        const model = await this.ensureEmbedModel();
        const context = await model.createEmbeddingContext();
        this.embedContext = context;
        return context;
      })();

      try {
        const context = await this.embedContextCreatePromise;
        this.touchActivity();
        return context;
      } finally {
        this.embedContextCreatePromise = null;
      }
    }
    this.touchActivity();
    return this.embedContext;
  }

  /**
   * Load generation model (lazy) - context is created fresh per call
   */
  private async ensureGenerateModel(): Promise<LlamaModel> {
    if (!this.generateModel) {
      if (this.generateModelLoadPromise) {
        return await this.generateModelLoadPromise;
      }

      this.generateModelLoadPromise = (async () => {
        const llama = await this.ensureLlama();
        const modelPath = await this.resolveModel(this.generateModelUri);
        const model = await llama.loadModel({ modelPath });
        this.generateModel = model;
        return model;
      })();

      try {
        await this.generateModelLoadPromise;
      } finally {
        this.generateModelLoadPromise = null;
      }
    }
    this.touchActivity();
    if (!this.generateModel) {
      throw new Error("Generate model not loaded");
    }
    return this.generateModel;
  }

  /**
   * Load rerank model (lazy)
   */
  private async ensureRerankModel(): Promise<LlamaModel> {
    if (this.rerankModel) {
      return this.rerankModel;
    }
    if (this.rerankModelLoadPromise) {
      return await this.rerankModelLoadPromise;
    }

    this.rerankModelLoadPromise = (async () => {
      const llama = await this.ensureLlama();
      const modelPath = await this.resolveModel(this.rerankModelUri);
      const model = await llama.loadModel({ modelPath });
      this.rerankModel = model;
      // Model loading counts as activity - ping to keep alive
      this.touchActivity();
      return model;
    })();

    try {
      return await this.rerankModelLoadPromise;
    } finally {
      this.rerankModelLoadPromise = null;
    }
  }

  /**
   * Load rerank context (lazy). Context can be disposed and recreated without reloading the model.
   */
  private async ensureRerankContext(): Promise<Awaited<ReturnType<LlamaModel["createRankingContext"]>>> {
    if (!this.rerankContext) {
      const model = await this.ensureRerankModel();
      this.rerankContext = await model.createRankingContext();
    }
    this.touchActivity();
    return this.rerankContext;
  }

  private buildApiUrl(baseUrl: string, path: string): string {
    const normalizedBase = baseUrl.replace(/\/+$/, "");
    const normalizedPath = path.replace(/^\/+/, "");
    return `${normalizedBase}/${normalizedPath}`;
  }

  private async postApiJson(
    url: string,
    apiConfig: ApiModelRuntimeConfig,
    payload: Record<string, unknown>
  ): Promise<unknown> {
    const authHeader = apiConfig.key.toLowerCase().startsWith("bearer ")
      ? apiConfig.key
      : `Bearer ${apiConfig.key}`;
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: authHeader,
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const body = await response.text().catch(() => "");
      const snippet = body ? ` - ${body.slice(0, 300)}` : "";
      throw new Error(`API request failed (${response.status} ${response.statusText})${snippet}`);
    }

    try {
      return await response.json();
    } catch {
      throw new Error("API returned non-JSON response");
    }
  }

  private parseQueryablesFromText(rawText: string, query: string, includeLexical: boolean): Queryable[] {
    const lines = rawText.trim().split("\n");
    const queryTerms = query
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter(Boolean);

    const hasQueryTerm = (text: string): boolean => {
      const lower = text.toLowerCase();
      if (queryTerms.length === 0) return true;
      return queryTerms.some(term => lower.includes(term));
    };

    const queryables: Queryable[] = lines
      .map(line => {
        const colonIdx = line.indexOf(":");
        if (colonIdx === -1) return null;
        const type = line.slice(0, colonIdx).trim();
        if (type !== "lex" && type !== "vec" && type !== "hyde") return null;
        const text = line.slice(colonIdx + 1).trim();
        if (!text || !hasQueryTerm(text)) return null;
        return { type: type as QueryType, text };
      })
      .filter((q): q is Queryable => q !== null);

    return includeLexical ? queryables : queryables.filter(q => q.type !== "lex");
  }

  private buildExpandFallback(query: string, includeLexical: boolean): Queryable[] {
    const fallback: Queryable[] = [
      { type: "hyde", text: `Information about ${query}` },
      { type: "lex", text: query },
      { type: "vec", text: query },
    ];
    return includeLexical ? fallback : fallback.filter(q => q.type !== "lex");
  }

  private buildExpandErrorFallback(query: string, includeLexical: boolean): Queryable[] {
    const fallback: Queryable[] = [{ type: "vec", text: query }];
    if (includeLexical) {
      fallback.unshift({ type: "lex", text: query });
    }
    return fallback;
  }

  private async embedWithApi(text: string): Promise<EmbeddingResult> {
    if (!this.embedApi) {
      throw new Error("Embed API config is missing");
    }
    const payload = await this.postApiJson(
      this.buildApiUrl(this.embedApi.baseUrl, "/embeddings"),
      this.embedApi,
      {
        model: this.embedApi.model,
        input: text,
      }
    ) as { data?: Array<{ embedding?: unknown[] }> };

    const rawEmbedding = payload.data?.[0]?.embedding;
    if (!Array.isArray(rawEmbedding)) {
      throw new Error("Embedding API response missing data[0].embedding");
    }
    const embedding = rawEmbedding.map((value) => Number(value));
    if (embedding.some((value) => !Number.isFinite(value))) {
      throw new Error("Embedding API returned non-numeric embedding values");
    }

    return {
      embedding,
      model: this.embedApi.model,
    };
  }

  private extractCompletionText(response: unknown): string | null {
    const data = response as {
      choices?: Array<{
        text?: unknown;
        message?: { content?: unknown };
      }>;
    };
    const choice = data.choices?.[0];
    if (!choice) return null;
    if (typeof choice.text === "string") return choice.text;

    const content = choice.message?.content;
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
      const merged = content
        .map((part) => {
          if (typeof part === "string") return part;
          if (part && typeof part === "object" && "text" in part) {
            const text = (part as { text?: unknown }).text;
            return typeof text === "string" ? text : "";
          }
          return "";
        })
        .join("");
      return merged || null;
    }
    return null;
  }

  private async generateWithApi(prompt: string, options: GenerateOptions = {}): Promise<GenerateResult> {
    if (!this.queryApi) {
      throw new Error("Query API config is missing");
    }
    const payload = await this.postApiJson(
      this.buildApiUrl(this.queryApi.baseUrl, "/chat/completions"),
      this.queryApi,
      {
        model: this.queryApi.model,
        messages: [{ role: "user", content: prompt }],
        temperature: options.temperature ?? 0.7,
        max_tokens: options.maxTokens ?? 150,
      }
    );

    const text = this.extractCompletionText(payload);
    if (!text) {
      throw new Error("Chat completion API response missing choices[0].message.content");
    }

    return {
      text,
      model: this.queryApi.model,
      done: true,
    };
  }

  private async rerankWithApi(query: string, documents: RerankDocument[]): Promise<RerankResult> {
    if (!this.rerankApi) {
      throw new Error("Rerank API config is missing");
    }
    if (documents.length === 0) {
      return { results: [], model: this.rerankApi.model };
    }

    const payload = await this.postApiJson(
      this.buildApiUrl(this.rerankApi.baseUrl, "/rerank"),
      this.rerankApi,
      {
        model: this.rerankApi.model,
        query,
        documents: documents.map((doc) => doc.text),
        top_n: documents.length,
        return_documents: false,
      }
    ) as {
      results?: Array<{ index?: number; score?: number; relevance_score?: number; document_index?: number }>;
      data?: Array<{ index?: number; score?: number; relevance_score?: number; document_index?: number }>;
      scores?: number[];
    };

    const entries = Array.isArray(payload.results)
      ? payload.results
      : (Array.isArray(payload.data) ? payload.data : []);

    let results: RerankDocumentResult[] = entries
      .map((entry) => {
        const index = typeof entry.index === "number"
          ? entry.index
          : (typeof entry.document_index === "number" ? entry.document_index : -1);
        const score = typeof entry.relevance_score === "number"
          ? entry.relevance_score
          : (typeof entry.score === "number" ? entry.score : NaN);
        if (index < 0 || index >= documents.length || !Number.isFinite(score)) {
          return null;
        }
        return {
          file: documents[index]!.file,
          score,
          index,
        };
      })
      .filter((item): item is RerankDocumentResult => item !== null);

    if (results.length === 0 && Array.isArray(payload.scores)) {
      results = payload.scores
        .map((score, index) => {
          if (!Number.isFinite(score) || index >= documents.length) return null;
          return {
            file: documents[index]!.file,
            score,
            index,
          };
        })
        .filter((item): item is RerankDocumentResult => item !== null);
    }

    if (results.length === 0) {
      throw new Error("Rerank API response missing valid ranked results");
    }

    results.sort((a, b) => b.score - a.score);
    return {
      results,
      model: this.rerankApi.model,
    };
  }

  // ==========================================================================
  // Tokenization
  // ==========================================================================

  /**
   * Tokenize text using the embedding model's tokenizer
   * Returns tokenizer tokens (opaque type from node-llama-cpp)
   */
  async tokenize(text: string): Promise<readonly LlamaToken[]> {
    await this.ensureEmbedContext();  // Ensure model is loaded
    if (!this.embedModel) {
      throw new Error("Embed model not loaded");
    }
    return this.embedModel.tokenize(text);
  }

  /**
   * Count tokens in text using the embedding model's tokenizer
   */
  async countTokens(text: string): Promise<number> {
    const tokens = await this.tokenize(text);
    return tokens.length;
  }

  /**
   * Detokenize token IDs back to text
   */
  async detokenize(tokens: readonly LlamaToken[]): Promise<string> {
    await this.ensureEmbedContext();
    if (!this.embedModel) {
      throw new Error("Embed model not loaded");
    }
    return this.embedModel.detokenize(tokens);
  }

  // ==========================================================================
  // Core API methods
  // ==========================================================================

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    // Ping activity at start to keep models alive during this operation
    this.touchActivity();

    try {
      if (this.embedApi) {
        return await this.embedWithApi(text);
      }

      const context = await this.ensureEmbedContext();
      const embedding = await context.getEmbeddingFor(text);

      return {
        embedding: Array.from(embedding.vector),
        model: this.embedModelUri,
      };
    } catch (error) {
      console.error("Embedding error:", error);
      return null;
    }
  }

  /**
   * Batch embed multiple texts efficiently
   * Uses Promise.all for parallel embedding - node-llama-cpp handles batching internally
   */
  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    // Ping activity at start to keep models alive during this operation
    this.touchActivity();

    if (texts.length === 0) return [];

    if (this.embedApi) {
      return await Promise.all(
        texts.map(async (text) => {
          try {
            return await this.embedWithApi(text);
          } catch (error) {
            console.error("Embedding API error for text:", error);
            return null;
          }
        })
      );
    }

    try {
      const context = await this.ensureEmbedContext();

      // node-llama-cpp handles batching internally when we make parallel requests
      const embeddings = await Promise.all(
        texts.map(async (text) => {
          try {
            const embedding = await context.getEmbeddingFor(text);
            this.touchActivity();  // Keep-alive during slow batches
            return {
              embedding: Array.from(embedding.vector),
              model: this.embedModelUri,
            };
          } catch (err) {
            console.error("Embedding error for text:", err);
            return null;
          }
        })
      );

      return embeddings;
    } catch (error) {
      console.error("Batch embedding error:", error);
      return texts.map(() => null);
    }
  }

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerateResult | null> {
    // Ping activity at start to keep models alive during this operation
    this.touchActivity();

    if (this.queryApi) {
      try {
        return await this.generateWithApi(prompt, options);
      } catch (error) {
        console.error("Generate API error:", error);
        return null;
      }
    }

    // Ensure model is loaded
    await this.ensureGenerateModel();

    // Create fresh context -> sequence -> session for each call
    const context = await this.generateModel!.createContext();
    const sequence = context.getSequence();
    const session = new LlamaChatSession({ contextSequence: sequence });

    const maxTokens = options.maxTokens ?? 150;
    // Qwen3 recommends temp=0.7, topP=0.8, topK=20 for non-thinking mode
    // DO NOT use greedy decoding (temp=0) - causes repetition loops
    const temperature = options.temperature ?? 0.7;

    let result = "";
    try {
      await session.prompt(prompt, {
        maxTokens,
        temperature,
        topK: 20,
        topP: 0.8,
        onTextChunk: (text) => {
          result += text;
        },
      });

      return {
        text: result,
        model: this.generateModelUri,
        done: true,
      };
    } finally {
      // Dispose context (which disposes dependent sequences/sessions per lifecycle rules)
      await context.dispose();
    }
  }

  async modelExists(modelUri: string): Promise<ModelInfo> {
    // For HuggingFace URIs, we assume they exist
    // For local paths, check if file exists
    if (modelUri.startsWith("hf:")) {
      return { name: modelUri, exists: true };
    }

    const exists = existsSync(modelUri);
    return {
      name: modelUri,
      exists,
      path: exists ? modelUri : undefined,
    };
  }

  // ==========================================================================
  // High-level abstractions
  // ==========================================================================

  async expandQuery(query: string, options: { context?: string, includeLexical?: boolean } = {}): Promise<Queryable[]> {
    // Ping activity at start to keep models alive during this operation
    this.touchActivity();

    const includeLexical = options.includeLexical ?? true;
    const contextHint = options.context?.trim();

    if (this.queryApi) {
      const prompt = [
        "Expand this search query into variants for retrieval.",
        "Return one item per line in exactly this format: <type>: <text>",
        includeLexical
          ? 'Allowed <type>: lex, vec, hyde'
          : 'Allowed <type>: vec, hyde',
        "Do not output extra commentary, bullets, markdown, or numbering.",
        contextHint ? `Additional context: ${contextHint}` : "",
        `Query: ${query}`,
      ].filter(Boolean).join("\n");

      try {
        const generated = await this.generateWithApi(prompt, {
          maxTokens: 600,
          temperature: 0.3,
        });
        const parsed = this.parseQueryablesFromText(generated.text, query, includeLexical);
        if (parsed.length > 0) return parsed;
        return this.buildExpandFallback(query, includeLexical);
      } catch (error) {
        console.error("API query expansion failed:", error);
        return this.buildExpandErrorFallback(query, includeLexical);
      }
    }

    const llama = await this.ensureLlama();
    await this.ensureGenerateModel();

    const grammar = await llama.createGrammar({
      grammar: `
        root ::= line+
        line ::= type ": " content "\\n"
        type ::= "lex" | "vec" | "hyde"
        content ::= [^\\n]+
      `
    });

    const prompt = `/no_think Expand this search query: ${query}`;

    // Create fresh context for each call
    const genContext = await this.generateModel!.createContext();
    const sequence = genContext.getSequence();
    const session = new LlamaChatSession({ contextSequence: sequence });

    try {
      // Qwen3 recommended settings for non-thinking mode:
      // temp=0.7, topP=0.8, topK=20, presence_penalty for repetition
      // DO NOT use greedy decoding (temp=0) - causes infinite loops
      const result = await session.prompt(prompt, {
        grammar,
        maxTokens: 600,
        temperature: 0.7,
        topK: 20,
        topP: 0.8,
        repeatPenalty: {
          lastTokens: 64,
          presencePenalty: 0.5,
        },
      });

      const parsed = this.parseQueryablesFromText(result, query, includeLexical);
      if (parsed.length > 0) return parsed;
      return this.buildExpandFallback(query, includeLexical);
    } catch (error) {
      console.error("Structured query expansion failed:", error);
      return this.buildExpandErrorFallback(query, includeLexical);
    } finally {
      await genContext.dispose();
    }
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options: RerankOptions = {}
  ): Promise<RerankResult> {
    // Ping activity at start to keep models alive during this operation
    this.touchActivity();

    if (this.rerankApi) {
      return await this.rerankWithApi(query, documents);
    }

    const context = await this.ensureRerankContext();

    // Build a map from document text to original indices (for lookup after sorting)
    const textToDoc = new Map<string, { file: string; index: number }>();
    documents.forEach((doc, index) => {
      textToDoc.set(doc.text, { file: doc.file, index });
    });

    // Extract just the text for ranking
    const texts = documents.map((doc) => doc.text);

    // Use the proper ranking API - returns [{document: string, score: number}] sorted by score
    const ranked = await context.rankAndSort(query, texts);

    // Map back to our result format using the text-to-doc map
    const results: RerankDocumentResult[] = ranked.map((item) => {
      const docInfo = textToDoc.get(item.document)!;
      return {
        file: docInfo.file,
        score: item.score,
        index: docInfo.index,
      };
    });

    return {
      results,
      model: this.rerankModelUri,
    };
  }

  async dispose(): Promise<void> {
    // Prevent double-dispose
    if (this.disposed) {
      return;
    }
    this.disposed = true;

    // Clear inactivity timer
    if (this.inactivityTimer) {
      clearTimeout(this.inactivityTimer);
      this.inactivityTimer = null;
    }

    // Disposing llama cascades to models and contexts automatically
    // See: https://node-llama-cpp.withcat.ai/guide/objects-lifecycle
    // Note: llama.dispose() can hang indefinitely, so we use a timeout
    if (this.llama) {
      const disposePromise = this.llama.dispose();
      const timeoutPromise = new Promise<void>((resolve) => setTimeout(resolve, 1000));
      await Promise.race([disposePromise, timeoutPromise]);
    }

    // Clear references
    this.embedContext = null;
    this.rerankContext = null;
    this.embedModel = null;
    this.generateModel = null;
    this.rerankModel = null;
    this.llama = null;

    // Clear any in-flight load/create promises
    this.embedModelLoadPromise = null;
    this.embedContextCreatePromise = null;
    this.generateModelLoadPromise = null;
    this.rerankModelLoadPromise = null;
  }
}

// =============================================================================
// Session Management Layer
// =============================================================================

/**
 * Manages LLM session lifecycle with reference counting.
 * Coordinates with LlamaCpp idle timeout to prevent disposal during active sessions.
 */
class LLMSessionManager {
  private llm: LlamaCpp;
  private _activeSessionCount = 0;
  private _inFlightOperations = 0;

  constructor(llm: LlamaCpp) {
    this.llm = llm;
  }

  get activeSessionCount(): number {
    return this._activeSessionCount;
  }

  get inFlightOperations(): number {
    return this._inFlightOperations;
  }

  /**
   * Returns true only when both session count and in-flight operations are 0.
   * Used by LlamaCpp to determine if idle unload is safe.
   */
  canUnload(): boolean {
    return this._activeSessionCount === 0 && this._inFlightOperations === 0;
  }

  acquire(): void {
    this._activeSessionCount++;
  }

  release(): void {
    this._activeSessionCount = Math.max(0, this._activeSessionCount - 1);
  }

  operationStart(): void {
    this._inFlightOperations++;
  }

  operationEnd(): void {
    this._inFlightOperations = Math.max(0, this._inFlightOperations - 1);
  }

  getLlamaCpp(): LlamaCpp {
    return this.llm;
  }
}

/**
 * Error thrown when an operation is attempted on a released or aborted session.
 */
export class SessionReleasedError extends Error {
  constructor(message = "LLM session has been released or aborted") {
    super(message);
    this.name = "SessionReleasedError";
  }
}

/**
 * Scoped LLM session with automatic lifecycle management.
 * Wraps LlamaCpp methods with operation tracking and abort handling.
 */
class LLMSession implements ILLMSession {
  private manager: LLMSessionManager;
  private released = false;
  private abortController: AbortController;
  private maxDurationTimer: ReturnType<typeof setTimeout> | null = null;
  private name: string;

  constructor(manager: LLMSessionManager, options: LLMSessionOptions = {}) {
    this.manager = manager;
    this.name = options.name || "unnamed";
    this.abortController = new AbortController();

    // Link external abort signal if provided
    if (options.signal) {
      if (options.signal.aborted) {
        this.abortController.abort(options.signal.reason);
      } else {
        options.signal.addEventListener("abort", () => {
          this.abortController.abort(options.signal!.reason);
        }, { once: true });
      }
    }

    // Set up max duration timer
    const maxDuration = options.maxDuration ?? 10 * 60 * 1000; // Default 10 minutes
    if (maxDuration > 0) {
      this.maxDurationTimer = setTimeout(() => {
        this.abortController.abort(new Error(`Session "${this.name}" exceeded max duration of ${maxDuration}ms`));
      }, maxDuration);
      this.maxDurationTimer.unref(); // Don't keep process alive
    }

    // Acquire session lease
    this.manager.acquire();
  }

  get isValid(): boolean {
    return !this.released && !this.abortController.signal.aborted;
  }

  get signal(): AbortSignal {
    return this.abortController.signal;
  }

  /**
   * Release the session and decrement ref count.
   * Called automatically by withLLMSession when the callback completes.
   */
  release(): void {
    if (this.released) return;
    this.released = true;

    if (this.maxDurationTimer) {
      clearTimeout(this.maxDurationTimer);
      this.maxDurationTimer = null;
    }

    this.abortController.abort(new Error("Session released"));
    this.manager.release();
  }

  /**
   * Wrap an operation with tracking and abort checking.
   */
  private async withOperation<T>(fn: () => Promise<T>): Promise<T> {
    if (!this.isValid) {
      throw new SessionReleasedError();
    }

    this.manager.operationStart();
    try {
      // Check abort before starting
      if (this.abortController.signal.aborted) {
        throw new SessionReleasedError(
          this.abortController.signal.reason?.message || "Session aborted"
        );
      }
      return await fn();
    } finally {
      this.manager.operationEnd();
    }
  }

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    return this.withOperation(() => this.manager.getLlamaCpp().embed(text, options));
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    return this.withOperation(() => this.manager.getLlamaCpp().embedBatch(texts));
  }

  async expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean }
  ): Promise<Queryable[]> {
    return this.withOperation(() => this.manager.getLlamaCpp().expandQuery(query, options));
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options?: RerankOptions
  ): Promise<RerankResult> {
    return this.withOperation(() => this.manager.getLlamaCpp().rerank(query, documents, options));
  }
}

// Session manager for the default LlamaCpp instance
let defaultSessionManager: LLMSessionManager | null = null;

/**
 * Get the session manager for the default LlamaCpp instance.
 */
function getSessionManager(): LLMSessionManager {
  const llm = getDefaultLlamaCpp();
  if (!defaultSessionManager || defaultSessionManager.getLlamaCpp() !== llm) {
    defaultSessionManager = new LLMSessionManager(llm);
  }
  return defaultSessionManager;
}

/**
 * Execute a function with a scoped LLM session.
 * The session provides lifecycle guarantees - resources won't be disposed mid-operation.
 *
 * @example
 * ```typescript
 * await withLLMSession(async (session) => {
 *   const expanded = await session.expandQuery(query);
 *   const embeddings = await session.embedBatch(texts);
 *   const reranked = await session.rerank(query, docs);
 *   return reranked;
 * }, { maxDuration: 10 * 60 * 1000, name: 'querySearch' });
 * ```
 */
export async function withLLMSession<T>(
  fn: (session: ILLMSession) => Promise<T>,
  options?: LLMSessionOptions
): Promise<T> {
  const manager = getSessionManager();
  const session = new LLMSession(manager, options);

  try {
    return await fn(session);
  } finally {
    session.release();
  }
}

/**
 * Check if idle unload is safe (no active sessions or operations).
 * Used internally by LlamaCpp idle timer.
 */
export function canUnloadLLM(): boolean {
  if (!defaultSessionManager) return true;
  return defaultSessionManager.canUnload();
}

// =============================================================================
// Singleton for default LlamaCpp instance
// =============================================================================

let defaultLlamaCpp: LlamaCpp | null = null;

function getDefaultLlamaCppConfig(): LlamaCppConfig {
  const apiConfigs = resolveApiModelConfigsFromYaml();
  return {
    embedApi: apiConfigs.embedApi,
    queryApi: apiConfigs.queryApi,
    rerankApi: apiConfigs.rerankApi,
  };
}

/**
 * Get the default LlamaCpp instance (creates one if needed)
 */
export function getDefaultLlamaCpp(): LlamaCpp {
  if (!defaultLlamaCpp) {
    defaultLlamaCpp = new LlamaCpp(getDefaultLlamaCppConfig());
  }
  return defaultLlamaCpp;
}

/**
 * Set a custom default LlamaCpp instance (useful for testing)
 */
export function setDefaultLlamaCpp(llm: LlamaCpp | null): void {
  defaultLlamaCpp = llm;
}

/**
 * Dispose the default LlamaCpp instance if it exists.
 * Call this before process exit to prevent NAPI crashes.
 */
export async function disposeDefaultLlamaCpp(): Promise<void> {
  if (defaultLlamaCpp) {
    await defaultLlamaCpp.dispose();
    defaultLlamaCpp = null;
  }
}
