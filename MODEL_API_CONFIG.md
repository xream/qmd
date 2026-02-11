# QMD API Model Configuration

QMD can route each model capability to an API model configured in `~/.config/qmd/index.yml`.

If a capability (`embed`, `query`, or `rerank`) has all three fields configured (`base_url`, `key`, `model`), QMD uses the API for that capability.

If a capability is not configured, QMD keeps using the existing local GGUF model.

## index.yml example

```yaml
global_context: "Your global context"

models:
  embed:
    base_url: "https://api.example.com/v1"
    key: "sk-..."
    model: "text-embedding-3-large"
  query:
    base_url: "https://api.example.com/v1"
    key: "sk-..."
    model: "gpt-4o-mini"
  rerank:
    base_url: "https://api.example.com/v1"
    key: "sk-..."
    model: "rerank-v3.5"

collections:
  notes:
    path: ~/Documents/Notes
    pattern: "**/*.md"
```

## API endpoints expected by QMD

- `embed` uses `POST <base_url>/embeddings`
- `query` uses `POST <base_url>/chat/completions`
- `rerank` uses `POST <base_url>/rerank`

QMD sends `Authorization: Bearer <key>` and `Content-Type: application/json`.

## Notes

- Configure capabilities independently; mixed mode is supported.
- Changes take effect when a new QMD process starts (restart daemon/server if running).
