# DocRAG

Retrieval-Augmented Generation over your own documents, in C#/.NET 9. Everything runs locally except the
OpenAI API calls (embeddings + generation). Vector storage is SQLite + [sqlite-vec](https://github.com/asg017/sqlite-vec).

## Architecture

```
DocRAG.Core             interfaces + records only, zero dependencies
DocRAG.Ingestion   ->   Core   markdown/PDF parsers, token-budgeted chunker, ingestion pipeline
DocRAG.Infrastructure -> Core  OpenAI embedder/generator (Microsoft.Extensions.AI), SQLite vector store
DocRAG.Api         ->   Ingestion + Infrastructure   minimal API host
tests/DocRAG.Tests      unit tests (fake embedder, no network)
tests/DocRAG.Eval       evaluation harness (placeholder, next increment)
```

Only Infrastructure talks to the network. Tests use `FakeEmbedder` (deterministic bag-of-words vectors) and a stub `IChatClient`.

**Chunking**: split by heading, then paragraph, then sentence/word, packed to a 400-token target (500 hard max,
o200k tokenizer) with ~12% overlap between adjacent chunks in the same section. Each chunk keeps source file,
page number (PDF) and heading path.

**Store**: one SQLite file with a `chunks` table, a `vec0` cosine index and an FTS5 keyword index (BM25) sharing rowids.
`SearchAsync` (vector) and `KeywordSearchAsync` are both on `IVectorStore`, ready to be fused for hybrid search.

**Answering**: chunks are tagged `[chunk:<id>]` in the prompt and the model must reply with JSON
(`sufficient`, `answer`, `citations`). Two guards yield `Status: "InsufficientContext"` (distinct from `Answered`,
with no answer text): if the best vector hit is below `Answer:MinSimilarity` the model isn't called at all, and the
model may itself declare the context insufficient. Citations the model invents are dropped.

## Setup

Prerequisites: .NET 9 SDK, an OpenAI API key.

```powershell
# 1. Fetch the sqlite-vec extension for Windows (one-time; Linux/Docker fetches its own)
./scripts/fetch-sqlite-vec.ps1

# 2. Store your API key in user secrets (never committed)
dotnet user-secrets set "OpenAI:ApiKey" "sk-..." --project src/DocRAG.Api

# 3. Run
dotnet run --project src/DocRAG.Api --launch-profile http     # http://localhost:5041
```

The key can also come from the `OpenAI__ApiKey` environment variable. `appsettings.json` holds all other settings
(models, chunk sizes, similarity threshold, DB path); the database is created at `data/docrag.db`.

### Docker

```bash
docker build -t docrag .
docker run -p 8080:8080 -e OpenAI__ApiKey=sk-... -v docrag-data:/data docrag
```

## Try it

```bash
# ingest (.md, .txt or .pdf)
curl -F "file=@samples/acme-handbook.md" http://localhost:5041/documents

# ask
curl -X POST http://localhost:5041/ask -H "Content-Type: application/json" \
     -d '{"question": "How many vacation days can I carry over?"}'
```

`/ask` returns `status`, `answer`, `citations` (chunk id, file, page, heading path) and the `retrievedChunks` with scores.
A question the documents can't answer returns `"status": "InsufficientContext"`. [samples/requests.http](samples/requests.http) has the same requests for the VS Code REST Client.

## Tests

```powershell
dotnet test
```
