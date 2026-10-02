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

## Why sqlite-vec, and its limitations

sqlite-vec fits this project's goal: zero infrastructure, a single file, and keyword + vector search in the same
database. That choice has trade-offs compared with popular or standalone vector databases (Qdrant, pgvector,
Pinecone, Weaviate, Milvus):

- **Exact (brute-force) search only.** `vec0` compares the query against every stored vector. There is no
  approximate-nearest-neighbour index (HNSW/IVF) in the 0.1.x releases, so query time grows linearly with corpus
  size. That is fine for thousands to low hundreds of thousands of chunks; beyond that, latency becomes a problem.
- **Pre-1.0 software.** The project is at 0.1.x, so APIs, the `vec0` syntax and the on-disk format may change between
  versions. The version is pinned in [scripts/fetch-sqlite-vec.ps1](scripts/fetch-sqlite-vec.ps1) and the Dockerfile.
- **Embedded, single-node.** SQLite lives in-process and has one writer at a time. There is no network server,
  replication, sharding, authentication or horizontal scaling, and several app instances cannot safely share one file.
- **Native extension to ship.** The loadable `vec0` binary must match the OS and CPU architecture and be present at
  runtime (`native/<rid>/`). A missing or mismatched binary is a startup-time failure, and some hosted or locked-down
  environments don't allow loading extensions at all.
- **Fixed embedding dimension.** Dimensions are set when the `vec_chunks` table is created. Changing the embedding
  model to one with a different size means deleting the database and re-ingesting.
- **Limited filtering and query features.** Filtering by metadata (for example restricting to one source file) is
  much weaker than in a dedicated engine, so filters are best applied after the vector search or by joining back to
  `chunks`. There is no built-in hybrid ranking, reranking, multi-vector support or namespaces/multi-tenancy;
  hybrid search here is something this project implements on top.
- **No managed operations.** Backups, index tuning, monitoring, upgrades and access control are all up to you.
  Standalone databases provide these out of the box.
- **Smaller ecosystem.** Fewer client libraries, tutorials and integrations than pgvector or the major vector
  databases, and little community knowledge of production tuning.

The vector store sits behind `IVectorStore`, so swapping in another backend later means writing one new
implementation in `DocRAG.Infrastructure`; Core, Ingestion and the API don't change.

## Setup

Prerequisites: .NET 9 SDK, an OpenAI API key.

```powershell
# 1. Fetch the sqlite-vec extension for Windows (one-time; Linux/Docker fetches its own)
./scripts/fetch-sqlite-vec.ps1

# 2. Provide your API key - either set the OPENAI_API_KEY environment variable, or use user secrets (never committed)
dotnet user-secrets set "OpenAI:ApiKey" "sk-..." --project src/DocRAG.Api

# 3. Run
dotnet run --project src/DocRAG.Api --launch-profile http     # http://localhost:5041
```

Key lookup order: `OpenAI:ApiKey` (user secrets or the `OpenAI__ApiKey` env var), then `OPENAI_API_KEY`. `appsettings.json` holds all other settings
(models, chunk sizes, similarity threshold, DB path); the database is created at `data/docrag.db`.

### Docker

```bash
docker compose up --build        # reads OPENAI_API_KEY from your shell or a git-ignored .env file
```

The API is then at `http://localhost:8080` (use that port in the examples below instead of 5041). The database
lives in the `docrag-data` volume, so ingested documents survive restarts; `docker compose down -v` wipes them.
The image downloads the Linux sqlite-vec build itself, so no local setup script is needed.

Without compose: `docker build -t docrag .` then
`docker run -p 8080:8080 -e OPENAI_API_KEY=sk-... -v docrag-data:/data docrag`.

## Try it

The easiest way is **Swagger UI**: open `http://localhost:5041/` (or `:8080` in Docker), which redirects to `/swagger`.
Use *Try it out* on `POST /documents` to pick a file, then on `POST /ask`. The raw OpenAPI document is at `/openapi/v1.json`.
Swagger is enabled in every environment because this is a local demo; put it behind a flag or auth before deploying anywhere public.

Or from the command line:

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
