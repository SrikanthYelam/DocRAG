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
page number (PDF) and heading path. PDF pages have no reliable headings, so consecutive pages are chunked as one
continuous flow: text and overlap carry across page breaks (a sentence that starts on one page and ends on the next
stays in one chunk) and each chunk reports the page its new text starts on. See [Chunking strategies](#chunking-strategies).

**Store**: one SQLite file with a `chunks` table, a `vec0` cosine index and an FTS5 keyword index (BM25) sharing rowids.
`SearchAsync` (vector) and `KeywordSearchAsync` are both on `IVectorStore`.

**Retrieval**: `/ask` takes an optional `mode` (`Vector`, `Keyword` or `Hybrid`; default `Retrieval:Mode`, which is `Hybrid`).
- *Vector*: the question is embedded with the same model as the chunks and the store returns the `topK` nearest chunks
  (default 5, max 20) by cosine similarity. sqlite-vec reports cosine distance and the store converts it to similarity
  (1.0 = identical, 0 = unrelated). It matches meaning rather than exact terms, so it can miss rare tokens such as
  error codes or names, and it always returns K results even when none are relevant.
- *Keyword*: FTS5 BM25 over chunk text and heading path.
- *Hybrid*: both searches run over a wider candidate pool (`Retrieval:CandidatePoolSize`, default 20) and are merged
  with Reciprocal Rank Fusion (`score = sum of 1 / (60 + rank)`), which uses ranks only because cosine and BM25 scores
  aren't comparable. Chunks found by both lists rise to the top; each result keeps its original cosine similarity in
  `vectorScore` (null if only the keyword search found it), because fused scores are not similarities.

When ingesting, each chunk's heading path is prepended to the text that gets embedded (not to the stored text).

**Answering**: chunks are tagged `[chunk:<id>]` in the prompt and the model must reply with JSON
(`sufficient`, `answer`, `citations`). Two guards yield `Status: "InsufficientContext"` (distinct from `Answered`,
with no answer text): if the best `vectorScore` among the retrieved chunks is below `Answer:MinSimilarity` the model isn't
called at all (so strong keyword matches alone can't override a weak semantic match; keyword-only mode has no vector
scores and is not gated), and the model may itself declare the context insufficient. Citations the model invents are dropped.
`Answer:MinSimilarity` defaults to 0.25; see [Evaluation findings](#evaluation-findings) for why it was lowered from 0.30.

## Chunking strategies

DocRAG has one chunker ([Chunker.cs](src/DocRAG.Ingestion/Chunker.cs)) for every file type. It is **recursive**
chunking, with an overlap borrowed from fixed-size chunking and section boundaries borrowed from document-based
chunking. The table lists the eight common strategies, whether each is used, and why.

| Strategy | Used? | Why |
| --- | --- | --- |
| **Recursive** | **Yes, the core** | Splits on paragraphs first, then sentences, then words, then characters, only when a piece is still over budget, and packs pieces up to 400 tokens. Deterministic, no extra API calls, and a good fit for the prose documents DocRAG ingests. |
| **Document-based** | **Partly** | The parsers mark the structural breaks: markdown headings (with the heading path, ignoring `#` inside code fences) and PDF pages. A chunk never crosses a heading. Inside a section, splitting follows the token budget, not the document's own markup (tables and code blocks are not treated specially). The heading path is stored as metadata and prepended to the text that is embedded. |
| **Fixed-size** | **Only the overlap** | Adjacent chunks in a section share ~12% of the budget, taken on whole-word boundaries. There are no pure fixed-length cuts, which ignore sentence boundaries. |
| **Semantic** | No | Needs an embedding call per sentence at ingestion, and the boundaries depend on a similarity threshold that needs tuning. Heading and paragraph boundaries already give topic-sized chunks for markdown, and no evaluation failure traced back to a missed topic shift. |
| **LLM-based (propositions)** | No | One LLM call per document, expensive across a large corpus. It also rewrites the text, so cited chunks would no longer quote the original wording. |
| **Agentic** | No | Only useful when the corpus mixes document types that need different splitters. DocRAG accepts markdown/text and PDF, the file extension already selects the parser, and the evaluation found nothing a per-file choice would improve. |
| **Late chunking** | No | Needs a long-context embedding model that exposes token-level embeddings. The OpenAI embeddings API used here returns one vector per input. |
| **Hierarchical** | No, not yet | Needs parent/child storage and retrieval. The heading-path prefix already gives each chunk some section context. It is the best next candidate if evaluation shows answers losing context, for example small chunks that need their section to be understood. |

Chunk size was measured, not assumed: 200-token chunks fixed one question but lowered ranking quality, so 400 stays
the default (see [Evaluation findings](#evaluation-findings)).

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

`/ask` returns `question`, `status`, `answer`, `sources` and `retrievedChunks`:

```json
{
  "question": "How many vacation days can I carry over?",
  "status": "Answered",
  "answer": "You can carry over up to 5 unused vacation days ...",
  "sources": [
    { "document": "acme-handbook.md", "chunkId": "b4b15a646a05", "page": null,
      "headingPath": "Acme Employee Handbook > Time Off > Vacation", "content": "Full-time employees accrue ..." }
  ],
  "retrievedChunks": [
    { "id": "b4b15a646a05", "score": 0.032, "vectorScore": 0.52, "source": "Hybrid", "sourceFile": "acme-handbook.md",
      "page": null, "headingPath": ["Acme Employee Handbook", "Time Off", "Vacation"], "text": "..." }
  ]
}
```

`sources` are the chunks the answer cites; `retrievedChunks` is everything retrieved, cited or not, with scores (useful for
debugging and evaluation). For unanswerable questions `status` is `InsufficientContext`, `answer` is null and `sources` is empty.
A question the documents can't answer returns `"status": "InsufficientContext"`. [samples/requests.http](samples/requests.http) has the same requests for the VS Code REST Client.

## Tests

```powershell
dotnet test
```

## Evaluation findings

These come from running the [EvalHarness](../EvalHarness) dataset `datasets/acme-hard.json` against this API. It has 25
questions over a 3-page PDF ([acme-travel-policy.pdf](../EvalHarness/samples/acme-travel-policy.pdf)) and a markdown
runbook with tables and a code block ([acme-engineering-runbook.md](../EvalHarness/samples/acme-engineering-runbook.md)):
two questions whose answers straddle a PDF page break, one needing two sources, table and code-block questions, and four
questions the documents do not answer (the API should decline them). Scores are temperature-0 and were identical across
repeated runs, so differences between settings are real, but the dataset is small (one question is 4 to 5 points).

**PDF page breaks (fixed).** The chunker used to treat every PDF page as a separate section, so a sentence running over a
page break was cut in two and no chunk contained it ("A loss that is | reported late may lead to disciplinary action").
Pages are now chunked as one flow (see [Chunking strategies](#chunking-strategies)). Two unit tests and the two page-break
questions fail without the fix and pass with it.

**Chunk size and similarity gate** (21 questions, 5 repeats each):

| Setting | Pass rate | Answer correctness | Recall@1 | Recall@3 / @5 |
| --- | --- | --- | --- | --- |
| 400 tokens, gate 0.30 (previous default) | 81.0% | 0.810 | 0.905 | 1.00 / 1.00 |
| 200 tokens, gate 0.30 | 85.7% | 0.838 | 0.762 | 1.00 / 1.00 |
| **400 tokens, gate 0.25 (current default)** | **85.7%** | **0.860** | **0.905** | **1.00 / 1.00** |

- Smaller chunks fixed one question (`pdf-hotel-london`) but pushed the right chunk out of first place for three more, so
  400 tokens stays.
- That question was actually a gate problem. PDFs have no headings, so the first chunk mixes the title, travel approval,
  booking and lodging, which dilutes its embedding: it scored 0.28 against the question, under the 0.30 gate, and the model
  was never called. Lowering `Answer:MinSimilarity` to 0.25 fixes it without touching chunk size.
- A lower gate risks answering questions the documents don't cover, so the dataset now has four must-decline questions
  (two unrelated, two near-misses such as on-call pay for a SEV1). All four are still declined at 0.25, because the model's
  own "sufficient" check backs up the gate. With them the dataset scores 22 of 25 (88.0%).

**Known issues, not fixed:**

- **Words fused at line ends in PDFs.** PdfPig's page text joins the last word of a line to the first word of the next
  (`ofapproval.`) when the PDF has no space glyph there. This is in the PDF parser, not the chunker; building the text from
  PdfPig's word extraction would likely fix it. The PDF test fixture ends each line with a space, so it does not exercise this.
- **Three questions still fail at every setting.** `md-rollback-migration` is declined by the model although the exact sentence
  is in the top chunk (cause not investigated); `pdf-device-wipe` omits "locked immediately"; `md-friday-release` misreads
  an ambiguously worded sentence.
- **Only four decline tests.** They check the gate, not every kind of out-of-scope question.

To reproduce, ingest both sample files through `POST /documents`, then from the EvalHarness folder run
`dotnet run --project src/EvalHarness.Cli -- run --dataset datasets/acme-hard.json --rag-url http://localhost:8080 --repeats 3`.
