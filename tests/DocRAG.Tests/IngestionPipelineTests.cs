using System.Text;
using DocRAG.Core;
using DocRAG.Infrastructure;
using DocRAG.Infrastructure.Sqlite;
using DocRAG.Ingestion;
using DocRAG.Tests.Fakes;
using Microsoft.Extensions.Options;

namespace DocRAG.Tests;

/// <summary>Ingest -> store -> retrieve end to end with the fake embedder (no network).</summary>
public sealed class IngestionPipelineTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "docrag-tests-" + Guid.NewGuid().ToString("N"));
    private readonly FakeEmbedder _embedder = new(64);
    private readonly SqliteVectorStore _store;
    private readonly IngestionService _ingestion;

    public IngestionPipelineTests()
    {
        _store = new SqliteVectorStore(Options.Create(new SqliteVectorStoreOptions
        {
            DatabasePath = Path.Combine(_dir, "test.db"),
            Dimensions = 64,
        }));
        var chunker = new Chunker(new WhitespaceTokenCounter(), new ChunkingOptions { TargetTokens = 60, MaxTokens = 80 });
        _ingestion = new IngestionService([new MarkdownParser()], chunker, _embedder, _store);
    }

    private static MemoryStream Md(string text) => new(Encoding.UTF8.GetBytes(text));

    private const string Handbook = """
        # Handbook

        ## Refunds
        Customers may request a refund within thirty days of purchase for any reason.

        ## Shipping
        Orders ship from the warehouse in Rotterdam and arrive within five business days.
        """;

    [Fact]
    public async Task Retrieval_finds_the_relevant_section_first()
    {
        await _ingestion.IngestAsync(Md(Handbook), "handbook.md");
        var retriever = new HybridRetriever(_embedder, _store, Options.Create(new RetrievalOptions()));

        var results = await retriever.RetrieveAsync("how many days for a refund purchase", topK: 2, mode: RetrievalSource.Vector);

        Assert.Equal(["Handbook", "Refunds"], results[0].Chunk.Metadata.HeadingPath);
        Assert.Equal(2, results.Count);
    }

    [Fact]
    public async Task Hybrid_retrieval_through_the_real_store_fuses_vector_and_keyword_hits()
    {
        await _ingestion.IngestAsync(Md(Handbook), "handbook.md");
        var retriever = new HybridRetriever(_embedder, _store, Options.Create(new RetrievalOptions()));

        var results = await retriever.RetrieveAsync("warehouse Rotterdam shipping", topK: 2, mode: RetrievalSource.Hybrid);

        Assert.Equal(["Handbook", "Shipping"], results[0].Chunk.Metadata.HeadingPath);
        Assert.All(results, r => Assert.Equal(RetrievalSource.Hybrid, r.Source));
        // Found by both searches: the cosine score must survive fusion for the answer gate.
        Assert.InRange(results[0].VectorScore!.Value, 0.1, 1.0);
    }

    [Fact]
    public async Task Reingesting_a_file_replaces_rather_than_duplicates()
    {
        await _ingestion.IngestAsync(Md(Handbook), "handbook.md");
        var second = await _ingestion.IngestAsync(Md(Handbook), "handbook.md");

        var hits = await _store.SearchAsync(_embedder.Embed("refund"), topK: 50);

        Assert.Equal(second.ChunkCount, hits.Count);
    }

    [Fact]
    public async Task Chunks_are_embedded_in_batches_in_one_call_per_document()
    {
        await _ingestion.IngestAsync(Md(Handbook), "handbook.md");

        var batch = Assert.Single(_embedder.BatchSizes);
        Assert.True(batch >= 2);
    }

    [Fact]
    public async Task Unsupported_extension_is_rejected()
    {
        await Assert.ThrowsAsync<NotSupportedException>(() => _ingestion.IngestAsync(Md("x"), "image.png"));
    }

    public void Dispose()
    {
        _store.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }
}
