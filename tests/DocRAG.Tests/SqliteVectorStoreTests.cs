using DocRAG.Core;
using DocRAG.Infrastructure.Sqlite;
using Microsoft.Extensions.Options;

namespace DocRAG.Tests;

public sealed class SqliteVectorStoreTests : IDisposable
{
    private const int Dims = 4;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "docrag-tests-" + Guid.NewGuid().ToString("N"));
    private readonly SqliteVectorStore _store;

    public SqliteVectorStoreTests()
    {
        _store = new SqliteVectorStore(Options.Create(new SqliteVectorStoreOptions
        {
            DatabasePath = Path.Combine(_dir, "test.db"),
            Dimensions = Dims,
        }));
    }

    private static DocumentChunk Chunk(string id, string text, float[] vector, string source = "a.md", int index = 0) =>
        new(id, text, text.Split(' ').Length, index, new ChunkMetadata(source, null, ["H1", "H2"]), vector);

    [Fact]
    public async Task Search_returns_top_k_ordered_by_cosine_similarity()
    {
        await _store.UpsertAsync([
            Chunk("exact", "exact", [1, 0, 0, 0]),
            Chunk("close", "close", [0.9f, 0.1f, 0, 0]),
            Chunk("far", "far", [0.2f, 0.9f, 0, 0]),
            Chunk("opposite", "opposite", [-1, 0, 0, 0]),
        ]);

        var results = await _store.SearchAsync(new float[] { 1, 0, 0, 0 }, topK: 3);

        Assert.Equal(["exact", "close", "far"], results.Select(r => r.Chunk.Id));
        Assert.Equal(1.0, results[0].Score, 4);
        Assert.True(results[0].Score > results[1].Score && results[1].Score > results[2].Score);
        Assert.All(results, r => Assert.Equal(RetrievalSource.Vector, r.Source));
    }

    [Fact]
    public async Task Search_score_is_cosine_similarity_not_distance()
    {
        await _store.UpsertAsync([Chunk("orth", "orth", [0, 1, 0, 0]), Chunk("opp", "opp", [-1, 0, 0, 0])]);

        var results = await _store.SearchAsync(new float[] { 1, 0, 0, 0 }, topK: 2);

        Assert.Equal(0.0, results.Single(r => r.Chunk.Id == "orth").Score, 4);
        Assert.Equal(-1.0, results.Single(r => r.Chunk.Id == "opp").Score, 4);
    }

    [Fact]
    public async Task Search_round_trips_chunk_content_and_metadata()
    {
        var original = new DocumentChunk("c1", "hello world", 2, 7, new ChunkMetadata("doc.pdf", 5, ["A", "B"]),
            new float[] { 1, 0, 0, 0 });
        await _store.UpsertAsync([original]);

        var hit = (await _store.SearchAsync(new float[] { 1, 0, 0, 0 }, 1)).Single().Chunk;

        Assert.Equal("hello world", hit.Text);
        Assert.Equal(7, hit.ChunkIndex);
        Assert.Equal(2, hit.TokenCount);
        Assert.Equal("doc.pdf", hit.Metadata.SourceFile);
        Assert.Equal(5, hit.Metadata.PageNumber);
        Assert.Equal(["A", "B"], hit.Metadata.HeadingPath);
    }

    [Fact]
    public async Task Upsert_replaces_an_existing_chunk_by_id()
    {
        await _store.UpsertAsync([Chunk("c1", "old text", [1, 0, 0, 0])]);
        await _store.UpsertAsync([Chunk("c1", "new text", [0, 1, 0, 0])]);

        var results = await _store.SearchAsync(new float[] { 0, 1, 0, 0 }, 5);

        var only = Assert.Single(results);
        Assert.Equal("new text", only.Chunk.Text);
        Assert.Empty(await _store.KeywordSearchAsync("old", 5));
    }

    [Fact]
    public async Task DeleteBySource_removes_only_that_sources_chunks_from_every_index()
    {
        await _store.UpsertAsync([
            Chunk("a1", "apples grow", [1, 0, 0, 0], "a.md"),
            Chunk("b1", "bananas grow", [0, 1, 0, 0], "b.md"),
        ]);

        await _store.DeleteBySourceAsync("a.md");

        Assert.Equal(["b1"], (await _store.SearchAsync(new float[] { 1, 0, 0, 0 }, 5)).Select(r => r.Chunk.Id));
        Assert.Equal(["b1"], (await _store.KeywordSearchAsync("grow", 5)).Select(r => r.Chunk.Id));
    }

    [Fact]
    public async Task Keyword_search_ranks_better_matches_first()
    {
        await _store.UpsertAsync([
            Chunk("one", "the invoice was paid", [1, 0, 0, 0]),
            Chunk("many", "invoice invoice invoice numbers and invoice totals", [0, 1, 0, 0]),
            Chunk("none", "unrelated gardening notes", [0, 0, 1, 0]),
        ]);

        var results = await _store.KeywordSearchAsync("invoice", 5);

        Assert.Equal(["many", "one"], results.Select(r => r.Chunk.Id));
        Assert.All(results, r => Assert.Equal(RetrievalSource.Keyword, r.Source));
    }

    [Fact]
    public async Task Keyword_search_tolerates_fts_syntax_in_user_input()
    {
        await _store.UpsertAsync([Chunk("c1", "refund policy details", [1, 0, 0, 0])]);

        var results = await _store.KeywordSearchAsync("refund\" OR (policy* NEAR -", 5);

        Assert.Single(results);
        Assert.Empty(await _store.KeywordSearchAsync("?!*", 5));
    }

    [Fact]
    public async Task Wrong_dimension_embeddings_are_rejected()
    {
        await Assert.ThrowsAsync<ArgumentException>(() => _store.UpsertAsync([Chunk("x", "x", [1, 0])]));
        await Assert.ThrowsAsync<ArgumentException>(() => _store.SearchAsync(new float[] { 1, 0 }, 1));
    }

    public void Dispose()
    {
        _store.Dispose();
        Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { /* best-effort cleanup */ }
    }
}
