using DocRAG.Core;
using DocRAG.Infrastructure;
using DocRAG.Tests.Fakes;
using Microsoft.Extensions.Options;
using static DocRAG.Tests.Fakes.ScriptedVectorStore;

namespace DocRAG.Tests;

public class HybridRetrieverTests
{
    private static HybridRetriever Retriever(ScriptedVectorStore store, RetrievalOptions? options = null) =>
        new(new FakeEmbedder(), store, Options.Create(options ?? new RetrievalOptions()));

    [Fact]
    public async Task A_chunk_found_by_both_searches_outranks_chunks_found_by_one()
    {
        // "both" is only 2nd in each list, but appears in both, so its RRF contributions add up.
        var store = new ScriptedVectorStore(
            vector: [V("v-only", 0.9), V("both", 0.8)],
            keyword: [K("k-only"), K("both")]);

        var results = await Retriever(store).RetrieveAsync("q", 3, RetrievalSource.Hybrid);

        Assert.Equal("both", results[0].Chunk.Id);
        Assert.All(results, r => Assert.Equal(RetrievalSource.Hybrid, r.Source));
    }

    [Fact]
    public async Task An_exact_term_hit_missed_by_vector_search_still_surfaces()
    {
        var vector = Enumerable.Range(1, 5).Select(i => V($"semantic{i}", 0.6 - i * 0.05)).ToList();
        var store = new ScriptedVectorStore(vector, keyword: [K("error-code-page")]);

        var results = await Retriever(store).RetrieveAsync("ERR_4012", 5, RetrievalSource.Hybrid);

        var hit = Assert.Single(results, r => r.Chunk.Id == "error-code-page");
        Assert.Null(hit.VectorScore);
    }

    [Fact]
    public async Task A_vector_only_hit_surfaces_and_keeps_its_cosine_score()
    {
        var store = new ScriptedVectorStore(vector: [V("semantic", 0.72)], keyword: []);

        var results = await Retriever(store).RetrieveAsync("q", 5, RetrievalSource.Hybrid);

        Assert.Equal(0.72, Assert.Single(results).VectorScore);
    }

    [Fact]
    public async Task A_chunk_in_both_lists_keeps_the_vector_score()
    {
        var store = new ScriptedVectorStore(vector: [V("both", 0.55)], keyword: [K("both")]);

        var results = await Retriever(store).RetrieveAsync("q", 5, RetrievalSource.Hybrid);

        Assert.Equal(0.55, Assert.Single(results).VectorScore);
    }

    [Fact]
    public async Task Fused_scores_follow_the_rrf_formula()
    {
        var store = new ScriptedVectorStore(vector: [V("a", 0.9), V("b", 0.8)], keyword: [K("b")]);

        var results = await Retriever(store, new RetrievalOptions { RrfK = 60 }).RetrieveAsync("q", 5, RetrievalSource.Hybrid);

        Assert.Equal(1.0 / 62 + 1.0 / 61, results.Single(r => r.Chunk.Id == "b").Score, 10);
        Assert.Equal(1.0 / 61, results.Single(r => r.Chunk.Id == "a").Score, 10);
    }

    [Fact]
    public async Task Ties_are_broken_deterministically_in_favour_of_the_vector_ranking()
    {
        // Rank 1 in one list each => identical RRF scores.
        var store = new ScriptedVectorStore(vector: [V("from-vector", 0.5)], keyword: [K("from-keyword")]);

        var results = await Retriever(store).RetrieveAsync("q", 5, RetrievalSource.Hybrid);

        Assert.Equal(["from-vector", "from-keyword"], results.Select(r => r.Chunk.Id));
    }

    [Fact]
    public async Task Result_is_cut_to_topK_but_each_search_gets_the_wider_candidate_pool()
    {
        var vector = Enumerable.Range(1, 30).Select(i => V($"v{i}", 0.9 - i * 0.01)).ToList();
        var store = new ScriptedVectorStore(vector, keyword: []);

        var results = await Retriever(store, new RetrievalOptions { CandidatePoolSize = 20 })
            .RetrieveAsync("q", 3, RetrievalSource.Hybrid);

        Assert.Equal(3, results.Count);
        Assert.Equal(20, store.LastVectorK);
        Assert.Equal(20, store.LastKeywordK);
    }

    [Fact]
    public async Task Candidate_pool_is_never_smaller_than_topK()
    {
        var store = new ScriptedVectorStore();

        await Retriever(store, new RetrievalOptions { CandidatePoolSize = 5 }).RetrieveAsync("q", 12, RetrievalSource.Hybrid);

        Assert.Equal(12, store.LastVectorK);
    }

    [Fact]
    public async Task Both_searches_empty_gives_empty_result()
    {
        var results = await Retriever(new ScriptedVectorStore()).RetrieveAsync("q", 5, RetrievalSource.Hybrid);

        Assert.Empty(results);
    }

    [Fact]
    public async Task Vector_mode_does_not_run_keyword_search()
    {
        var store = new ScriptedVectorStore(vector: [V("a", 0.9)], keyword: [K("b")]);

        var results = await Retriever(store).RetrieveAsync("q", 5, RetrievalSource.Vector);

        Assert.Equal(["a"], results.Select(r => r.Chunk.Id));
        Assert.Equal(0, store.KeywordCalls);
    }

    [Fact]
    public async Task Keyword_mode_does_not_embed_or_run_vector_search()
    {
        var embedder = new FakeEmbedder();
        var store = new ScriptedVectorStore(vector: [V("a", 0.9)], keyword: [K("b")]);
        var retriever = new HybridRetriever(embedder, store, Options.Create(new RetrievalOptions()));

        var results = await retriever.RetrieveAsync("q", 5, RetrievalSource.Keyword);

        Assert.Equal(["b"], results.Select(r => r.Chunk.Id));
        Assert.Equal(0, store.VectorCalls);
        Assert.Empty(embedder.BatchSizes);
    }

    [Fact]
    public async Task Mode_defaults_to_the_configured_value()
    {
        var store = new ScriptedVectorStore(vector: [V("a", 0.9)], keyword: [K("b")]);

        var results = await Retriever(store, new RetrievalOptions { Mode = RetrievalSource.Keyword }).RetrieveAsync("q", 5);

        Assert.Equal(["b"], results.Select(r => r.Chunk.Id));
    }
}
