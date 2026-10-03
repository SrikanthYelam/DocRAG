using DocRAG.Core;
using Microsoft.Extensions.Options;

namespace DocRAG.Infrastructure;

/// <summary>
/// Retrieval in three modes. Vector: cosine top-K. Keyword: BM25 top-K. Hybrid: both searches over a wider candidate
/// pool, merged with Reciprocal Rank Fusion, which uses only ranks because cosine and BM25 scores aren't comparable.
/// </summary>
public sealed class HybridRetriever(IEmbedder embedder, IVectorStore store, IOptions<RetrievalOptions> options) : IRetriever
{
    private readonly RetrievalOptions _options = options.Value;

    public async Task<IReadOnlyList<RetrievedChunk>> RetrieveAsync(
        string question, int topK, RetrievalSource? mode = null, CancellationToken ct = default)
    {
        switch (mode ?? _options.Mode)
        {
            case RetrievalSource.Vector:
                return await VectorAsync(question, topK, ct);
            case RetrievalSource.Keyword:
                return await store.KeywordSearchAsync(question, topK, ct);
            default:
                var pool = Math.Max(_options.CandidatePoolSize, topK);
                var vectorTask = VectorAsync(question, pool, ct);
                var keywordTask = store.KeywordSearchAsync(question, pool, ct);
                await Task.WhenAll(vectorTask, keywordTask);
                return Fuse(vectorTask.Result, keywordTask.Result, topK, _options.RrfK);
        }
    }

    private async Task<IReadOnlyList<RetrievedChunk>> VectorAsync(string question, int k, CancellationToken ct)
    {
        var vectors = await embedder.EmbedAsync([question], ct);
        return await store.SearchAsync(vectors[0], k, ct);
    }

    /// <summary>Each list contributes 1 / (k + rank) per chunk; chunks found by both lists add up and rise to the top.</summary>
    internal static IReadOnlyList<RetrievedChunk> Fuse(
        IReadOnlyList<RetrievedChunk> vector, IReadOnlyList<RetrievedChunk> keyword, int topK, int rrfK)
    {
        var fused = new Dictionary<string, (DocumentChunk Chunk, double Score, double? VectorScore, int FirstSeen)>();

        void Add(IReadOnlyList<RetrievedChunk> ranked, bool isVector)
        {
            for (var i = 0; i < ranked.Count; i++)
            {
                var hit = ranked[i];
                var contribution = 1.0 / (rrfK + i + 1);
                fused[hit.Chunk.Id] = fused.TryGetValue(hit.Chunk.Id, out var existing)
                    ? (existing.Chunk, existing.Score + contribution, existing.VectorScore ?? hit.VectorScore, existing.FirstSeen)
                    : (hit.Chunk, contribution, isVector ? hit.VectorScore : null, fused.Count);
            }
        }

        Add(vector, isVector: true);
        Add(keyword, isVector: false);

        return fused.Values
            .OrderByDescending(f => f.Score)
            .ThenBy(f => f.FirstSeen) // deterministic: on ties, the vector-ranked / earlier hit wins
            .Take(topK)
            .Select(f => new RetrievedChunk(f.Chunk, f.Score, RetrievalSource.Hybrid, f.VectorScore))
            .ToList();
    }
}
