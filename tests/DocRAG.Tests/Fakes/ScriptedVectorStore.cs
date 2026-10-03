using DocRAG.Core;

namespace DocRAG.Tests.Fakes;

/// <summary>Returns pre-set ranked lists so tests control exactly what each search path "finds".</summary>
public sealed class ScriptedVectorStore(
    IReadOnlyList<RetrievedChunk>? vector = null,
    IReadOnlyList<RetrievedChunk>? keyword = null) : IVectorStore
{
    public int? LastVectorK { get; private set; }
    public int? LastKeywordK { get; private set; }
    public int VectorCalls { get; private set; }
    public int KeywordCalls { get; private set; }

    public Task<IReadOnlyList<RetrievedChunk>> SearchAsync(ReadOnlyMemory<float> query, int topK, CancellationToken ct = default)
    {
        VectorCalls++;
        LastVectorK = topK;
        return Task.FromResult<IReadOnlyList<RetrievedChunk>>((vector ?? []).Take(topK).ToList());
    }

    public Task<IReadOnlyList<RetrievedChunk>> KeywordSearchAsync(string query, int topK, CancellationToken ct = default)
    {
        KeywordCalls++;
        LastKeywordK = topK;
        return Task.FromResult<IReadOnlyList<RetrievedChunk>>((keyword ?? []).Take(topK).ToList());
    }

    public Task UpsertAsync(IReadOnlyList<DocumentChunk> chunks, CancellationToken ct = default) => throw new NotSupportedException();
    public Task DeleteBySourceAsync(string sourceFile, CancellationToken ct = default) => throw new NotSupportedException();

    public static DocumentChunk Chunk(string id) =>
        new(id, $"text of {id}", 3, 0, new ChunkMetadata("doc.md", null, []));

    /// <summary>A vector-search hit; cosine similarity is both Score and VectorScore.</summary>
    public static RetrievedChunk V(string id, double similarity) =>
        new(Chunk(id), similarity, RetrievalSource.Vector, similarity);

    /// <summary>A keyword-search hit; no vector score.</summary>
    public static RetrievedChunk K(string id, double bm25 = 5) =>
        new(Chunk(id), bm25, RetrievalSource.Keyword);
}
