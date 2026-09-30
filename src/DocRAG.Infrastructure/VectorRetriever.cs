using DocRAG.Core;

namespace DocRAG.Infrastructure;

/// <summary>Dense retrieval: embed the question, then cosine top-K from the vector store.</summary>
public sealed class VectorRetriever(IEmbedder embedder, IVectorStore store) : IRetriever
{
    public async Task<IReadOnlyList<RetrievedChunk>> RetrieveAsync(
        string question, int topK, CancellationToken ct = default)
    {
        var vectors = await embedder.EmbedAsync([question], ct);
        return await store.SearchAsync(vectors[0], topK, ct);
    }
}
