namespace DocRAG.Core;

public interface ITokenCounter
{
    int Count(string text);
}

public interface IChunker
{
    IReadOnlyList<DocumentChunk> Chunk(ParsedDocument document);
}

public interface IEmbedder
{
    int Dimensions { get; }

    /// <summary>Embeds all texts, returning vectors in input order. Implementations batch internally.</summary>
    Task<IReadOnlyList<ReadOnlyMemory<float>>> EmbedAsync(IReadOnlyList<string> texts, CancellationToken ct = default);
}

public interface IVectorStore
{
    /// <summary>Inserts or replaces chunks by Id. Every chunk must carry an Embedding.</summary>
    Task UpsertAsync(IReadOnlyList<DocumentChunk> chunks, CancellationToken ct = default);

    /// <summary>Cosine-similarity top-K, highest score first.</summary>
    Task<IReadOnlyList<RetrievedChunk>> SearchAsync(ReadOnlyMemory<float> query, int topK, CancellationToken ct = default);

    /// <summary>Full-text top-K, best match first.</summary>
    Task<IReadOnlyList<RetrievedChunk>> KeywordSearchAsync(string query, int topK, CancellationToken ct = default);

    Task DeleteBySourceAsync(string sourceFile, CancellationToken ct = default);
}

public interface IRetriever
{
    /// <param name="mode">Vector, Keyword or Hybrid; null uses the configured default.</param>
    Task<IReadOnlyList<RetrievedChunk>> RetrieveAsync(
        string question, int topK, RetrievalSource? mode = null, CancellationToken ct = default);
}

public interface IAnswerGenerator
{
    Task<GeneratedAnswer> GenerateAsync(string question, IReadOnlyList<RetrievedChunk> context, CancellationToken ct = default);
}
