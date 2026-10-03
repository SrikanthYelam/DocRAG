namespace DocRAG.Core;

/// <summary>Where a chunk came from. PageNumber is null for markdown/text.</summary>
public sealed record ChunkMetadata(
    string SourceFile,
    int? PageNumber,
    IReadOnlyList<string> HeadingPath)
{
    public string HeadingPathText => string.Join(" > ", HeadingPath);
}

public sealed record DocumentChunk(
    string Id,
    string Text,
    int TokenCount,
    int ChunkIndex,
    ChunkMetadata Metadata,
    ReadOnlyMemory<float>? Embedding = null);

public enum RetrievalSource { Vector, Keyword, Hybrid }

/// <summary>
/// Score meaning depends on Source: cosine similarity (Vector), BM25-derived score (Keyword), or a
/// Reciprocal Rank Fusion score (Hybrid). VectorScore is always the cosine similarity when the chunk was found by
/// vector search (null otherwise), so confidence checks keep working after fusion replaces Score.
/// </summary>
public sealed record RetrievedChunk(
    DocumentChunk Chunk,
    double Score,
    RetrievalSource Source,
    double? VectorScore = null);

public sealed record Citation(string ChunkId, string SourceFile, int? PageNumber, string HeadingPath);

public enum AnswerStatus { Answered, InsufficientContext }

public sealed record GeneratedAnswer(
    AnswerStatus Status,
    string? Text,
    IReadOnlyList<Citation> Citations,
    IReadOnlyList<RetrievedChunk> RetrievedChunks);

/// <summary>A run of text from a parsed file, with its position in the source.</summary>
public sealed record DocumentSection(int? PageNumber, IReadOnlyList<string> HeadingPath, string Text);

public sealed record ParsedDocument(string SourceFile, IReadOnlyList<DocumentSection> Sections);
