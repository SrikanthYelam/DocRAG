namespace DocRAG.Core;

public sealed class ChunkingOptions
{
    public const string SectionName = "Chunking";

    /// <summary>Chunks are packed up to this many tokens.</summary>
    public int TargetTokens { get; set; } = 400;

    /// <summary>Hard cap; no chunk (including overlap) exceeds this.</summary>
    public int MaxTokens { get; set; } = 500;

    /// <summary>Fraction of the previous chunk repeated at the start of the next.</summary>
    public double OverlapFraction { get; set; } = 0.12;
}

public sealed class RetrievalOptions
{
    public const string SectionName = "Retrieval";

    /// <summary>Default mode when a request doesn't specify one.</summary>
    public RetrievalSource Mode { get; set; } = RetrievalSource.Hybrid;

    /// <summary>Candidates fetched from each search before fusing; at least topK is always fetched.</summary>
    public int CandidatePoolSize { get; set; } = 20;

    /// <summary>Reciprocal Rank Fusion constant: score = sum of 1 / (RrfK + rank). 60 is the standard value.</summary>
    public int RrfK { get; set; } = 60;
}

public sealed class AnswerOptions
{
    public const string SectionName = "Answer";

    /// <summary>If no retrieved chunk reaches this cosine similarity, skip the LLM and return InsufficientContext.</summary>
    public double MinSimilarity { get; set; } = 0.30;
}
