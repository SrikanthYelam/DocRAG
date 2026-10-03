using DocRAG.Core;

namespace DocRAG.Api;

public sealed record AskRequest(string Question, int? TopK, RetrievalSource? Mode);

public sealed record AskResponse(
    string Question,
    string Status,
    string? Answer,
    IReadOnlyList<SourceDto> Sources,
    IReadOnlyList<RetrievedChunkDto> RetrievedChunks)
{
    public static AskResponse From(string question, GeneratedAnswer answer)
    {
        var chunksById = answer.RetrievedChunks.Select(r => r.Chunk).DistinctBy(c => c.Id).ToDictionary(c => c.Id);

        // Citations are validated against the retrieved set by the generator, so the lookup always succeeds.
        var sources = answer.Citations
            .Where(c => chunksById.ContainsKey(c.ChunkId))
            .Select(c => new SourceDto(c.SourceFile, c.ChunkId, c.PageNumber, c.HeadingPath, chunksById[c.ChunkId].Text))
            .ToList();

        var retrieved = answer.RetrievedChunks.Select(r => new RetrievedChunkDto(
            r.Chunk.Id, r.Score, r.VectorScore, r.Source.ToString(), r.Chunk.Metadata.SourceFile,
            r.Chunk.Metadata.PageNumber, r.Chunk.Metadata.HeadingPath, r.Chunk.Text)).ToList();

        return new AskResponse(question, answer.Status.ToString(), answer.Text, sources, retrieved);
    }
}

/// <summary>A chunk the answer cites.</summary>
public sealed record SourceDto(string Document, string ChunkId, int? Page, string HeadingPath, string Content);

/// <summary>Any chunk that was retrieved, cited or not, with its scores.</summary>
public sealed record RetrievedChunkDto(
    string Id, double Score, double? VectorScore, string Source, string SourceFile, int? Page,
    IReadOnlyList<string> HeadingPath, string Text);
