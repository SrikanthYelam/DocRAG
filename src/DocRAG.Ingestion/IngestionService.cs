using DocRAG.Core;

namespace DocRAG.Ingestion;

public sealed record IngestionResult(string SourceFile, int ChunkCount, int TotalTokens);

/// <summary>Parse -> chunk -> embed -> store. Re-ingesting a file replaces its previous chunks.</summary>
public sealed class IngestionService(
    IEnumerable<IDocumentParser> parsers,
    IChunker chunker,
    IEmbedder embedder,
    IVectorStore store)
{
    public async Task<IngestionResult> IngestAsync(Stream content, string fileName, CancellationToken ct = default)
    {
        var parser = parsers.FirstOrDefault(p => p.CanParse(fileName))
            ?? throw new NotSupportedException($"No parser for '{Path.GetExtension(fileName)}' files.");

        var chunks = chunker.Chunk(parser.Parse(content, fileName));
        if (chunks.Count == 0)
            return new IngestionResult(fileName, 0, 0);

        // Prefixing the heading path gives the embedding the section context the chunk text omits.
        var inputs = chunks.Select(c => c.Metadata.HeadingPath.Count == 0
            ? c.Text
            : $"{c.Metadata.HeadingPathText}\n\n{c.Text}").ToList();
        var vectors = await embedder.EmbedAsync(inputs, ct);

        var embedded = chunks.Select((c, i) => c with { Embedding = vectors[i] }).ToList();
        await store.DeleteBySourceAsync(fileName, ct);
        await store.UpsertAsync(embedded, ct);
        return new IngestionResult(fileName, chunks.Count, chunks.Sum(c => c.TokenCount));
    }
}
