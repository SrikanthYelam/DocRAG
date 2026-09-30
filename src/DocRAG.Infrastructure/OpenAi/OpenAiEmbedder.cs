using DocRAG.Core;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Options;

namespace DocRAG.Infrastructure.OpenAi;

public sealed class OpenAiOptions
{
    public const string SectionName = "OpenAI";

    /// <summary>Supply via user secrets or the OpenAI__ApiKey environment variable; never commit it.</summary>
    public string? ApiKey { get; set; }
    public string EmbeddingModel { get; set; } = "text-embedding-3-small";
    public string ChatModel { get; set; } = "gpt-4o-mini";
    public int EmbeddingDimensions { get; set; } = 1536;

    /// <summary>Texts per embeddings request (the API allows up to 2048).</summary>
    public int EmbeddingBatchSize { get; set; } = 96;
}

/// <summary>IEmbedder over any Microsoft.Extensions.AI embedding generator, sending texts in batches.</summary>
public sealed class OpenAiEmbedder(
    IEmbeddingGenerator<string, Embedding<float>> generator,
    IOptions<OpenAiOptions> options) : IEmbedder
{
    private readonly OpenAiOptions _options = options.Value;

    public int Dimensions => _options.EmbeddingDimensions;

    public async Task<IReadOnlyList<ReadOnlyMemory<float>>> EmbedAsync(
        IReadOnlyList<string> texts, CancellationToken ct = default)
    {
        var results = new List<ReadOnlyMemory<float>>(texts.Count);
        var batchSize = Math.Max(1, _options.EmbeddingBatchSize);

        for (var i = 0; i < texts.Count; i += batchSize)
        {
            var batch = texts.Skip(i).Take(batchSize).ToList();
            var embeddings = await generator.GenerateAsync(batch, cancellationToken: ct);
            if (embeddings.Count != batch.Count)
                throw new InvalidOperationException(
                    $"Embedding service returned {embeddings.Count} vectors for {batch.Count} inputs.");
            results.AddRange(embeddings.Select(e => e.Vector));
        }
        return results;
    }
}
