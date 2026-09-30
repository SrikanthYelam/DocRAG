using DocRAG.Core;

namespace DocRAG.Tests.Fakes;

/// <summary>
/// Deterministic bag-of-words embedder: each word hashes into one of <see cref="Dimensions"/> buckets and the
/// vector is L2-normalised. Texts sharing words get high cosine similarity; no network involved.
/// </summary>
public sealed class FakeEmbedder(int dimensions = 64) : IEmbedder
{
    public int Dimensions { get; } = dimensions;
    public List<int> BatchSizes { get; } = [];

    public Task<IReadOnlyList<ReadOnlyMemory<float>>> EmbedAsync(IReadOnlyList<string> texts, CancellationToken ct = default)
    {
        BatchSizes.Add(texts.Count);
        return Task.FromResult<IReadOnlyList<ReadOnlyMemory<float>>>(texts.Select(Embed).ToList());
    }

    public ReadOnlyMemory<float> Embed(string text)
    {
        var v = new float[Dimensions];
        foreach (var word in text.ToLowerInvariant().Split([' ', '\n', '.', ',', '?', '!', ':', '"'], StringSplitOptions.RemoveEmptyEntries))
            v[StableHash(word) % Dimensions] += 1f;

        var norm = MathF.Sqrt(v.Sum(x => x * x));
        if (norm > 0) for (var i = 0; i < v.Length; i++) v[i] /= norm;
        return v;
    }

    // string.GetHashCode is randomised per process; use a stable one so tests are deterministic.
    private static int StableHash(string s)
    {
        var h = 17;
        foreach (var c in s) h = unchecked(h * 31 + c);
        return h & int.MaxValue;
    }
}
