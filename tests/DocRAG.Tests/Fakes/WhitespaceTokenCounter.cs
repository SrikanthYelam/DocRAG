using DocRAG.Core;

namespace DocRAG.Tests.Fakes;

/// <summary>One token per whitespace-separated word, which makes budget/overlap assertions exact.</summary>
public sealed class WhitespaceTokenCounter : ITokenCounter
{
    public int Count(string text) => text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Length;
}
