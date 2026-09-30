using DocRAG.Core;
using Microsoft.ML.Tokenizers;

namespace DocRAG.Ingestion;

/// <summary>Counts tokens with the o200k_base encoding used by the gpt-4o family.</summary>
public sealed class TiktokenTokenCounter : ITokenCounter
{
    private readonly Tokenizer _tokenizer = TiktokenTokenizer.CreateForModel("gpt-4o");

    public int Count(string text) => string.IsNullOrEmpty(text) ? 0 : _tokenizer.CountTokens(text);
}
