using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using DocRAG.Core;

namespace DocRAG.Ingestion;

/// <summary>
/// Splits each section into paragraphs, then sentences, then words as needed so every unit fits the
/// budget; packs units up to TargetTokens; and starts each following chunk in the same section with the
/// tail of the previous one. Overlap counts toward the budget, so no chunk exceeds TargetTokens
/// (which is itself capped by MaxTokens).
/// </summary>
public sealed partial class Chunker : IChunker
{
    private readonly ITokenCounter _tokens;
    private readonly int _overlapTokens;
    private readonly int _budget;

    public Chunker(ITokenCounter tokens, ChunkingOptions options)
    {
        if (options.TargetTokens > options.MaxTokens)
            throw new ArgumentException("TargetTokens must not exceed MaxTokens.", nameof(options));
        if (options.OverlapFraction is < 0 or >= 0.5)
            throw new ArgumentException("OverlapFraction must be in [0, 0.5).", nameof(options));

        _tokens = tokens;
        _budget = options.TargetTokens;
        _overlapTokens = (int)Math.Round(options.TargetTokens * options.OverlapFraction);
    }

    public IReadOnlyList<DocumentChunk> Chunk(ParsedDocument document)
    {
        var result = new List<DocumentChunk>();
        foreach (var section in document.Sections)
        {
            var meta = new ChunkMetadata(document.SourceFile, section.PageNumber, section.HeadingPath);
            foreach (var text in ChunkSection(section.Text))
            {
                var index = result.Count;
                result.Add(new DocumentChunk(MakeId(document.SourceFile, index, text), text,
                    _tokens.Count(text), index, meta));
            }
        }
        return result;
    }

    private IEnumerable<string> ChunkSection(string text)
    {
        // Units must leave room for the overlap prefix and the "\n\n" joiner.
        var unitLimit = Math.Max(1, _budget - _overlapTokens - 2);
        var units = SplitParagraphs(text).SelectMany(p => FitUnit(p, unitLimit)).ToList();

        var current = "";
        var hasNewContent = false;

        foreach (var unit in units)
        {
            var candidate = current.Length == 0 ? unit : current + "\n\n" + unit;
            if (current.Length == 0 || _tokens.Count(candidate) <= _budget)
            {
                current = candidate;
                hasNewContent = true;
                continue;
            }

            yield return current;

            var overlap = Tail(current, _overlapTokens);
            var withOverlap = overlap.Length == 0 ? unit : overlap + "\n\n" + unit;
            // Tokenization at the join can add a token or two; fall back to no overlap rather than exceed budget.
            current = _tokens.Count(withOverlap) <= _budget ? withOverlap : unit;
            hasNewContent = true;
        }

        if (current.Length > 0 && hasNewContent)
            yield return current;
    }

    private static IEnumerable<string> SplitParagraphs(string text) =>
        ParagraphBreak().Split(text).Select(p => p.Trim()).Where(p => p.Length > 0);

    /// <summary>Returns the paragraph whole if it fits, else splits by sentence, then word, then character.</summary>
    private IEnumerable<string> FitUnit(string paragraph, int limit)
    {
        if (_tokens.Count(paragraph) <= limit)
        {
            yield return paragraph;
            yield break;
        }

        var pieces = SentenceBreak().Split(paragraph).Where(s => s.Length > 0)
            .SelectMany(s => _tokens.Count(s) <= limit ? [s] : SplitWords(s, limit));
        foreach (var packed in Pack(pieces, " ", limit))
            yield return packed;
    }

    private IEnumerable<string> SplitWords(string text, int limit)
    {
        var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)
            .SelectMany(w => _tokens.Count(w) <= limit ? [w] : SplitChars(w, limit));
        return Pack(words, " ", limit);
    }

    /// <summary>Greedily joins pieces with a separator without exceeding the token limit.</summary>
    private IEnumerable<string> Pack(IEnumerable<string> pieces, string separator, int limit)
    {
        var buffer = "";
        foreach (var piece in pieces)
        {
            var candidate = buffer.Length == 0 ? piece : buffer + separator + piece;
            if (buffer.Length > 0 && _tokens.Count(candidate) > limit)
            {
                yield return buffer;
                buffer = piece;
            }
            else
            {
                buffer = candidate;
            }
        }
        if (buffer.Length > 0) yield return buffer;
    }

    private IEnumerable<string> SplitChars(string word, int limit)
    {
        var start = 0;
        while (start < word.Length)
        {
            var len = Math.Min(word.Length - start, limit * 2);
            while (len > 1 && _tokens.Count(word.Substring(start, len)) > limit) len = len * 3 / 4;
            yield return word.Substring(start, len);
            start += len;
        }
    }

    /// <summary>Longest whitespace-aligned suffix of text within maxTokens.</summary>
    private string Tail(string text, int maxTokens)
    {
        if (maxTokens <= 0) return "";
        var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var best = "";
        for (var n = 1; n <= words.Length; n++)
        {
            var candidate = string.Join(' ', words[^n..]);
            if (_tokens.Count(candidate) > maxTokens) break;
            best = candidate;
        }
        return best;
    }

    private static string MakeId(string source, int index, string text)
    {
        var hash = SHA256.HashData(Encoding.UTF8.GetBytes($"{source}|{index}|{text}"));
        return Convert.ToHexString(hash)[..12].ToLowerInvariant();
    }

    [GeneratedRegex(@"\n\s*\n")]
    private static partial Regex ParagraphBreak();

    [GeneratedRegex(@"(?<=[.!?])\s+")]
    private static partial Regex SentenceBreak();
}
