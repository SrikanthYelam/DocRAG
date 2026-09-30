using System.Text;
using DocRAG.Core;

namespace DocRAG.Ingestion;

/// <summary>Splits markdown (or plain text) into sections keyed by their heading path.</summary>
public sealed class MarkdownParser : IDocumentParser
{
    private static readonly string[] Extensions = [".md", ".markdown", ".txt"];

    public bool CanParse(string fileName) =>
        Extensions.Contains(Path.GetExtension(fileName), StringComparer.OrdinalIgnoreCase);

    public ParsedDocument Parse(Stream content, string sourceFile)
    {
        using var reader = new StreamReader(content, Encoding.UTF8);
        return ParseText(reader.ReadToEnd(), sourceFile);
    }

    public static ParsedDocument ParseText(string text, string sourceFile)
    {
        var sections = new List<DocumentSection>();
        var headings = new List<(int Level, string Title)>();
        var body = new StringBuilder();
        var inFence = false;

        void Flush()
        {
            var t = body.ToString().Trim();
            if (t.Length > 0)
                sections.Add(new DocumentSection(null, headings.Select(h => h.Title).ToArray(), t));
            body.Clear();
        }

        foreach (var line in text.ReplaceLineEndings("\n").Split('\n'))
        {
            if (line.TrimStart().StartsWith("```", StringComparison.Ordinal))
                inFence = !inFence;

            if (!inFence && TryParseHeading(line, out var level, out var title))
            {
                Flush();
                headings.RemoveAll(h => h.Level >= level);
                headings.Add((level, title));
            }
            else
            {
                body.Append(line).Append('\n');
            }
        }
        Flush();
        return new ParsedDocument(sourceFile, sections);
    }

    private static bool TryParseHeading(string line, out int level, out string title)
    {
        level = 0;
        title = "";
        var i = 0;
        while (i < line.Length && line[i] == '#') i++;
        if (i is 0 or > 6 || i >= line.Length || line[i] != ' ') return false;
        level = i;
        title = line[i..].Trim().TrimEnd('#').Trim();
        return title.Length > 0;
    }
}
