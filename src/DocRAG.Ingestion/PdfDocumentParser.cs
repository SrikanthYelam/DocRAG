using DocRAG.Core;
using UglyToad.PdfPig;

namespace DocRAG.Ingestion;

/// <summary>One section per non-empty page; PDFs carry no reliable heading structure.</summary>
public sealed class PdfDocumentParser : IDocumentParser
{
    public bool CanParse(string fileName) =>
        string.Equals(Path.GetExtension(fileName), ".pdf", StringComparison.OrdinalIgnoreCase);

    public ParsedDocument Parse(Stream content, string sourceFile)
    {
        using var pdf = PdfDocument.Open(content);
        var sections = new List<DocumentSection>();
        foreach (var page in pdf.GetPages())
        {
            var text = page.Text.Trim();
            if (text.Length > 0)
                sections.Add(new DocumentSection(page.Number, [], text));
        }
        return new ParsedDocument(sourceFile, sections);
    }
}
