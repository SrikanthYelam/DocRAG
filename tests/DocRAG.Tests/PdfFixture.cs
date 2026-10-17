using UglyToad.PdfPig.Core;
using UglyToad.PdfPig.Fonts.Standard14Fonts;
using UglyToad.PdfPig.Writer;

namespace DocRAG.Tests;

/// <summary>
/// Builds a PDF by wrapping text into lines and breaking pages by line count, so sentences run across pages like real documents.
/// Each line ends with a space glyph, as word processors emit; PdfPig fuses words across line ends when the PDF omits it.
/// </summary>
public static class PdfFixture
{
    public static byte[] Build(string text, int linesPerPage = 40, int charsPerLine = 90)
    {
        var builder = new PdfDocumentBuilder();
        var font = builder.AddStandard14Font(Standard14Font.Helvetica);
        var lines = Wrap(text, charsPerLine);

        for (var i = 0; i < lines.Count; i += linesPerPage)
        {
            var page = builder.AddPage(595, 842);
            var y = 800.0;
            foreach (var line in lines.Skip(i).Take(linesPerPage))
            {
                page.AddText(line + " ", 11, new PdfPoint(50, y), font);
                y -= 16;
            }
        }
        return builder.Build();
    }

    private static List<string> Wrap(string text, int width)
    {
        var lines = new List<string>();
        var current = "";
        foreach (var word in text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
        {
            if (current.Length > 0 && current.Length + 1 + word.Length > width)
            {
                lines.Add(current);
                current = word;
            }
            else
            {
                current = current.Length == 0 ? word : current + " " + word;
            }
        }
        if (current.Length > 0) lines.Add(current);
        return lines;
    }
}
