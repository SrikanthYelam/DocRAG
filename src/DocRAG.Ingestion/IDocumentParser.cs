using DocRAG.Core;

namespace DocRAG.Ingestion;

public interface IDocumentParser
{
    bool CanParse(string fileName);
    ParsedDocument Parse(Stream content, string sourceFile);
}
