using DocRAG.Core;
using DocRAG.Ingestion;
using DocRAG.Tests.Fakes;

namespace DocRAG.Tests;

public class ChunkerTests
{
    private static readonly ChunkingOptions Options = new() { TargetTokens = 100, MaxTokens = 120, OverlapFraction = 0.12 };

    private static string Words(int count, string prefix = "w") =>
        string.Join(' ', Enumerable.Range(0, count).Select(i => $"{prefix}{i}"));

    // Sentences of 10 words each so the chunker has natural split points.
    private static string Sentences(int count) =>
        string.Join(' ', Enumerable.Range(0, count).Select(i => $"{Words(9, $"s{i}x")}."));

    private static ParsedDocument Doc(params DocumentSection[] sections) => new("doc.md", sections);

    private static DocumentSection Section(string text, int? page = null, params string[] headings) =>
        new(page, headings, text);

    [Fact]
    public void Every_chunk_respects_the_token_budget()
    {
        var chunker = new Chunker(new WhitespaceTokenCounter(), Options);

        var chunks = chunker.Chunk(Doc(Section(Sentences(200))));

        Assert.True(chunks.Count > 5);
        Assert.All(chunks, c => Assert.InRange(c.TokenCount, 1, Options.MaxTokens));
        Assert.All(chunks, c => Assert.True(c.TokenCount <= Options.TargetTokens));
    }

    [Fact]
    public void Budget_holds_with_the_real_tokenizer()
    {
        var options = new ChunkingOptions { TargetTokens = 400, MaxTokens = 500, OverlapFraction = 0.12 };
        var counter = new TiktokenTokenCounter();
        var text = string.Join("\n\n", Enumerable.Range(0, 60).Select(i =>
            $"Paragraph {i} explains retrieval augmented generation, embeddings, chunking and cosine similarity in some detail. " +
            "Vector databases store embeddings; hybrid search combines keyword and semantic ranking."));

        var chunks = new Chunker(counter, options).Chunk(Doc(Section(text)));

        Assert.True(chunks.Count > 1);
        Assert.All(chunks, c => Assert.True(counter.Count(c.Text) <= options.MaxTokens));
        Assert.All(chunks, c => Assert.Equal(counter.Count(c.Text), c.TokenCount));
    }

    [Fact]
    public void Adjacent_chunks_overlap_by_roughly_the_configured_fraction()
    {
        var counter = new WhitespaceTokenCounter();
        var chunks = new Chunker(counter, Options).Chunk(Doc(Section(Sentences(200))));

        for (var i = 1; i < chunks.Count; i++)
        {
            var prev = chunks[i - 1].Text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            var next = chunks[i].Text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);

            // Longest k where the last k words of prev equal the first k words of next.
            var overlap = Enumerable.Range(1, Math.Min(prev.Length, next.Length))
                .Where(k => prev[^k..].SequenceEqual(next[..k]))
                .DefaultIfEmpty(0).Max();

            Assert.InRange(overlap, 1, (int)(Options.TargetTokens * 0.15));
        }
    }

    [Fact]
    public void Overlap_is_not_carried_across_sections()
    {
        var chunker = new Chunker(new WhitespaceTokenCounter(), Options);

        var chunks = chunker.Chunk(Doc(
            Section(Words(30, "alpha"), null, "A"),
            Section(Words(30, "beta"), null, "B")));

        Assert.Equal(2, chunks.Count);
        Assert.DoesNotContain("alpha", chunks[1].Text);
    }

    [Fact]
    public void Metadata_carries_source_page_and_heading_path()
    {
        var chunker = new Chunker(new WhitespaceTokenCounter(), Options);

        var chunks = chunker.Chunk(Doc(Section("Some text here.", 3, "Guide", "Install")));

        var meta = Assert.Single(chunks).Metadata;
        Assert.Equal("doc.md", meta.SourceFile);
        Assert.Equal(3, meta.PageNumber);
        Assert.Equal(["Guide", "Install"], meta.HeadingPath);
    }

    [Fact]
    public void Chunk_indexes_and_ids_are_sequential_and_deterministic()
    {
        var chunker = new Chunker(new WhitespaceTokenCounter(), Options);
        var doc = Doc(Section(Sentences(60)));

        var first = chunker.Chunk(doc);
        var second = chunker.Chunk(doc);

        Assert.Equal(Enumerable.Range(0, first.Count), first.Select(c => c.ChunkIndex));
        Assert.Equal(first.Select(c => c.Id), second.Select(c => c.Id));
        Assert.Equal(first.Count, first.Select(c => c.Id).Distinct().Count());
    }

    [Fact]
    public void A_single_oversized_word_is_still_split_within_budget()
    {
        var counter = new TiktokenTokenCounter();
        var options = new ChunkingOptions { TargetTokens = 50, MaxTokens = 60, OverlapFraction = 0.1 };
        var blob = new string('x', 2000) + Guid.NewGuid().ToString("N");

        var chunks = new Chunker(counter, options).Chunk(Doc(Section(blob)));

        Assert.True(chunks.Count > 1);
        Assert.All(chunks, c => Assert.True(counter.Count(c.Text) <= options.MaxTokens));
    }

    [Fact]
    public void Invalid_options_are_rejected()
    {
        Assert.Throws<ArgumentException>(() =>
            new Chunker(new WhitespaceTokenCounter(), new ChunkingOptions { TargetTokens = 600, MaxTokens = 500 }));
    }

    [Fact]
    public void Markdown_parser_builds_nested_heading_paths()
    {
        var md = "# Guide\nintro\n## Install\nrun it\n```\n# not a heading\n```\n## Use\nuse it\n# Other\nend";

        var doc = MarkdownParser.ParseText(md, "g.md");

        Assert.Equal(["Guide"], doc.Sections[0].HeadingPath);
        Assert.Equal(["Guide", "Install"], doc.Sections[1].HeadingPath);
        Assert.Contains("# not a heading", doc.Sections[1].Text);
        Assert.Equal(["Guide", "Use"], doc.Sections[2].HeadingPath);
        Assert.Equal(["Other"], doc.Sections[3].HeadingPath);
    }
}
