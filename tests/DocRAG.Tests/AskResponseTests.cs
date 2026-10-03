using System.Text.Json;
using System.Text.Json.Serialization;
using DocRAG.Api;
using DocRAG.Core;

namespace DocRAG.Tests;

public class AskResponseTests
{
    private static RetrievedChunk Hit(string id, string text, double score = 0.5) =>
        new(new DocumentChunk(id, text, 3, 0, new ChunkMetadata("refund-policy.pdf", 2, ["Policy", "Refunds"])),
            score, RetrievalSource.Vector, score);

    [Fact]
    public void Answered_response_echoes_the_question_and_inlines_cited_chunk_content()
    {
        var retrieved = new[] { Hit("c17", "Customers may request a refund within 30 days."), Hit("c18", "Unrelated chunk.") };
        var answer = new GeneratedAnswer(AnswerStatus.Answered, "The refund period is 30 days.",
            [new Citation("c17", "refund-policy.pdf", 2, "Policy > Refunds")], retrieved);

        var response = AskResponse.From("What is the refund period?", answer);

        Assert.Equal("What is the refund period?", response.Question);
        Assert.Equal("Answered", response.Status);
        Assert.Equal("The refund period is 30 days.", response.Answer);
        var source = Assert.Single(response.Sources);
        Assert.Equal(new SourceDto("refund-policy.pdf", "c17", 2, "Policy > Refunds", "Customers may request a refund within 30 days."), source);
        Assert.Equal(2, response.RetrievedChunks.Count); // everything retrieved, not just what was cited
    }

    [Fact]
    public void Insufficient_context_has_no_answer_and_no_sources_but_still_lists_what_was_retrieved()
    {
        var answer = new GeneratedAnswer(AnswerStatus.InsufficientContext, null, [], [Hit("c1", "text", 0.05)]);

        var response = AskResponse.From("Capital of Mongolia?", answer);

        Assert.Equal("InsufficientContext", response.Status);
        Assert.Null(response.Answer);
        Assert.Empty(response.Sources);
        Assert.Equal(0.05, Assert.Single(response.RetrievedChunks).VectorScore);
    }

    [Fact]
    public void Json_shape_uses_camel_case_names_the_clients_depend_on()
    {
        var answer = new GeneratedAnswer(AnswerStatus.Answered, "a",
            [new Citation("c17", "doc.pdf", null, "")], [Hit("c17", "body")]);
        var options = new JsonSerializerOptions(JsonSerializerDefaults.Web) { Converters = { new JsonStringEnumConverter() } };

        using var json = JsonDocument.Parse(JsonSerializer.Serialize(AskResponse.From("q", answer), options));
        var root = json.RootElement;

        Assert.Equal(["question", "status", "answer", "sources", "retrievedChunks"],
            root.EnumerateObject().Select(p => p.Name));
        Assert.Equal(["document", "chunkId", "page", "headingPath", "content"],
            root.GetProperty("sources")[0].EnumerateObject().Select(p => p.Name));
    }
}
