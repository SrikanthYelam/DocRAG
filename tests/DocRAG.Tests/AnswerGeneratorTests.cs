using DocRAG.Core;
using DocRAG.Infrastructure.OpenAi;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Options;

namespace DocRAG.Tests;

public class AnswerGeneratorTests
{
    private sealed class StubChatClient(string reply) : IChatClient
    {
        public int Calls { get; private set; }
        public IList<ChatMessage>? LastMessages { get; private set; }

        public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken ct = default)
        {
            Calls++;
            LastMessages = messages.ToList();
            return Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, reply)));
        }

        public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken ct = default) =>
            throw new NotSupportedException();

        public object? GetService(Type serviceType, object? serviceKey = null) => null;
        public void Dispose() { }
    }

    private static RetrievedChunk Hit(string id, double score, string text = "text") =>
        new(new DocumentChunk(id, text, 1, 0, new ChunkMetadata("kb.md", 2, ["Refunds"])), score, RetrievalSource.Vector);

    private static OpenAiAnswerGenerator Generator(StubChatClient chat, double min = 0.3) =>
        new(chat, Options.Create(new AnswerOptions { MinSimilarity = min }));

    [Fact]
    public async Task Weak_retrieval_returns_insufficient_without_calling_the_model()
    {
        var chat = new StubChatClient("{}");

        var answer = await Generator(chat).GenerateAsync("q?", [Hit("a", 0.1)]);

        Assert.Equal(AnswerStatus.InsufficientContext, answer.Status);
        Assert.Null(answer.Text);
        Assert.Equal(0, chat.Calls);
        Assert.Single(answer.RetrievedChunks);
    }

    [Fact]
    public async Task Empty_context_is_insufficient()
    {
        var answer = await Generator(new StubChatClient("{}")).GenerateAsync("q?", []);

        Assert.Equal(AnswerStatus.InsufficientContext, answer.Status);
    }

    [Fact]
    public async Task Model_can_declare_context_insufficient()
    {
        var chat = new StubChatClient("""{"sufficient": false, "answer": "", "citations": []}""");

        var answer = await Generator(chat).GenerateAsync("q?", [Hit("a", 0.8)]);

        Assert.Equal(AnswerStatus.InsufficientContext, answer.Status);
        Assert.Null(answer.Text);
        Assert.Equal(1, chat.Calls);
    }

    [Fact]
    public async Task Answered_result_carries_text_and_only_real_citations()
    {
        var chat = new StubChatClient("""{"sufficient": true, "answer": "30 days [chunk:a]", "citations": ["a", "invented"]}""");

        var answer = await Generator(chat).GenerateAsync("refund window?", [Hit("a", 0.8), Hit("b", 0.7)]);

        Assert.Equal(AnswerStatus.Answered, answer.Status);
        Assert.Equal("30 days [chunk:a]", answer.Text);
        var citation = Assert.Single(answer.Citations);
        Assert.Equal(new Citation("a", "kb.md", 2, "Refunds"), citation);
    }

    [Fact]
    public async Task Prompt_tags_each_chunk_with_its_id_and_source()
    {
        var chat = new StubChatClient("""{"sufficient": true, "answer": "x", "citations": []}""");

        await Generator(chat).GenerateAsync("the question", [Hit("abc123", 0.9, "chunk body")]);

        var user = chat.LastMessages!.Single(m => m.Role == ChatRole.User).Text;
        Assert.Contains("[chunk:abc123] source=kb.md page=2 section=\"Refunds\"", user);
        Assert.Contains("chunk body", user);
        Assert.Contains("the question", user);
        Assert.Contains("do not guess", chat.LastMessages!.Single(m => m.Role == ChatRole.System).Text);
    }

    [Fact]
    public async Task Non_json_model_output_is_an_error_not_a_fake_answer()
    {
        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            Generator(new StubChatClient("Sure! The answer is 30 days.")).GenerateAsync("q?", [Hit("a", 0.9)]));
    }
}
