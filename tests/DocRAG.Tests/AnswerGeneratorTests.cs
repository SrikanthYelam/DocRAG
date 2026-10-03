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
        new(new DocumentChunk(id, text, 1, 0, new ChunkMetadata("kb.md", 2, ["Refunds"])), score, RetrievalSource.Vector, score);

    // Mimics a fused result: Score is a tiny RRF value, VectorScore carries the real cosine similarity.
    private static RetrievedChunk Fused(string id, double rrfScore, double? vectorScore) =>
        new(new DocumentChunk(id, "text", 1, 0, new ChunkMetadata("kb.md", 2, ["Refunds"])),
            rrfScore, RetrievalSource.Hybrid, vectorScore);

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
    public async Task Gate_uses_vector_similarity_not_the_fused_rrf_score()
    {
        // RRF scores (~0.03) are far below 0.3, but the cosine similarity (0.8) is strong: the model must be called.
        var chat = new StubChatClient("""{"sufficient": true, "answer": "x", "citations": []}""");

        var answer = await Generator(chat).GenerateAsync("q?", [Fused("a", 0.033, 0.8)]);

        Assert.Equal(AnswerStatus.Answered, answer.Status);
        Assert.Equal(1, chat.Calls);
    }

    [Fact]
    public async Task Fused_results_with_weak_vector_similarity_are_gated_even_if_keywords_matched()
    {
        var chat = new StubChatClient("{}");

        var answer = await Generator(chat).GenerateAsync("q?", [Fused("kw", 0.033, null), Fused("weak", 0.016, 0.1)]);

        Assert.Equal(AnswerStatus.InsufficientContext, answer.Status);
        Assert.Equal(0, chat.Calls);
    }

    [Fact]
    public async Task Keyword_only_context_has_no_vector_scores_so_it_is_not_gated()
    {
        var chat = new StubChatClient("""{"sufficient": true, "answer": "x", "citations": []}""");
        var keywordHit = new RetrievedChunk(
            new DocumentChunk("k", "text", 1, 0, new ChunkMetadata("kb.md", null, [])), 4.2, RetrievalSource.Keyword);

        var answer = await Generator(chat).GenerateAsync("q?", [keywordHit]);

        Assert.Equal(AnswerStatus.Answered, answer.Status);
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

    [Theory]
    [InlineData("chunk:a")]
    [InlineData("[chunk:a]")]
    [InlineData(" a ")]
    public async Task Citation_ids_are_accepted_with_or_without_the_chunk_tag(string modelId)
    {
        var chat = new StubChatClient($$"""{"sufficient": true, "answer": "x", "citations": ["{{modelId}}"]}""");

        var answer = await Generator(chat).GenerateAsync("q?", [Hit("a", 0.8)]);

        Assert.Equal("a", Assert.Single(answer.Citations).ChunkId);
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
