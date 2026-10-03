using DocRAG.Core;
using DocRAG.Infrastructure;
using DocRAG.Infrastructure.OpenAi;
using DocRAG.Infrastructure.Sqlite;
using DocRAG.Ingestion;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Options;
using OpenAI;

namespace DocRAG.Api;

public static class ServiceCollectionExtensions
{
    public static IServiceCollection AddDocRag(this IServiceCollection services, IConfiguration config)
    {
        services.Configure<OpenAiOptions>(config.GetSection(OpenAiOptions.SectionName));
        services.Configure<SqliteVectorStoreOptions>(config.GetSection(SqliteVectorStoreOptions.SectionName));
        services.Configure<ChunkingOptions>(config.GetSection(ChunkingOptions.SectionName));
        services.Configure<RetrievalOptions>(config.GetSection(RetrievalOptions.SectionName));
        services.Configure<AnswerOptions>(config.GetSection(AnswerOptions.SectionName));

        // The OpenAI client is created lazily so the app can start (and /documents parse errors surface)
        // without a key, but any call that needs OpenAI fails with an actionable message.
        services.AddSingleton(sp =>
        {
            // Precedence: OpenAI:ApiKey (user secrets / OpenAI__ApiKey), then the conventional OPENAI_API_KEY.
            var key = sp.GetRequiredService<IOptions<OpenAiOptions>>().Value.ApiKey;
            if (string.IsNullOrWhiteSpace(key))
                key = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
            if (string.IsNullOrWhiteSpace(key))
                throw new InvalidOperationException(
                    "No OpenAI API key found. Set the OPENAI_API_KEY environment variable, or use " +
                    "'dotnet user-secrets set \"OpenAI:ApiKey\" \"sk-...\"' / the OpenAI__ApiKey environment variable.");
            return new OpenAIClient(key);
        });
        services.AddSingleton<IEmbeddingGenerator<string, Embedding<float>>>(sp =>
            sp.GetRequiredService<OpenAIClient>()
                .GetEmbeddingClient(sp.GetRequiredService<IOptions<OpenAiOptions>>().Value.EmbeddingModel)
                .AsIEmbeddingGenerator());
        services.AddSingleton<IChatClient>(sp =>
            sp.GetRequiredService<OpenAIClient>()
                .GetChatClient(sp.GetRequiredService<IOptions<OpenAiOptions>>().Value.ChatModel)
                .AsIChatClient());

        services.AddSingleton<ITokenCounter, TiktokenTokenCounter>();
        services.AddSingleton<IChunker>(sp =>
            new Chunker(sp.GetRequiredService<ITokenCounter>(), sp.GetRequiredService<IOptions<ChunkingOptions>>().Value));
        services.AddSingleton<IDocumentParser, MarkdownParser>();
        services.AddSingleton<IDocumentParser, PdfDocumentParser>();
        services.AddSingleton<IEmbedder, OpenAiEmbedder>();
        services.AddSingleton<IVectorStore, SqliteVectorStore>();
        services.AddSingleton<IRetriever, HybridRetriever>();
        services.AddSingleton<IAnswerGenerator, OpenAiAnswerGenerator>();
        services.AddSingleton<IngestionService>();
        return services;
    }
}
