using DocRAG.Api;
using DocRAG.Core;
using DocRAG.Ingestion;
using Microsoft.AspNetCore.Http.HttpResults;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddDocRag(builder.Configuration);

var app = builder.Build();

app.MapPost("/documents", async Task<Results<Ok<IngestionResult>, BadRequest<string>>> (
    IFormFile file, IngestionService ingestion, CancellationToken ct) =>
{
    // Path.GetFileName strips any directory components a client may send.
    var name = Path.GetFileName(file.FileName);
    if (string.IsNullOrWhiteSpace(name)) return TypedResults.BadRequest("A file name is required.");

    try
    {
        await using var stream = file.OpenReadStream();
        return TypedResults.Ok(await ingestion.IngestAsync(stream, name, ct));
    }
    catch (NotSupportedException ex)
    {
        return TypedResults.BadRequest(ex.Message);
    }
}).DisableAntiforgery();

app.MapPost("/ask", async Task<Results<Ok<AskResponse>, BadRequest<string>>> (
    AskRequest request, IRetriever retriever, IAnswerGenerator generator, CancellationToken ct) =>
{
    if (string.IsNullOrWhiteSpace(request.Question)) return TypedResults.BadRequest("A question is required.");
    var topK = Math.Clamp(request.TopK ?? 5, 1, 20);

    var retrieved = await retriever.RetrieveAsync(request.Question, topK, ct);
    var answer = await generator.GenerateAsync(request.Question, retrieved, ct);
    return TypedResults.Ok(AskResponse.From(answer));
});

app.Run();

public sealed record AskRequest(string Question, int? TopK);

public sealed record AskResponse(
    string Status,
    string? Answer,
    IReadOnlyList<Citation> Citations,
    IReadOnlyList<RetrievedChunkDto> RetrievedChunks)
{
    public static AskResponse From(GeneratedAnswer a) => new(
        a.Status.ToString(),
        a.Text,
        a.Citations,
        a.RetrievedChunks.Select(r => new RetrievedChunkDto(
            r.Chunk.Id, r.Score, r.Source.ToString(), r.Chunk.Metadata.SourceFile,
            r.Chunk.Metadata.PageNumber, r.Chunk.Metadata.HeadingPath, r.Chunk.Text)).ToList());
}

public sealed record RetrievedChunkDto(
    string Id, double Score, string Source, string SourceFile, int? Page,
    IReadOnlyList<string> HeadingPath, string Text);

// Exposes the entry point to WebApplicationFactory-based tests.
public partial class Program;
