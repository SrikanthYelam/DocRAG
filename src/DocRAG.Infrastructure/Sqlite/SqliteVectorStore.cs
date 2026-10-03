using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using DocRAG.Core;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.Options;

namespace DocRAG.Infrastructure.Sqlite;

/// <summary>
/// SQLite store: a <c>chunks</c> table for content and metadata, a sqlite-vec <c>vec0</c> table for
/// embeddings (cosine distance), and an FTS5 table for keyword search. All three share the chunk's rowid.
/// </summary>
public sealed class SqliteVectorStore : IVectorStore, IDisposable
{
    private readonly SqliteVectorStoreOptions _options;
    private readonly string _connectionString;
    private readonly SemaphoreSlim _initLock = new(1, 1);
    private volatile bool _initialized;

    public SqliteVectorStore(IOptions<SqliteVectorStoreOptions> options)
    {
        _options = options.Value;
        var dir = Path.GetDirectoryName(Path.GetFullPath(_options.DatabasePath));
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        _connectionString = new SqliteConnectionStringBuilder
        {
            DataSource = _options.DatabasePath,
            Pooling = false,
        }.ToString();
    }

    public async Task UpsertAsync(IReadOnlyList<DocumentChunk> chunks, CancellationToken ct = default)
    {
        if (chunks.Count == 0) return;
        foreach (var c in chunks)
        {
            if (c.Embedding is not { } e) throw new ArgumentException($"Chunk {c.Id} has no embedding.");
            if (e.Length != _options.Dimensions)
                throw new ArgumentException($"Chunk {c.Id} embedding has {e.Length} dimensions, expected {_options.Dimensions}.");
        }

        await using var conn = await OpenAsync(ct);
        await using var tx = (SqliteTransaction)await conn.BeginTransactionAsync(ct);

        foreach (var chunk in chunks)
        {
            await DeleteByIdAsync(conn, tx, chunk.Id, ct);

            long pk;
            await using (var insert = conn.CreateCommand())
            {
                insert.Transaction = tx;
                insert.CommandText = """
                    INSERT INTO chunks (id, source_file, page_number, heading_path, chunk_index, token_count, text)
                    VALUES ($id, $source, $page, $heading, $idx, $tokens, $text)
                    RETURNING pk;
                    """;
                insert.Parameters.AddWithValue("$id", chunk.Id);
                insert.Parameters.AddWithValue("$source", chunk.Metadata.SourceFile);
                insert.Parameters.AddWithValue("$page", (object?)chunk.Metadata.PageNumber ?? DBNull.Value);
                insert.Parameters.AddWithValue("$heading", JsonSerializer.Serialize(chunk.Metadata.HeadingPath));
                insert.Parameters.AddWithValue("$idx", chunk.ChunkIndex);
                insert.Parameters.AddWithValue("$tokens", chunk.TokenCount);
                insert.Parameters.AddWithValue("$text", chunk.Text);
                pk = (long)(await insert.ExecuteScalarAsync(ct))!;
            }

            await using var vec = conn.CreateCommand();
            vec.Transaction = tx;
            vec.CommandText = "INSERT INTO vec_chunks (rowid, embedding) VALUES ($pk, $embedding);";
            vec.Parameters.AddWithValue("$pk", pk);
            vec.Parameters.AddWithValue("$embedding", ToBlob(chunk.Embedding!.Value));
            await vec.ExecuteNonQueryAsync(ct);
        }

        await tx.CommitAsync(ct);
    }

    public async Task<IReadOnlyList<RetrievedChunk>> SearchAsync(
        ReadOnlyMemory<float> query, int topK, CancellationToken ct = default)
    {
        if (query.Length != _options.Dimensions)
            throw new ArgumentException($"Query has {query.Length} dimensions, expected {_options.Dimensions}.");

        await using var conn = await OpenAsync(ct);
        await using var cmd = conn.CreateCommand();
        // vec0 cosine distance = 1 - cosine similarity.
        cmd.CommandText = $"""
            SELECT {ChunkColumns}, 1.0 - v.distance
            FROM (SELECT rowid, distance FROM vec_chunks WHERE embedding MATCH $query AND k = $k) v
            JOIN chunks c ON c.pk = v.rowid
            ORDER BY v.distance;
            """;
        cmd.Parameters.AddWithValue("$query", ToBlob(query));
        cmd.Parameters.AddWithValue("$k", topK);
        return await ReadAsync(cmd, RetrievalSource.Vector, ct);
    }

    public async Task<IReadOnlyList<RetrievedChunk>> KeywordSearchAsync(
        string query, int topK, CancellationToken ct = default)
    {
        var match = BuildFtsQuery(query);
        if (match is null) return [];

        await using var conn = await OpenAsync(ct);
        await using var cmd = conn.CreateCommand();
        // bm25() is lower-is-better and negative; flip the sign so higher score = better match.
        cmd.CommandText = $"""
            SELECT {ChunkColumns}, -bm25(chunks_fts)
            FROM chunks_fts
            JOIN chunks c ON c.pk = chunks_fts.rowid
            WHERE chunks_fts MATCH $match
            ORDER BY bm25(chunks_fts)
            LIMIT $k;
            """;
        cmd.Parameters.AddWithValue("$match", match);
        cmd.Parameters.AddWithValue("$k", topK);
        return await ReadAsync(cmd, RetrievalSource.Keyword, ct);
    }

    public async Task DeleteBySourceAsync(string sourceFile, CancellationToken ct = default)
    {
        await using var conn = await OpenAsync(ct);
        await using var tx = (SqliteTransaction)await conn.BeginTransactionAsync(ct);

        var ids = new List<string>();
        await using (var select = conn.CreateCommand())
        {
            select.Transaction = tx;
            select.CommandText = "SELECT id FROM chunks WHERE source_file = $source;";
            select.Parameters.AddWithValue("$source", sourceFile);
            await using var reader = await select.ExecuteReaderAsync(ct);
            while (await reader.ReadAsync(ct)) ids.Add(reader.GetString(0));
        }
        foreach (var id in ids) await DeleteByIdAsync(conn, tx, id, ct);

        await tx.CommitAsync(ct);
    }

    public void Dispose() => _initLock.Dispose();

    private const string ChunkColumns =
        "c.id, c.source_file, c.page_number, c.heading_path, c.chunk_index, c.token_count, c.text";

    private static async Task<IReadOnlyList<RetrievedChunk>> ReadAsync(
        SqliteCommand cmd, RetrievalSource source, CancellationToken ct)
    {
        var results = new List<RetrievedChunk>();
        await using var r = await cmd.ExecuteReaderAsync(ct);
        while (await r.ReadAsync(ct))
        {
            var headings = JsonSerializer.Deserialize<string[]>(r.GetString(3)) ?? [];
            var meta = new ChunkMetadata(r.GetString(1), r.IsDBNull(2) ? null : r.GetInt32(2), headings);
            var chunk = new DocumentChunk(r.GetString(0), r.GetString(6), r.GetInt32(5), r.GetInt32(4), meta);
            var score = r.GetDouble(7);
            results.Add(new RetrievedChunk(chunk, score, source,
                source == RetrievalSource.Vector ? score : null));
        }
        return results;
    }

    private static async Task DeleteByIdAsync(SqliteConnection conn, SqliteTransaction tx, string id, CancellationToken ct)
    {
        long? pk;
        await using (var find = conn.CreateCommand())
        {
            find.Transaction = tx;
            find.CommandText = "SELECT pk FROM chunks WHERE id = $id;";
            find.Parameters.AddWithValue("$id", id);
            pk = (long?)await find.ExecuteScalarAsync(ct);
        }
        if (pk is null) return;

        await using var del = conn.CreateCommand();
        del.Transaction = tx;
        del.CommandText = """
            DELETE FROM vec_chunks WHERE rowid = $pk;
            DELETE FROM chunks WHERE pk = $pk;
            """;
        del.Parameters.AddWithValue("$pk", pk.Value);
        await del.ExecuteNonQueryAsync(ct);
    }

    /// <summary>Turns free text into a safe FTS5 query: quoted terms OR-ed together (no operator injection).</summary>
    internal static string? BuildFtsQuery(string text)
    {
        var terms = new List<string>();
        var sb = new StringBuilder();
        foreach (var ch in text)
        {
            if (char.IsLetterOrDigit(ch)) sb.Append(ch);
            else if (sb.Length > 0) { terms.Add(sb.ToString()); sb.Clear(); }
        }
        if (sb.Length > 0) terms.Add(sb.ToString());
        return terms.Count == 0 ? null : string.Join(" OR ", terms.Select(t => $"\"{t}\""));
    }

    private static byte[] ToBlob(ReadOnlyMemory<float> v) => MemoryMarshal.AsBytes(v.Span).ToArray();

    private async Task<SqliteConnection> OpenAsync(CancellationToken ct)
    {
        var conn = new SqliteConnection(_connectionString);
        try
        {
            await conn.OpenAsync(ct);
            conn.EnableExtensions(true);
            conn.LoadExtension(SqliteVecLocator.Resolve(_options.ExtensionPath));
            conn.EnableExtensions(false);
            await EnsureSchemaAsync(conn, ct);
            return conn;
        }
        catch
        {
            await conn.DisposeAsync();
            throw;
        }
    }

    private async Task EnsureSchemaAsync(SqliteConnection conn, CancellationToken ct)
    {
        if (_initialized) return;
        await _initLock.WaitAsync(ct);
        try
        {
            if (_initialized) return;
            await using var cmd = conn.CreateCommand();
            cmd.CommandText = $"""
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS chunks (
                    pk INTEGER PRIMARY KEY,
                    id TEXT NOT NULL UNIQUE,
                    source_file TEXT NOT NULL,
                    page_number INTEGER,
                    heading_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    token_count INTEGER NOT NULL,
                    text TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_chunks_source ON chunks(source_file);
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                    embedding float[{_options.Dimensions}] distance_metric=cosine
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    text, heading_path, content='chunks', content_rowid='pk'
                );
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, text, heading_path) VALUES (new.pk, new.text, new.heading_path);
                END;
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, text, heading_path)
                    VALUES ('delete', old.pk, old.text, old.heading_path);
                END;
                """;
            await cmd.ExecuteNonQueryAsync(ct);
            _initialized = true;
        }
        finally
        {
            _initLock.Release();
        }
    }
}

public sealed class SqliteVectorStoreOptions
{
    public const string SectionName = "VectorStore";

    public string DatabasePath { get; set; } = "data/docrag.db";

    /// <summary>Embedding size; must match the embedder (text-embedding-3-small = 1536).</summary>
    public int Dimensions { get; set; } = 1536;

    /// <summary>Path to the sqlite-vec loadable extension. Defaults to native/&lt;rid&gt;/vec0.* next to the app.</summary>
    public string? ExtensionPath { get; set; }
}

internal static class SqliteVecLocator
{
    public static string Resolve(string? configured)
    {
        if (!string.IsNullOrWhiteSpace(configured)) return configured;

        var (rid, file) =
            RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? ("win-x64", "vec0.dll") :
            RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? (
                RuntimeInformation.ProcessArchitecture == Architecture.Arm64 ? "osx-arm64" : "osx-x64", "vec0.dylib") :
            (RuntimeInformation.ProcessArchitecture == Architecture.Arm64 ? "linux-arm64" : "linux-x64", "vec0.so");

        var path = Path.Combine(AppContext.BaseDirectory, "native", rid, file);
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"sqlite-vec extension not found at '{path}'. Run scripts/fetch-sqlite-vec.ps1 or set VectorStore:ExtensionPath.", path);
        return path;
    }
}
