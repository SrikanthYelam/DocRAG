FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src
COPY DocRAG.sln ./
COPY src/ src/
COPY tests/ tests/
RUN dotnet publish src/DocRAG.Api -c Release -o /app

# Official sqlite-vec loadable extension for Linux, placed where SqliteVecLocator looks for it.
FROM build AS vec
ARG SQLITE_VEC_VERSION=0.1.9
ARG TARGETARCH
RUN case "$TARGETARCH" in arm64) A=aarch64; R=linux-arm64;; *) A=x86_64; R=linux-x64;; esac \
 && mkdir -p /vec/native/$R \
 && curl -fsSL "https://github.com/asg017/sqlite-vec/releases/download/v${SQLITE_VEC_VERSION}/sqlite-vec-${SQLITE_VEC_VERSION}-loadable-linux-${A}.tar.gz" \
    | tar -xz -C /vec/native/$R

FROM mcr.microsoft.com/dotnet/aspnet:9.0
WORKDIR /app
COPY --from=build /app .
COPY --from=vec /vec/native ./native
ENV ASPNETCORE_URLS=http://+:8080 \
    VectorStore__DatabasePath=/data/docrag.db
VOLUME /data
EXPOSE 8080
ENTRYPOINT ["dotnet", "DocRAG.Api.dll"]
