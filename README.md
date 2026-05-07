# floating-finch-zig

Versão **Zig** do detector de fraude `floating-finch` para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026).

Repositório companion: [lemesdaniel/floating-finch](https://github.com/lemesdaniel/floating-finch) (versão Nim, mais performática).

## Stack

- **Zig 0.14.1** (single-thread, Connection: close por request)
- **Python** (preprocessamento k-means do dataset → `index.bin`)
- **nginx** (round-robin upstream)
- **Docker linux/amd64** (multi-stage)

## Algoritmo

IVF (Inverted File) com 2048 clusters, vetores quantizados em int16 (escala 10000),
busca KNN top-5 com bbox repair (re-checa clusters vizinhos quando a contagem de
fraudes nos top-5 está em zona ambígua [1, 4]).

Layout do `index.bin` versão 2: **SoA per-cluster** (cada cluster armazena dim-major
para SIMD friendly). Ver `nim/docs/INDEX_BIN_FORMAT.md` no repo Nim para detalhes.

## Build

```bash
# Local (Mac ARM ou Linux x86_64)
cd zig && zig build -Doptimize=ReleaseFast

# Docker (linux/amd64)
docker buildx build --platform linux/amd64 -f docker/Dockerfile.zig -t floating-finch-zig:test --load .
```

## Imagem pública

[`ghcr.io/lemesdaniel/floating-finch-zig:v1`](https://ghcr.io/lemesdaniel/floating-finch-zig)

## Branch `submission`

Contém apenas `docker-compose.yml` + `nginx.conf` + `info.json` (estrutura mínima
exigida pela Rinha).

## Licença

MIT.
