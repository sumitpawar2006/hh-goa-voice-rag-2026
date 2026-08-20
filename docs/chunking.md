# Chunking design and experiments

## Strategies

### Fixed

`FixedSizeChunker` creates non-overlapping word windows. It offers the lowest storage cost and predictable size, but can split a statement from its supporting sentence.

### Overlap

`OverlapChunker` advances by `CHUNK_SIZE - CHUNK_OVERLAP`. It preserves boundary context at the cost of more vectors and duplicated text. Invalid overlap greater than or equal to the chunk size is rejected.

### Semantic

`SemanticChunker` detects sentence boundaries across Latin and Indic punctuation (`. ! ? । ॥` plus CJK terminators), groups sentences toward the configured target, caps oversized sentences, and carries one short sentence across a boundary. This is the default because MSMARCO passages are natural-language evidence rather than arbitrary byte streams.

### Metadata-aware

`MetadataAwareChunker` never mixes document records, prefers paragraph boundaries, and makes language, query type, selected-passage status, and passage position available as Qdrant facets.

## Preserved contract

Every strategy emits the same `Chunk` schema:

```text
document_id, chunk_id, text, source, metadata,
strategy, position, token_count
```

This lets the ingestion and retrieval layers switch strategies without branching.

## Statistics

The router reports total chunks, source document count, average/min/max token count, configured overlap, and size buckets (`1–64`, `65–128`, `129–256`, `257+`). Storage impact is estimated as `chunk_count × embedding_dimension` vector values.

## Experiment protocol

`python -m scripts.chunk_experiment --query-limit 30`:

1. Loads the same normalized real MSMARCO-XI document sample.
2. Chunks it independently with each strategy.
3. Uses the same production multilingual embedding model.
4. Builds an isolated Qdrant collection for each strategy.
5. Runs the same held-out translated queries.
6. Scores against `is_selected` passage IDs.
7. Reports Recall@K, MRR, failure rate, retrieval latency, chunk count, and vector storage.

No strategy is declared best in advance. The measured report in `reports/chunking_experiment.json` is authoritative.

## Measured comparison

Run on 20 August 2026 using the same 500 real documents and 20 judged queries for every strategy:

| Strategy | Chunks | Recall@5 | MRR | Failure rate | Mean latency | Vector values |
|---|---:|---:|---:|---:|---:|---:|
| Fixed | 507 | 0.500 | 0.3083 | 0.500 | 1.301 ms | 194,688 |
| Overlap | 509 | 0.500 | 0.3083 | 0.500 | 1.304 ms | 195,456 |
| Semantic | 505 | 0.500 | 0.3083 | 0.500 | 1.277 ms | 193,920 |
| Metadata-aware | 507 | 0.500 | 0.3083 | 0.500 | 1.302 ms | 194,688 |

Quality tied on this bounded sample. Semantic chunking is selected by the declared tie-break order—MRR, Recall@K, storage, then retrieval latency—because it used the fewest vectors and also had the lowest measured latency. This is a measured resource-efficiency choice, not a claim of statistically superior relevance.
