# Evaluation and latency methodology

## Retrieval judgments

MSMARCO-XI exposes `passages.is_selected`. Ingestion converts selected passage positions into stable relevant document IDs. Evaluation queries are stored separately, avoiding manual or invented relevance labels.

Metrics:

- Recall@K: fraction of selected document IDs returned in top K.
- Mean Reciprocal Rank: inverse rank of the first selected document.
- Failure rate: fraction of judged queries with no selected hit.
- Mean measured retrieval latency.

`scripts/evaluate.py` runs vector-only and vector-plus-lexical-reranking on identical candidates. It recommends lexical reranking only when measured MRR is strictly higher.

### Measured retrieval result

Run on 20 August 2026 against 100 real Hindi validation queries with isolated cold query embeddings, `TOP_K=5`, and `CANDIDATE_K=12`:

| Mode | Recall@5 | MRR | Failure rate | Mean search latency |
|---|---:|---:|---:|---:|
| Vector only | 0.525 | 0.3012 | 0.46 | 13.438 ms |
| Vector + lexical reranking | 0.590 | 0.3492 | 0.40 | 16.705 ms |

Mean shared query-embedding time was 135.446 ms. Lexical reranking is the measured recommendation: it added 3.267 ms mean search latency while improving both quality metrics.

## Answer checks

The output gate verifies:

- valid structured output;
- at least one citation;
- citations limited to retrieved chunk IDs;
- sentence-level lexical/sequence support from cited text;
- no final answer when the generator declares insufficient context.

Automated tests include a directly copied supported answer and an unrelated Mars/restaurant claim that must be rejected.

## Latency

`scripts/benchmark.py` samples multiple real queries over multiple iterations. It records every successful request and calculates nearest-rank P50, P70, and P100 for embedding, vector search, reranking, generation, grounding, and total.

The benchmark report explicitly says `RAG text query; excludes external speech-to-text`. `/voice-query` separately returns STT time and full voice-to-answer wall time. This distinction prevents external provider time from being hidden inside a claimed RAG number.

## Interpreting the 200 ms target

- Compliance requires measured P100 under 200 ms for the stated scope.
- Warm-cache and cold-start results must not be mixed without labeling.
- A single fastest request is not evidence.
- Local LLM generation on CPU is expected to exceed the full-pipeline target; this must be reported rather than concealed.

## Measured latency result

The optimized benchmark used 30 distinct real queries, cold per-query embedding-cache entries, a prewarmed ONNX embedding model, lexical reranking, and the deterministic grounded extractive generator.

| Stage | P50 | P70 | P100 |
|---|---:|---:|---:|
| Embedding | 143.387 ms | 171.675 ms | 216.643 ms |
| Vector search | 15.909 ms | 18.647 ms | 23.558 ms |
| Reranking | 3.008 ms | 3.428 ms | 12.781 ms |
| Generation | 1.328 ms | 1.482 ms | 6.738 ms |
| Grounding | 1.042 ms | 1.130 ms | 9.589 ms |
| **Total** | **165.411 ms** | **191.633 ms** | **267.974 ms** |

P50 and P70 are under 200 ms; P100 is not. This report excludes external speech-to-text and therefore is not a voice end-to-end compliance claim.

A separate three-request local Qwen2.5-0.5B llama.cpp CPU run measured total P50 12,961.944 ms and P70/P100 24,170.064 ms. Two requests were grounded and one was refused. It proves the local LLM route works, but the sample is intentionally labeled small and is not used to claim competition latency.

Generated artifacts:

| Report | Command |
|---|---|
| `reports/dataset_inspection.json` | `python -m scripts.inspect_dataset` |
| `reports/ingestion.json` | `python -m scripts.ingest` |
| `reports/evaluation.json` | `python -m scripts.evaluate` |
| `reports/chunking_experiment.json` | `python -m scripts.chunk_experiment` |
| `reports/benchmark.json` | `python -m scripts.benchmark` |
| `reports/benchmark_llm.json` | `GENERATOR_PROVIDER=llama_server python -m scripts.benchmark --queries 3 --iterations 1` |
| `reports/final_readiness.json` | `python -m scripts.final_check` |
