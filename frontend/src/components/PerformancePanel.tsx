import { Clock, Gauge } from '@phosphor-icons/react'
import type { RAGResponse } from '../types'

interface Props { result: RAGResponse | null; sttLatency: number | null; totalLatency: number }

export function PerformancePanel({ result, sttLatency, totalLatency }: Props) {
  const retrieval = result ? result.latency.embedding_ms + result.latency.vector_search_ms + result.latency.reranking_ms : 0
  const metrics = [['STT', sttLatency], ['RETRIEVAL', result ? retrieval : null], ['GENERATION', result?.latency.generation_ms ?? null], ['GROUNDING', result?.latency.grounding_ms ?? null]] as const
  return <article className="performance-panel panel">
    <div className="panel-heading"><div><span className="section-index">03</span><h2>Performance</h2></div><Gauge size={21} aria-hidden="true" /></div>
    <div className="metric-list">{metrics.map(([label, value]) => <div key={label}><span>{label}</span><strong>{value == null ? '—' : `${value.toFixed(1)} ms`}</strong></div>)}</div>
    <div className="total-metric"><span><Clock size={18} aria-hidden="true" /> TOTAL</span><strong>{result ? `${totalLatency.toFixed(1)} ms` : '—'}</strong></div>
    <p>Measured wall-clock stages. STT is excluded from text-only totals.</p>
  </article>
}
