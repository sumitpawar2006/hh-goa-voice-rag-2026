import { Check, Circle, SpinnerGap, X } from '@phosphor-icons/react'
import type { StageTrace } from '../types'

interface Props { trace: StageTrace[]; state: string; hasTranscript: boolean }
const stages = [['voice', 'VOICE'], ['stt', 'STT'], ['embedding', 'EMBED'], ['vector_search', 'RETRIEVE'], ['reranking', 'RERANK'], ['generation', 'GENERATE'], ['grounding', 'VERIFY']] as const

export function PipelineTrace({ trace, state, hasTranscript }: Props) {
  const byName = new Map(trace.map((item) => [item.stage, item]))
  return <section className="pipeline-panel panel" aria-labelledby="pipeline-title">
    <div className="panel-heading"><div><span className="section-index">00</span><h2 id="pipeline-title">Live pipeline</h2></div><span className="technical-tag">MODEL HARNESS / TRACE</span></div>
    <ol className="pipeline-track">
      {stages.map(([key, label], index) => {
        const traceItem = byName.get(key)
        const passed = traceItem?.status === 'passed' || (key === 'stt' && hasTranscript)
        const failed = traceItem?.status === 'failed'
        const active = state === 'processing' && !passed && index === 2
        return <li key={key} className={passed ? 'passed' : failed ? 'failed' : active ? 'active' : ''}>
          <span className="stage-node" aria-hidden="true">{passed ? <Check size={15} weight="bold" /> : failed ? <X size={15} weight="bold" /> : active ? <SpinnerGap size={15} /> : <Circle size={8} weight="fill" />}</span>
          <span>{label}</span><small>{traceItem ? `${traceItem.latency_ms.toFixed(1)} ms` : '—'}</small>
        </li>
      })}
    </ol>
  </section>
}
