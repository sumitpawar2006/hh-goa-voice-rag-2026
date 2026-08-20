import { CheckCircle, Info, ThumbsDown, ThumbsUp, WarningCircle } from '@phosphor-icons/react'
import type { RAGResponse } from '../types'

interface Props { result: RAGResponse | null; state: string; onFeedback: (rating: -1 | 1) => Promise<void> }

export function AnswerPanel({ result, state, onFeedback }: Props) {
  return (
    <article className="answer-panel panel">
      <div className="panel-heading">
        <div><span className="section-index">02</span><h2>Grounded answer</h2></div>
        {result && <span className={`grounding-badge ${result.grounded ? 'grounded' : 'refused'}`}>
          {result.grounded ? <CheckCircle size={16} weight="fill" /> : <WarningCircle size={16} weight="fill" />}
          {result.grounded ? 'VERIFIED' : 'REFUSED'}
        </span>}
      </div>
      {!result && state !== 'processing' && <div className="empty-state"><Info size={28} /><p>Your grounded answer will appear here after the pipeline finds supporting evidence.</p></div>}
      {state === 'processing' && <div className="answer-skeleton" aria-label="Generating grounded answer"><span /><span /><span /></div>}
      {result && <>
        <p className="answer-copy">{result.answer}</p>
        <div className="answer-meta">
          <span>CONFIDENCE <strong>{Math.round(result.confidence * 100)}%</strong></span>
          <span>GENERATOR <strong>{result.generator}</strong></span>
          {result.refusal_reason && <span>REASON <strong>{result.refusal_reason}</strong></span>}
        </div>
        <div className="feedback-row"><span>Was this evidence useful?</span>
          <button type="button" onClick={() => void onFeedback(1)} aria-label="Mark answer useful"><ThumbsUp size={18} /></button>
          <button type="button" onClick={() => void onFeedback(-1)} aria-label="Mark answer not useful"><ThumbsDown size={18} /></button>
        </div>
      </>}
    </article>
  )
}
