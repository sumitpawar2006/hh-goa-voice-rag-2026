import { CaretDown, Database } from '@phosphor-icons/react'
import type { SourceReference } from '../types'

interface Props { sources: SourceReference[]; retrievalCount: number }

export function SourcesPanel({ sources, retrievalCount }: Props) {
  return <section className="sources-panel panel" aria-labelledby="sources-title">
    <div className="panel-heading"><div><span className="section-index">04</span><h2 id="sources-title">Retrieved evidence</h2></div><span className="technical-tag">{retrievalCount} RETRIEVED / {sources.length} CITED</span></div>
    {sources.length === 0 ? <div className="empty-state compact"><Database size={24} /><p>No cited chunks yet. Retrieval evidence appears after a supported answer.</p></div> : <div className="source-list">
      {sources.map((source, index) => <details key={source.chunk_id} className="source-card" open={index === 0}>
        <summary><span className="source-rank">{String(index + 1).padStart(2, '0')}</span><span className="source-id"><strong>{source.document_id}</strong><small>{source.chunk_id}</small></span><span className="score"><strong>{(source.similarity_score * 100).toFixed(1)}%</strong><small>SIMILARITY</small></span><CaretDown size={18} aria-hidden="true" /></summary>
        <div className="source-body"><p>{source.text}</p><dl>
          <div><dt>strategy</dt><dd>{source.strategy}</dd></div><div><dt>language</dt><dd>{String(source.metadata.language ?? 'unknown')}</dd></div><div><dt>query type</dt><dd>{String(source.metadata.query_type ?? 'unknown')}</dd></div><div><dt>position</dt><dd>{source.position}</dd></div>
        </dl></div>
      </details>)}
    </div>}
  </section>
}
