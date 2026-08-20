import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { SourcesPanel } from './SourcesPanel'

describe('SourcesPanel', () => {
  it('renders provenance, score, strategy, and source text', () => {
    render(
      <SourcesPanel
        retrievalCount={5}
        sources={[{
          document_id: 'doc_123',
          chunk_id: 'chk_456',
          text: 'Retrieved factual evidence.',
          source: 'hf://dataset',
          similarity_score: 0.91,
          metadata: { language: 'hi', query_type: 'DESCRIPTION' },
          strategy: 'semantic',
          position: 0,
        }]}
      />,
    )
    expect(screen.getByText('doc_123')).toBeInTheDocument()
    expect(screen.getByText('91.0%')).toBeInTheDocument()
    expect(screen.getByText('Retrieved factual evidence.')).toBeInTheDocument()
    expect(screen.getByText('semantic')).toBeInTheDocument()
  })
})
