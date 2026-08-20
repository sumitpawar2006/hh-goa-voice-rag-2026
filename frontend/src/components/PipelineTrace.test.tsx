import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { PipelineTrace } from './PipelineTrace'

describe('PipelineTrace', () => {
  it('exposes the full RAG pipeline and measured stages', () => {
    render(
      <PipelineTrace
        state="complete"
        hasTranscript
        trace={[{ stage: 'embedding', status: 'passed', latency_ms: 4.25 }]}
      />,
    )
    expect(screen.getByRole('heading', { name: /live pipeline/i })).toBeInTheDocument()
    expect(screen.getByText('VOICE')).toBeInTheDocument()
    expect(screen.getByText('STT')).toBeInTheDocument()
    expect(screen.getByText('RETRIEVE')).toBeInTheDocument()
    expect(screen.getByText('4.3 ms')).toBeInTheDocument()
  })
})
