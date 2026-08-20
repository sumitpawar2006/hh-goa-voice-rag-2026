import type { HealthResponse, RAGResponse, VoiceQueryResponse } from '../types'

const API_BASE = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '') ?? ''

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = `Request failed with status ${response.status}`
    try {
      const body = (await response.json()) as { detail?: string }
      if (body.detail) message = body.detail
    } catch { /* Safe HTTP status message remains available. */ }
    throw new Error(message)
  }
  return (await response.json()) as T
}
export async function getHealth(): Promise<HealthResponse> { return parseResponse(await fetch(`${API_BASE}/health`)) }
export async function queryKnowledge(question: string): Promise<RAGResponse> {
  return parseResponse(await fetch(`${API_BASE}/query`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question }) }))
}
export async function voiceQuery(blob: Blob): Promise<VoiceQueryResponse> {
  const extension = blob.type.includes('ogg') ? 'ogg' : 'webm'
  const form = new FormData(); form.append('audio', blob, `recording.${extension}`)
  return parseResponse(await fetch(`${API_BASE}/voice-query`, { method: 'POST', body: form }))
}
export async function sendFeedback(requestId: string, rating: -1 | 1): Promise<void> {
  await parseResponse(await fetch(`${API_BASE}/feedback`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ request_id: requestId, rating }) }))
}
