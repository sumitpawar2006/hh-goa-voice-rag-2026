import { useEffect, useMemo, useState } from 'react'
import {
  ArrowRight,
  Database,
  GithubLogo,
  Microphone,
  PaperPlaneTilt,
  ShieldCheck,
  Stop,
  Waveform,
} from '@phosphor-icons/react'
import { AnswerPanel } from './components/AnswerPanel'
import { PerformancePanel } from './components/PerformancePanel'
import { PipelineTrace } from './components/PipelineTrace'
import { SourcesPanel } from './components/SourcesPanel'
import { getHealth, queryKnowledge, sendFeedback, voiceQuery } from './lib/api'
import type { HealthResponse, RAGResponse, Transcript } from './types'
import './App.css'

type ViewState = 'idle' | 'recording' | 'processing' | 'complete' | 'error'

const suggestedQueries = [
  'मैनहट्टन परियोजना की सफलता का तत्काल प्रभाव क्या था?',
  'What does the knowledge base say about the Manhattan Project?',
]

function App() {
  const [question, setQuestion] = useState('')
  const [viewState, setViewState] = useState<ViewState>('idle')
  const [result, setResult] = useState<RAGResponse | null>(null)
  const [transcript, setTranscript] = useState<Transcript | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [recorder, setRecorder] = useState<MediaRecorder | null>(null)

  useEffect(() => {
    void getHealth().then(setHealth).catch(() => setHealth(null))
  }, [])

  const systemReady = health?.services.vector_store.status === 'ready'
  const sttReady = health?.services.speech_to_text.ready ?? false
  const totalLatency = useMemo(
    () => (result ? result.latency.total_ms + (transcript?.latency_ms ?? 0) : 0),
    [result, transcript],
  )

  async function runTextQuery(query = question) {
    const normalized = query.trim()
    if (!normalized) return
    setQuestion(normalized)
    setViewState('processing')
    setResult(null)
    setTranscript(null)
    setError(null)
    try {
      setResult(await queryKnowledge(normalized))
      setViewState('complete')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'The query could not be completed.')
      setViewState('error')
    }
  }

  async function startRecording() {
    setError(null)
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      setError('This browser does not support microphone recording.')
      setViewState('error')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const preferredType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg'].find((type) =>
        MediaRecorder.isTypeSupported(type),
      )
      const mediaRecorder = new MediaRecorder(stream, preferredType ? { mimeType: preferredType } : {})
      const chunks: Blob[] = []
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunks.push(event.data)
      }
      mediaRecorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop())
        void submitVoice(new Blob(chunks, { type: mediaRecorder.mimeType || 'audio/webm' }))
      }
      mediaRecorder.start(250)
      setRecorder(mediaRecorder)
      setViewState('recording')
      setResult(null)
      setTranscript(null)
    } catch {
      setError('Microphone permission was denied or no microphone is available.')
      setViewState('error')
    }
  }

  function stopRecording() {
    if (recorder?.state === 'recording') {
      recorder.stop()
      setRecorder(null)
      setViewState('processing')
    }
  }

  async function submitVoice(blob: Blob) {
    setViewState('processing')
    setError(null)
    try {
      const response = await voiceQuery(blob)
      setTranscript(response.transcript)
      setQuestion(response.transcript.text)
      setResult(response.result)
      setViewState('complete')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Voice processing failed.')
      setViewState('error')
    }
  }

  async function handleFeedback(rating: -1 | 1) {
    if (result) await sendFeedback(result.request_id, rating)
  }

  const isBusy = viewState === 'processing'

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">Skip to main content</a>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="NEXUS home">
          <span className="brand-mark" aria-hidden="true"><Waveform size={22} weight="bold" /></span>
          <span>NEXUS</span><span className="brand-divider" aria-hidden="true" />
          <span className="brand-subtitle">VOICE RAG</span>
        </a>
        <div className="topbar-actions">
          <span className={`system-status ${systemReady ? 'online' : 'offline'}`}>
            <span className="status-dot" aria-hidden="true" />
            {systemReady ? 'INDEX ONLINE' : 'INDEX OFFLINE'}
          </span>
          <a className="icon-link" href="https://github.com/sumitpawar2006/hh-goa-voice-rag-2026" aria-label="Open project repository">
            <GithubLogo size={20} aria-hidden="true" />
          </a>
        </div>
      </header>

      <main id="main-content">
        <section className="hero" id="top" aria-labelledby="page-title">
          <div className="eyebrow"><span>HH GOA 2026</span><span>AI4BHARAT MSMARCO-XI</span></div>
          <h1 id="page-title">Ask the knowledge base.<br /><span>Hear only what it knows.</span></h1>
          <p className="hero-copy">A multilingual voice search engine that retrieves evidence, generates a grounded answer, and exposes every decision in the pipeline.</p>

          <div className={`voice-stage ${viewState}`} aria-busy={isBusy}>
            <div className="voice-grid" aria-hidden="true" />
            <div className="voice-control">
              <div className="pulse-ring ring-one" aria-hidden="true" /><div className="pulse-ring ring-two" aria-hidden="true" />
              <button className="mic-button" type="button" onClick={viewState === 'recording' ? stopRecording : startRecording} disabled={isBusy} aria-label={viewState === 'recording' ? 'Stop recording' : 'Start voice recording'}>
                {viewState === 'recording' ? <Stop size={38} weight="fill" /> : <Microphone size={42} weight="bold" />}
              </button>
            </div>
            <p className="voice-kicker">{viewState === 'recording' ? 'LISTENING — TAP TO STOP' : isBusy ? 'PROCESSING REQUEST' : 'TAP TO SPEAK'}</p>
            <p className="voice-hint">{sttReady ? 'ElevenLabs Scribe v2 is ready' : 'Live STT requires ELEVENLABS_API_KEY'}</p>
          </div>

          <div className="or-divider"><span>OR TYPE A QUESTION</span></div>
          <form className="query-form" onSubmit={(event) => { event.preventDefault(); void runTextQuery() }}>
            <label className="sr-only" htmlFor="question">Question for the knowledge base</label>
            <textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') void runTextQuery() }} placeholder="Ask in English, Hindi, Marathi, Tamil, or another indexed language…" maxLength={1000} rows={2} disabled={isBusy || viewState === 'recording'} />
            <button type="submit" disabled={isBusy || !question.trim()} aria-label="Submit question"><PaperPlaneTilt size={22} weight="fill" aria-hidden="true" /></button>
          </form>
          <div className="suggestions" aria-label="Suggested questions">
            {suggestedQueries.map((query) => <button key={query} type="button" onClick={() => void runTextQuery(query)} disabled={isBusy}>{query}<ArrowRight size={15} aria-hidden="true" /></button>)}
          </div>
        </section>

        <section className="workspace" aria-label="RAG result workspace">
          <PipelineTrace trace={result?.trace ?? []} state={viewState} hasTranscript={Boolean(transcript)} />
          {transcript && <article className="transcript-panel panel"><div className="panel-heading"><div><span className="section-index">01</span><h2>Transcription</h2></div><span className="technical-tag">{transcript.provider} / {transcript.model}</span></div><blockquote>{transcript.text}</blockquote><p>{transcript.language_code ?? 'auto-detected'} · {(transcript.language_probability ?? 0).toLocaleString(undefined, { style: 'percent' })} confidence</p></article>}
          {error && <div className="error-banner" role="alert"><ShieldCheck size={22} aria-hidden="true" /><div><strong>Request stopped safely</strong><p>{error}</p></div></div>}
          <div className="result-grid">
            <AnswerPanel result={result} state={viewState} onFeedback={handleFeedback} />
            <PerformancePanel result={result} sttLatency={transcript?.latency_ms ?? null} totalLatency={totalLatency} />
          </div>
          <SourcesPanel sources={result?.sources ?? []} retrievalCount={result?.retrieval_count ?? 0} />
        </section>

        <section className="proof-strip" aria-label="System implementation summary">
          <div><Database size={24} aria-hidden="true" /><span>VECTOR STORE</span><strong>Qdrant / cosine</strong></div>
          <div><Waveform size={24} aria-hidden="true" /><span>EMBEDDINGS</span><strong>Multilingual MiniLM</strong></div>
          <div><ShieldCheck size={24} aria-hidden="true" /><span>GUARDRAILS</span><strong>Input + grounding</strong></div>
        </section>
      </main>
      <footer><span>NEXUS / HH GOA 2026</span><span>Grounded by design · #RAGInGoa</span></footer>
      <span className="sr-only" aria-live="polite">{viewState === 'processing' ? 'Processing your request' : viewState === 'complete' ? 'Answer ready' : ''}</span>
    </div>
  )
}

export default App
