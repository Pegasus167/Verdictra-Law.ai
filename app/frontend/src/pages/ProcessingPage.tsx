import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Loader2, CheckCircle, Circle, AlertCircle, RefreshCw } from 'lucide-react'
import { api } from '../lib/api'

const STAGES = [
  { num: 1, label: 'PDF extraction & OCR',              detail: 'pdfplumber + parallel Tesseract' },
  { num: 2, label: 'Document tree construction',         detail: 'Markdown → hierarchical tree' },
  { num: 3, label: 'Entity extraction (GLiNER)',         detail: 'Named entity recognition' },
  { num: 4, label: 'Knowledge graph construction',       detail: 'Neo4j + relationship extraction' },
  { num: 5, label: 'Entity resolution',                  detail: 'Clustering + LLM scoring' },
]

export default function ProcessingPage() {
  const { caseId } = useParams<{ caseId: string }>()
  const navigate    = useNavigate()

  const [currentStage, setCurrentStage] = useState(1)
  const [failed, setFailed]             = useState(false)
  const [errorMsg, setErrorMsg]         = useState('')
  const [retrying, setRetrying]         = useState(false)

  useEffect(() => {
    if (!caseId) return

    const interval = setInterval(async () => {
      try {
        const c = await api.getCase(caseId)

        // Update live stage indicator
        if (c.current_stage) setCurrentStage(c.current_stage)

        // Navigate on completion
        if (c.status === 'review') {
          clearInterval(interval)
          navigate(`/summary/${caseId}`)
        } else if (c.status === 'ready') {
          clearInterval(interval)
          navigate(`/query/${caseId}`)
        } else if (c.status === 'failed') {
          clearInterval(interval)
          setFailed(true)
          setErrorMsg(c.error || 'Ingestion failed. Please try again.')
        }
      } catch {}
    }, 5_000)

    return () => clearInterval(interval)
  }, [caseId, navigate])

  async function handleRetry() {
    if (!caseId) return
    setRetrying(true)
    setFailed(false)
    setErrorMsg('')
    setCurrentStage(1)

    try {
      // Re-trigger ingestion by calling a retry endpoint
      const res = await fetch(`/api/retry/${caseId}`, { method: 'POST' })
      if (!res.ok) throw new Error('Retry failed')
      // Polling will pick up the new status automatically
    } catch {
      setFailed(true)
      setErrorMsg('Retry failed. Please delete this case and upload again.')
    } finally {
      setRetrying(false)
    }
  }

  function handleDelete() {
    if (!caseId) return
    navigate('/')
  }

  // ── Failed state ────────────────────────────────────────────────────────────
  if (failed) {
    return (
      <div className="min-h-screen flex items-center justify-center p-8"
        style={{ background: 'var(--bg)' }}>
        <div className="rounded-2xl p-12 text-center w-full max-w-lg"
          style={{ background: 'var(--surface)', border: '1px solid #7f1d1d' }}>

          <AlertCircle size={48} className="mx-auto mb-6 text-red-400" />
          <h1 className="text-xl font-bold mb-2 text-red-400">Ingestion Failed</h1>
          <p className="text-sm mb-2" style={{ color: 'var(--muted)' }}>
            Something went wrong while processing this case.
          </p>
          {errorMsg && (
            <p className="text-xs mb-6 px-4 py-2 rounded"
              style={{ background: '#1c0a0a', border: '1px solid #7f1d1d', color: '#fca5a5' }}>
              {errorMsg}
            </p>
          )}

          <div className="flex gap-3 justify-center">
            <button
              onClick={handleRetry}
              disabled={retrying}
              className="flex items-center gap-2 px-6 py-2.5 rounded-lg text-sm font-bold"
              style={{ background: 'var(--accent)', color: 'white' }}>
              {retrying
                ? <Loader2 size={14} className="animate-spin" />
                : <RefreshCw size={14} />}
              {retrying ? 'Retrying...' : 'Retry Ingestion'}
            </button>
            <button
              onClick={() => navigate('/')}
              className="px-6 py-2.5 rounded-lg text-sm"
              style={{ background: 'var(--surface2)', color: 'var(--muted2)', border: '1px solid var(--border2)' }}>
              ← Back to Cases
            </button>
          </div>
        </div>
      </div>
    )
  }

  // ── Processing state ────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen flex items-center justify-center p-8"
      style={{ background: 'var(--bg)' }}>
      <div className="rounded-2xl p-12 text-center w-full max-w-lg"
        style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>

        <Loader2 size={48} className="animate-spin mx-auto mb-6 text-indigo-400" />
        <h1 className="text-xl font-bold mb-2">Processing Case</h1>
        <p className="text-sm mb-8" style={{ color: 'var(--muted)' }}>
          The ingestion pipeline is running. This takes 5–15 minutes
          depending on document size.
        </p>

        {/* Live stage indicators */}
        <div className="text-left flex flex-col gap-2 mb-8">
          {STAGES.map(stage => {
            const done    = currentStage > stage.num
            const active  = currentStage === stage.num
            const pending = currentStage < stage.num

            return (
              <div key={stage.num}
                className="flex items-center gap-3 px-4 py-3 rounded-lg text-xs transition-all"
                style={{
                  background: done   ? '#05261a'
                             : active ? '#1a1f35'
                             : 'var(--bg)',
                  border: `1px solid ${
                    done   ? '#065f46'
                  : active ? 'var(--accent)'
                  : 'var(--border)'
                  }`,
                }}>
                {done ? (
                  <CheckCircle size={14} className="text-emerald-400 flex-shrink-0" />
                ) : active ? (
                  <Loader2 size={14} className="animate-spin text-indigo-400 flex-shrink-0" />
                ) : (
                  <Circle size={14} className="flex-shrink-0" style={{ color: 'var(--muted)' }} />
                )}
                <div className="flex-1 min-w-0">
                  <div style={{
                    color: done ? '#10b981' : active ? 'var(--text)' : 'var(--muted)',
                    fontWeight: active ? 600 : 400,
                  }}>
                    Stage {stage.num} — {stage.label}
                  </div>
                  {active && (
                    <div className="text-xs mt-0.5" style={{ color: 'var(--muted)' }}>
                      {stage.detail}...
                    </div>
                  )}
                </div>
              </div>
            )
          })}
        </div>

        <div className="rounded-lg px-4 py-3 text-xs text-yellow-400"
          style={{ background: '#1c100322', border: '1px solid #78350f' }}>
          You can close this tab and come back later. The pipeline continues
          running in the background.
        </div>
      </div>
    </div>
  )
}