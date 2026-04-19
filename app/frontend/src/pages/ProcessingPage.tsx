import { useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Loader2 } from 'lucide-react'
import { api } from '../lib/api'

const STEPS = [
  'Stage 1 — PDF extraction & OCR',
  'Stage 2 — Entity extraction (GLiNER)',
  'Stage 3 — Knowledge graph construction',
  'Stage 4 — Entity resolution (human review)',
]

export default function ProcessingPage() {
  const { caseId } = useParams<{ caseId: string }>()
  const navigate = useNavigate()

  // Poll case status every 10s
  useEffect(() => {
    if (!caseId) return
    const interval = setInterval(async () => {
      try {
        const c = await api.getCase(caseId)
        if (c.status === 'review') navigate(`/summary/${caseId}`)
        else if (c.status === 'ready') navigate(`/query/${caseId}`)
      } catch {}
    }, 10_000)
    return () => clearInterval(interval)
  }, [caseId, navigate])

  return (
    <div className="min-h-screen flex items-center justify-center p-8"
      style={{ background: 'var(--bg)' }}>
      <div className="rounded-2xl p-12 text-center w-full max-w-lg"
        style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>

        <Loader2 size={48} className="animate-spin mx-auto mb-6 text-indigo-400" />
        <h1 className="text-xl font-bold mb-2">Processing Case</h1>
        <p className="text-sm mb-6" style={{ color: 'var(--muted)' }}>
          The ingestion pipeline is running. This takes approximately 10–15 minutes
          for a typical case document.
        </p>

        <div className="text-left flex flex-col gap-2 mb-8">
          {STEPS.map(step => (
            <div key={step} className="flex items-center gap-3 px-3 py-2 rounded-lg text-xs"
              style={{ background: 'var(--bg)', border: '1px solid var(--border)', color: 'var(--muted2)' }}>
              <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: 'var(--muted)' }} />
              {step}
            </div>
          ))}
        </div>

        <div className="rounded-lg px-4 py-3 text-xs text-yellow-400"
          style={{ background: '#1c100322', border: '1px solid #78350f' }}>
          You can close this tab and come back later. The pipeline will continue
          running in the background.
        </div>
      </div>
    </div>
  )
}