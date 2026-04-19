import { useParams, useNavigate } from 'react-router-dom'
import { CheckCircle } from 'lucide-react'

export default function CompletePage() {
  const { caseId } = useParams<{ caseId: string }>()
  const navigate = useNavigate()

  return (
    <div className="min-h-screen flex items-center justify-center p-8"
      style={{ background: 'var(--bg)' }}>
      <div className="rounded-2xl p-12 text-center w-full max-w-lg"
        style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>

        <CheckCircle size={48} className="mx-auto mb-6 text-emerald-400" />
        <h1 className="text-xl font-bold mb-2">Review Complete</h1>
        <p className="text-sm mb-8" style={{ color: 'var(--muted)' }}>
          All decisions have been applied to the knowledge graph.
          KGE training is running in the background.
        </p>

        <button
          onClick={() => navigate(`/query/${caseId}`)}
          className="w-full py-3 rounded-lg font-bold text-sm mb-3 transition-colors"
          style={{ background: 'var(--accent)', color: 'white' }}
        >
          ⚡ Start Asking Questions →
        </button>
        <button
          onClick={() => navigate('/')}
          className="w-full py-3 rounded-lg text-sm transition-colors"
          style={{ background: 'var(--surface2)', color: 'var(--muted2)', border: '1px solid var(--border2)' }}
        >
          ← Back to Cases
        </button>
      </div>
    </div>
  )
}