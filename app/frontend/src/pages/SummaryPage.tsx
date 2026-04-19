import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { CheckCircle, Clock, Minus, AlertTriangle, Loader2 } from 'lucide-react'
import { api } from '../lib/api'
import type { ResolutionState, Group, Candidate } from '../types'

export default function SummaryPage() {
  const { caseId } = useParams<{ caseId: string }>()
  const navigate = useNavigate()

  const [state, setState] = useState<ResolutionState | null>(null)
  const [caseName, setCaseName] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!caseId) return
    Promise.all([
      api.getResolutionState(caseId),
      api.getCase(caseId),
    ])
      .then(([s, c]) => {
        setState(s)
        setCaseName(c.case_name)
      })
      .catch(() => setError('Failed to load resolution summary'))
      .finally(() => setLoading(false))
  }, [caseId])

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center"
        style={{ background: 'var(--bg)' }}>
        <Loader2 size={32} className="animate-spin text-indigo-400" />
      </div>
    )
  }

  if (error || !state) {
    return (
      <div className="min-h-screen flex items-center justify-center"
        style={{ background: 'var(--bg)' }}>
        <div className="text-sm text-red-400">{error || 'Resolution state not found'}</div>
      </div>
    )
  }

  const { summary, auto_merge, needs_review, auto_keep } = state
  const createdAt = new Date(state.created_at).toLocaleString('en-IN', {
    day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })

  return (
    <div className="min-h-screen" style={{ background: 'var(--bg)' }}>

      {/* Header */}
      <header className="h-14 flex items-center justify-between px-10"
        style={{ background: 'var(--surface)', borderBottom: '1px solid var(--border)' }}>
        <div>
          <span className="font-bold text-indigo-300 tracking-wider text-sm">LAW.ai</span>
          <span className="text-sm ml-2" style={{ color: 'var(--muted)' }}>/ {caseName} / Resolution Summary</span>
        </div>
        <span className="text-xs" style={{ color: 'var(--muted)' }}>
          Processed {createdAt}
        </span>
      </header>

      <main className="max-w-4xl mx-auto px-10 py-10">

        {/* Stats */}
        <div className="grid grid-cols-3 gap-5 mb-10">
          <div className="rounded-xl p-7 text-center"
            style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
            <div className="text-4xl font-bold mb-2 text-emerald-400">
              {summary.auto_merge_count}
            </div>
            <div className="text-xs flex items-center justify-center gap-1"
              style={{ color: 'var(--muted)' }}>
              <CheckCircle size={12} /> Auto-merged
            </div>
          </div>
          <div className="rounded-xl p-7 text-center"
            style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
            <div className="text-4xl font-bold mb-2 text-yellow-400">
              {summary.needs_review_count}
            </div>
            <div className="text-xs flex items-center justify-center gap-1"
              style={{ color: 'var(--muted)' }}>
              <Clock size={12} /> Need your review
            </div>
          </div>
          <div className="rounded-xl p-7 text-center"
            style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
            <div className="text-4xl font-bold mb-2 text-slate-400">
              {summary.auto_keep_count}
            </div>
            <div className="text-xs flex items-center justify-center gap-1"
              style={{ color: 'var(--muted)' }}>
              <Minus size={12} /> Kept separate
            </div>
          </div>
        </div>

        {/* Action */}
        <div className="flex gap-3 mb-10">
          {summary.needs_review_count > 0 ? (
            <button
              onClick={() => navigate(`/review/${caseId}`)}
              className="px-8 py-3 rounded-lg font-bold text-sm transition-colors"
              style={{ background: 'var(--accent)', color: 'white' }}>
              ▶ Start Review ({summary.needs_review_count} groups) →
            </button>
          ) : (
            <button
              onClick={() => navigate(`/query/${caseId}`)}
              className="px-8 py-3 rounded-lg font-bold text-sm transition-colors"
              style={{ background: '#10b981', color: 'white' }}>
              ✓ All resolved — Start Querying →
            </button>
          )}
          <button
            onClick={() => navigate('/')}
            className="px-6 py-3 rounded-lg text-sm transition-colors"
            style={{ background: 'var(--surface2)', color: 'var(--muted2)', border: '1px solid var(--border2)' }}>
            ← Cases
          </button>
        </div>

        {/* Auto-merged groups */}
        <div className="rounded-xl overflow-hidden mb-6"
          style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
          <div className="flex items-center justify-between px-6 py-4"
            style={{ borderBottom: '1px solid var(--border)' }}>
            <h2 className="text-sm font-semibold">✅ Auto-merged groups</h2>
            <span className="text-xs px-3 py-1 rounded-full"
              style={{ background: '#05261a', border: '1px solid #065f46', color: '#10b981' }}>
              {auto_merge.length} groups
            </span>
          </div>
          {auto_merge.length === 0 ? (
            <div className="px-6 py-8 text-center text-xs" style={{ color: 'var(--muted)' }}>
              No auto-merges
            </div>
          ) : (
            <div className="divide-y" style={{ borderColor: 'var(--border)' }}>
              {auto_merge.map(group => (
                <div key={group.group_id} className="px-6 py-3 flex items-center gap-3 flex-wrap">
                  <span className="text-xs px-2 py-0.5 rounded"
                    style={{ background: '#1e1b4b', color: '#818cf8', border: '1px solid #3730a3' }}>
                    {group.schema_type}
                  </span>
                  <div className="flex flex-wrap gap-1 flex-1">
                    {group.candidates.map(c => (
                      <span key={c.canonical_name}
                        className="text-xs px-2 py-0.5 rounded"
                        style={{ background: 'var(--bg)', border: '1px solid var(--border2)', color: 'var(--muted2)' }}>
                        {c.text}
                        <span className="ml-1" style={{ color: 'var(--muted)' }}>p.{c.source_page}</span>
                      </span>
                    ))}
                  </div>
                  <span style={{ color: 'var(--muted)' }}>→</span>
                  <span className="text-xs font-bold text-indigo-300">{group.canonical_name}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Needs review preview */}
        {needs_review.length > 0 && (
          <div className="rounded-xl overflow-hidden mb-6"
            style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
            <div className="flex items-center justify-between px-6 py-4"
              style={{ borderBottom: '1px solid var(--border)' }}>
              <h2 className="text-sm font-semibold flex items-center gap-2">
                <AlertTriangle size={14} className="text-yellow-400" />
                Groups needing your review
              </h2>
              <span className="text-xs px-3 py-1 rounded-full"
                style={{ background: '#1c1003', border: '1px solid #78350f', color: '#f59e0b' }}>
                {needs_review.length} groups
              </span>
            </div>
            <div className="divide-y" style={{ borderColor: 'var(--border)' }}>
              {needs_review.slice(0, 10).map(group => (
                <div key={group.group_id} className="px-6 py-3 flex items-center gap-3 flex-wrap">
                  <span className="text-xs px-2 py-0.5 rounded"
                    style={{ background: '#1e1b4b', color: '#818cf8', border: '1px solid #3730a3' }}>
                    {group.schema_type}
                  </span>
                  <div className="flex flex-wrap gap-1 flex-1">
                    {group.candidates.slice(0, 4).map(c => (
                      <span key={c.canonical_name}
                        className="text-xs px-2 py-0.5 rounded"
                        style={{ background: 'var(--bg)', border: '1px solid var(--border2)', color: 'var(--muted2)' }}>
                        {c.text}
                      </span>
                    ))}
                    {group.candidates.length > 4 && (
                      <span className="text-xs" style={{ color: 'var(--muted)' }}>
                        +{group.candidates.length - 4} more
                      </span>
                    )}
                  </div>
                </div>
              ))}
              {needs_review.length > 10 && (
                <div className="px-6 py-3 text-xs" style={{ color: 'var(--muted)' }}>
                  ... and {needs_review.length - 10} more groups
                </div>
              )}
            </div>
          </div>
        )}

        {/* Auto-kept separate */}
        {auto_keep.length > 0 && (
          <div className="rounded-xl overflow-hidden"
            style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
            <div className="flex items-center justify-between px-6 py-4"
              style={{ borderBottom: '1px solid var(--border)' }}>
              <h2 className="text-sm font-semibold">— Kept separate (distinct entities)</h2>
              <span className="text-xs px-3 py-1 rounded-full"
                style={{ background: 'var(--surface2)', border: '1px solid var(--border2)', color: 'var(--muted)' }}>
                {auto_keep.length} groups
              </span>
            </div>
            <div className="divide-y" style={{ borderColor: 'var(--border)' }}>
              {auto_keep.map((group: Group) => (
                <div key={group.group_id} className="px-6 py-3 flex items-center gap-3 flex-wrap">
                  <span className="text-xs px-2 py-0.5 rounded"
                    style={{ background: '#1e1b4b', color: '#818cf8', border: '1px solid #3730a3' }}>
                    {group.schema_type}
                  </span>
                  <div className="flex flex-wrap gap-1">
                    {group.candidates.map((c: Candidate) => (
                      <span key={c.canonical_name}
                        className="text-xs px-2 py-0.5 rounded"
                        style={{ background: 'var(--bg)', border: '1px solid var(--border2)', color: 'var(--muted2)' }}>
                        {c.text}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}