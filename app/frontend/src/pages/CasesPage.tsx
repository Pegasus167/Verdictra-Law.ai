import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, Scale, Clock, AlertCircle, Loader2, Trash2, User } from 'lucide-react'
import { api } from '../lib/api'
import { formatDate } from '../lib/utils'
import type { Case } from '../types'
import WelcomePage from './WelcomePage'

const STATUS_CONFIG = {
  ready:      { label: 'Ready',        color: 'text-emerald-400', dot: 'bg-emerald-400' },
  review:     { label: 'Needs Review', color: 'text-yellow-400',  dot: 'bg-yellow-400'  },
  processing: { label: 'Processing',   color: 'text-slate-400',   dot: 'bg-slate-400'   },
}

interface Domain {
  id: string
  name: string
  description: string
}

export default function CasesPage() {
  const [cases, setCases]           = useState<Case[]>([])
  const [domains, setDomains]       = useState<Domain[]>([])
  const [loading, setLoading]       = useState(true)
  const [uploading, setUploading]   = useState(false)
  const [caseName, setCaseName]     = useState('')
  const [files, setFiles]           = useState<File[]>([])
  const [domain, setDomain]         = useState('constitutional')
  const [error, setError]           = useState('')
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const fileRef                     = useRef<HTMLInputElement>(null)
  const navigate                    = useNavigate()

  // Welcome screen — shown on first login only
  const [showWelcome, setShowWelcome] = useState(
    localStorage.getItem('first_login') === 'true'
  )

  useEffect(() => {
    api.getCases()
      .then(setCases)
      .catch(() => setError('Failed to load cases'))
      .finally(() => setLoading(false))

    api.getDomains()
      .then(d => { if (d.length > 0) setDomains(d) })
      .catch(() => {})
  }, [])

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault()
    if (!caseName.trim() || files.length === 0) return
    setUploading(true)
    setError('')
    try {
      const { case_id } = await api.uploadCase(caseName.trim(), files, domain)
      navigate(`/processing/${case_id}`)
    } catch (err: any) {
      setError(err.message || 'Upload failed. Please try again.')
      setUploading(false)
    }
  }

  function openCase(c: Case) {
    if (c.status === 'ready') navigate(`/query/${c.case_id}`)
    else if (c.status === 'review') navigate(`/review/${c.case_id}`)
    else navigate(`/processing/${c.case_id}`)
  }

  async function deleteCase(e: React.MouseEvent, caseId: string) {
    e.stopPropagation()
    if (!window.confirm('Delete this case? This cannot be undone.')) return
    setDeletingId(caseId)
    try {
      await api.deleteCase(caseId)
      setCases(prev => prev.filter(c => c.case_id !== caseId))
    } catch {
      setError('Failed to delete case. Please try again.')
    } finally {
      setDeletingId(null)
    }
  }

  const domainOptions: Domain[] = domains.length > 0 ? domains : [
    { id: 'constitutional', name: 'Constitutional & Writ',   description: '' },
    { id: 'property',       name: 'Property & Real Estate',  description: '' },
    { id: 'banking_finance',name: 'Banking & Finance',       description: '' },
    { id: 'corporate',      name: 'Corporate & Company Law', description: '' },
    { id: 'criminal',       name: 'Criminal Law',            description: '' },
    { id: 'tax',            name: 'Tax Law',                 description: '' },
    { id: 'labour',         name: 'Labour & Employment',     description: '' },
    { id: 'ip_patent',      name: 'IP & Patents',            description: '' },
  ]

  // Detect plan limit error from upload response
  const isPlanLimitError = error.toLowerCase().includes('case limit') ||
                           error.toLowerCase().includes('upgrade')

  return (
    <div className="min-h-screen" style={{ background: 'var(--bg)' }}>

      {/* Welcome overlay — first login only, not skippable until last step */}
      {showWelcome && (
        <WelcomePage
          userName={localStorage.getItem('user_name') || ''}
          onDismiss={() => {
            setShowWelcome(false)
            localStorage.removeItem('first_login')
          }}
        />
      )}

      {/* Header */}
      <header style={{ background: 'var(--surface)', borderBottom: '1px solid var(--border)' }}
        className="h-14 flex items-center justify-between px-10">
        <div className="flex items-center gap-2">
          <img src="/logo.png" alt="Verdictra" style={{ height: 32, width: 'auto' }} />
          <span className="font-bold tracking-wider text-sm"
            style={{ color: 'var(--accent)', fontFamily: 'Noto Serif, serif' }}>Verdictra</span>
          <span className="text-sm" style={{ color: 'var(--muted)' }}>Legal Intelligence</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs" style={{ color: 'var(--muted)' }}>
            {localStorage.getItem('user_name') || ''}
          </span>
          <button
            onClick={() => navigate('/profile')}
            className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded"
            style={{ background: 'var(--surface2)', color: 'var(--muted2)', border: '1px solid var(--border2)' }}
            title="Profile & plan">
            <User size={12} /> Profile
          </button>
          <button
            onClick={() => { localStorage.clear(); navigate('/login') }}
            className="text-xs px-3 py-1.5 rounded"
            style={{ background: 'var(--surface2)', color: 'var(--muted2)', border: '1px solid var(--border2)' }}>
            Sign out
          </button>
        </div>
      </header>

      <main className="mx-auto px-10 py-12" style={{ maxWidth: 900 }}>
        <h1 className="text-2xl font-bold mb-1">Cases</h1>
        <p style={{ color: 'var(--muted)' }} className="text-sm mb-10">
          Upload a new case or continue working on an existing one.
        </p>

        {/* Upload form */}
        <div style={{ background: 'var(--surface)', border: '1px solid var(--border2)' }}
          className="rounded-xl p-7 mb-10">
          <h2 className="text-xs font-semibold tracking-widest uppercase mb-5"
            style={{ color: 'var(--muted2)' }}>New Case</h2>

          <form onSubmit={handleUpload} className="flex flex-col gap-4">

            {/* Row 1 — case name + files */}
            <div className="flex gap-3 flex-wrap items-end">
              <div className="flex flex-col gap-1.5 flex-1 min-w-48">
                <label className="text-xs" style={{ color: 'var(--muted)' }}>Case name</label>
                <input
                  type="text"
                  value={caseName}
                  onChange={e => setCaseName(e.target.value)}
                  placeholder="e.g. CELIR LLP v. MIDC"
                  required
                  className="rounded-lg px-3 py-2 text-sm outline-none"
                  style={{ background: 'var(--bg)', border: '1px solid var(--border2)', color: 'var(--text)' }}
                />
              </div>

              <div className="flex flex-col gap-1.5 flex-1 min-w-48">
                <label className="text-xs" style={{ color: 'var(--muted)' }}>
                  Files
                  <span className="ml-1 font-normal" style={{ color: 'var(--muted)' }}>
                    (PDF, DOCX, images — select multiple)
                  </span>
                </label>
                <button
                  type="button"
                  onClick={() => fileRef.current?.click()}
                  className="rounded-lg px-3 py-2 text-sm text-left flex items-center gap-2"
                  style={{
                    background: 'var(--bg)',
                    border: `1px solid ${files.length > 0 ? '#10b981' : 'var(--border2)'}`,
                    color: files.length > 0 ? 'var(--text)' : 'var(--muted2)',
                  }}
                >
                  <Upload size={14} />
                  {files.length === 0
                    ? 'Choose files...'
                    : files.length === 1
                      ? files[0].name
                      : `${files.length} files selected`}
                </button>
                {files.length > 1 && (
                  <div className="flex flex-col gap-1 mt-1">
                    {files.map((f, i) => (
                      <div key={i} className="flex items-center gap-2 px-2 py-1 rounded text-xs"
                        style={{ background: 'var(--bg)', border: '1px solid var(--border)', color: 'var(--muted2)' }}>
                        <span className="truncate flex-1">{f.name}</span>
                        <span style={{ color: 'var(--muted)' }}>{(f.size / 1024 / 1024).toFixed(1)} MB</span>
                        <button
                          type="button"
                          onClick={e => { e.stopPropagation(); setFiles(prev => prev.filter((_, idx) => idx !== i)) }}
                          style={{ color: 'var(--muted)' }}>✕</button>
                      </div>
                    ))}
                  </div>
                )}
                <input
                  ref={fileRef}
                  type="file"
                  accept=".pdf,.docx,.doc,.eml,.txt,.jpg,.jpeg,.png"
                  multiple
                  className="hidden"
                  onChange={e => setFiles(Array.from(e.target.files ?? []))}
                />
              </div>
            </div>

            {/* Row 2 — domain + upload button */}
            <div className="flex gap-3 flex-wrap items-end">
              <div className="flex flex-col gap-1.5 flex-1 min-w-48">
                <label className="text-xs" style={{ color: 'var(--muted)' }}>
                  Case domain
                  <span className="ml-1.5 px-1.5 py-0.5 rounded text-xs"
                    style={{ background: 'var(--surface2)', color: 'var(--muted)', border: '1px solid var(--border)' }}>
                    determines entity extraction vocabulary
                  </span>
                </label>
                <select
                  value={domain}
                  onChange={e => setDomain(e.target.value)}
                  className="rounded-lg px-3 py-2 text-sm outline-none"
                  style={{ background: 'var(--bg)', border: '1px solid var(--border2)', color: 'var(--text)', cursor: 'pointer' }}
                >
                  {domainOptions.map(d => (
                    <option key={d.id} value={d.id}>{d.name}</option>
                  ))}
                </select>
                {domainOptions.find(d => d.id === domain)?.description && (
                  <p className="text-xs" style={{ color: 'var(--muted)' }}>
                    {domainOptions.find(d => d.id === domain)?.description}
                  </p>
                )}
              </div>

              <button
                type="submit"
                disabled={uploading || !caseName.trim() || files.length === 0}
                className="px-6 py-2 rounded-lg text-sm font-bold flex items-center gap-2"
                style={{
                  background: uploading || !caseName.trim() || files.length === 0 ? 'var(--border2)' : 'var(--accent)',
                  color: 'white',
                  height: 38,
                  alignSelf: 'flex-end',
                }}
              >
                {uploading && <Loader2 size={14} className="animate-spin" />}
                {uploading ? 'Uploading...' : 'Upload →'}
              </button>
            </div>
          </form>

          {error && (
            <div className="mt-3">
              <p className="text-xs text-red-400 flex items-center gap-1">
                <AlertCircle size={12} /> {error}
              </p>
              {isPlanLimitError && (
                <p className="text-xs mt-1.5" style={{ color: 'var(--muted)' }}>
                  <a
                    href="mailto:support@verdictra.ai?subject=Plan upgrade"
                    style={{ color: 'var(--accent2)', textDecoration: 'none' }}>
                    Email support@verdictra.ai →
                  </a>
                  {' '}to upgrade your plan.
                </p>
              )}
            </div>
          )}
        </div>

        {/* Case list header */}
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xs font-semibold tracking-widest uppercase"
            style={{ color: 'var(--muted2)' }}>Existing Cases</h2>
          <span className="text-xs px-3 py-1 rounded-full"
            style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--muted)' }}>
            {cases.length} case{cases.length !== 1 ? 's' : ''}
          </span>
        </div>

        {loading ? (
          <div className="flex justify-center py-16">
            <Loader2 size={24} className="animate-spin" style={{ color: 'var(--accent2)' }} />
          </div>

        ) : cases.length === 0 ? (

          /* ── Rich empty state ─────────────────────────────────────────────── */
          <div className="rounded-xl p-10 flex flex-col items-center text-center"
            style={{ border: '1px dashed var(--border2)', background: 'var(--surface)' }}>

            <div className="w-16 h-16 rounded-full flex items-center justify-center mb-6"
              style={{ background: 'var(--bg)', border: '1px solid var(--border2)' }}>
              <Scale size={28} style={{ color: 'var(--accent2)', opacity: 0.5 }} />
            </div>

            <h3 className="text-base font-semibold mb-2">Upload your first case</h3>
            <p className="text-sm mb-7 max-w-sm" style={{ color: 'var(--muted)', lineHeight: 1.65 }}>
              Add a case bundle — one matter, as many documents as you need. Verdictra extracts
              every party, date, amount, and order, then lets you query across all of them instantly.
            </p>

            <button
              onClick={() => document.querySelector<HTMLInputElement>('input[placeholder="e.g. CELIR LLP v. MIDC"]')?.focus()}
              className="px-6 py-2.5 rounded-lg text-sm font-bold mb-8"
              style={{ background: 'var(--accent)', color: '#fff', border: 'none', cursor: 'pointer' }}>
              + Start a new case above
            </button>

            {/* Format tip */}
            <div className="w-full max-w-md rounded-lg p-4 text-left mb-5"
              style={{ background: 'var(--bg)', border: '1px solid var(--border)' }}>
              <div className="text-xs font-semibold tracking-widest uppercase mb-3"
                style={{ color: 'var(--muted)' }}>Accepted formats</div>
              <div className="flex flex-wrap gap-2 mb-3">
                {['PDF', 'DOCX', 'Scanned orders', 'Email (.eml)', 'Images'].map(f => (
                  <span key={f} className="px-2 py-0.5 rounded text-xs"
                    style={{ background: 'var(--surface)', border: '1px solid var(--border2)', color: 'var(--muted2)' }}>
                    {f}
                  </span>
                ))}
              </div>
              <p className="text-xs" style={{ color: 'var(--muted)' }}>
                Max 50 MB per upload · Multiple files per case · Processing takes 1–3 min per document
              </p>
            </div>

            {/* What happens next */}
            <div className="w-full max-w-md rounded-lg overflow-hidden mb-6"
              style={{ border: '1px solid var(--border)' }}>
              {[
                { n: '01', t: 'Entity extraction', d: 'Parties, dates, amounts, orders — found automatically.' },
                { n: '02', t: 'Your review',        d: 'Confirm or correct extracted entities before they are committed.' },
                { n: '03', t: 'Instant queries',    d: 'Ask anything — get a cited answer in under 30 seconds.' },
              ].map((s, i) => (
                <div key={s.n} className="flex gap-4 p-4 text-left"
                  style={{ borderBottom: i < 2 ? '1px solid var(--border)' : 'none', background: 'var(--surface)' }}>
                  <span className="text-xs font-bold flex-shrink-0 pt-0.5"
                    style={{ color: 'var(--muted)', letterSpacing: '0.04em' }}>{s.n}</span>
                  <div>
                    <div className="text-xs font-semibold mb-1">{s.t}</div>
                    <div className="text-xs" style={{ color: 'var(--muted)', lineHeight: 1.5 }}>{s.d}</div>
                  </div>
                </div>
              ))}
            </div>

            <p className="text-xs" style={{ color: 'var(--muted)' }}>
              Questions?{' '}
              <a href="mailto:support@verdictra.ai"
                style={{ color: 'var(--accent2)', textDecoration: 'none' }}>
                support@verdictra.ai
              </a>
            </p>
          </div>
          /* ── End empty state ──────────────────────────────────────────────── */

        ) : (
          <div className="flex flex-col gap-2.5">
            {cases.map(c => {
              const cfg = STATUS_CONFIG[c.status as keyof typeof STATUS_CONFIG] ?? STATUS_CONFIG.processing
              return (
                <button
                  key={c.case_id}
                  onClick={() => openCase(c)}
                  className="w-full text-left rounded-xl px-6 py-4 flex items-center gap-4 transition-all group"
                  style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
                  onMouseEnter={e => (e.currentTarget.style.borderColor = 'var(--border2)')}
                  onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--border)')}
                >
                  <div className="w-10 h-10 rounded-xl flex items-center justify-center text-lg flex-shrink-0"
                    style={{ background: '#1a1f35', border: '1px solid var(--border2)' }}>
                    ⚖️
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="font-semibold text-sm truncate">{c.case_name}</div>
                    <div className="text-xs flex gap-3 mt-1" style={{ color: 'var(--muted)' }}>
                      <span className="flex items-center gap-1">
                        <Clock size={10} /> {c.created_at ? formatDate(c.created_at) : 'Unknown'}
                      </span>
                      {c.pages && <span>{c.pages} pages</span>}
                      {(c as any).domain && (
                        <span className="px-1.5 py-0.5 rounded"
                          style={{ background: 'var(--surface2)', border: '1px solid var(--border)' }}>
                          {domainOptions.find(d => d.id === (c as any).domain)?.name ?? (c as any).domain}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    <span className={`w-2 h-2 rounded-full ${cfg.dot}`} />
                    <span className={`text-xs font-semibold ${cfg.color}`}>{cfg.label}</span>
                  </div>
                  <span style={{ color: 'var(--muted)' }}
                    className="opacity-0 group-hover:opacity-100 transition-opacity">→</span>
                  <button
                    onClick={e => deleteCase(e, c.case_id)}
                    disabled={deletingId === c.case_id}
                    className="w-7 h-7 flex items-center justify-center rounded opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0"
                    style={{ color: '#ef4444', border: '1px solid #7f1d1d' }}
                    title="Delete case">
                    {deletingId === c.case_id
                      ? <Loader2 size={12} className="animate-spin" />
                      : <Trash2 size={12} />}
                  </button>
                </button>
              )
            })}
          </div>
        )}
      </main>
    </div>
  )
}