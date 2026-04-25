import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Upload, Scale, Clock, AlertCircle, Loader2, Trash2 } from 'lucide-react'
import { api } from '../lib/api'
import { formatDate } from '../lib/utils'
import type { Case } from '../types'

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
  const [cases, setCases]       = useState<Case[]>([])
  const [domains, setDomains]   = useState<Domain[]>([])
  const [loading, setLoading]   = useState(true)
  const [uploading, setUploading] = useState(false)
  const [caseName, setCaseName] = useState('')
  const [file, setFile]         = useState<File | null>(null)
  const [domain, setDomain]     = useState('constitutional')
  const [error, setError]       = useState('')
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const fileRef                 = useRef<HTMLInputElement>(null)
  const navigate                = useNavigate()

  useEffect(() => {
    api.getCases()
      .then(setCases)
      .catch(() => setError('Failed to load cases'))
      .finally(() => setLoading(false))

    // Load domains for dropdown
    api.getDomains()
      .then(d => { if (d.length > 0) setDomains(d) })
      .catch(() => {}) // non-fatal — fallback options always shown
  }, [])

  async function handleUpload(e: React.FormEvent) {
    e.preventDefault()
    if (!caseName.trim() || !file) return
    setUploading(true)
    setError('')
    try {
      const { case_id } = await api.uploadCase(caseName.trim(), file, domain)
      navigate(`/processing/${case_id}`)
    } catch {
      setError('Upload failed. Please try again.')
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
      const res = await fetch(`/api/cases/${caseId}`, { method: 'DELETE' })
      if (!res.ok) throw new Error('Delete failed')
      setCases(prev => prev.filter(c => c.case_id !== caseId))
    } catch {
      setError('Failed to delete case. Please try again.')
    } finally {
      setDeletingId(null)
    }
  }

  // Fallback domain options if /domains endpoint not yet available
  const domainOptions: Domain[] = domains.length > 0 ? domains : [
    { id: 'constitutional', name: 'Constitutional & Writ',    description: '' },
    { id: 'property',       name: 'Property & Real Estate',   description: '' },
    { id: 'banking_finance',name: 'Banking & Finance',        description: '' },
    { id: 'corporate',      name: 'Corporate & Company Law',  description: '' },
    { id: 'criminal',       name: 'Criminal Law',             description: '' },
    { id: 'tax',            name: 'Tax Law',                  description: '' },
    { id: 'labour',         name: 'Labour & Employment',      description: '' },
    { id: 'ip_patent',      name: 'IP & Patents',             description: '' },
  ]

  return (
    <div className="min-h-screen" style={{ background: 'var(--bg)' }}>

      {/* Header */}
      <header style={{ background: 'var(--surface)', borderBottom: '1px solid var(--border)' }}
        className="h-14 flex items-center justify-between px-10">
        <div className="flex items-center gap-2">
          <Scale size={16} className="text-indigo-400" />
          <span className="font-bold text-indigo-300 tracking-wider text-sm">LAW.ai</span>
          <span style={{ color: 'var(--muted)' }} className="text-sm">/ Case Intelligence</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-xs" style={{ color: 'var(--muted)' }}>
            {localStorage.getItem('user_name') || ''}
          </span>
          <button
            onClick={() => {
              localStorage.clear()
              navigate('/login')
            }}
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

            {/* Row 1 — case name + pdf */}
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
                  style={{
                    background: 'var(--bg)',
                    border: '1px solid var(--border2)',
                    color: 'var(--text)',
                  }}
                />
              </div>

              <div className="flex flex-col gap-1.5 flex-1 min-w-48">
                <label className="text-xs" style={{ color: 'var(--muted)' }}>PDF file</label>
                <button
                  type="button"
                  onClick={() => fileRef.current?.click()}
                  className="rounded-lg px-3 py-2 text-sm text-left flex items-center gap-2"
                  style={{
                    background: 'var(--bg)',
                    border: `1px solid ${file ? '#10b981' : 'var(--border2)'}`,
                    color: file ? 'var(--text)' : 'var(--muted2)',
                  }}
                >
                  <Upload size={14} />
                  {file ? file.name : 'Choose PDF...'}
                </button>
                <input
                  ref={fileRef}
                  type="file"
                  accept=".pdf"
                  className="hidden"
                  onChange={e => setFile(e.target.files?.[0] ?? null)}
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
                  style={{
                    background: 'var(--bg)',
                    border: '1px solid var(--border2)',
                    color: 'var(--text)',
                    cursor: 'pointer',
                  }}
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
                disabled={uploading || !caseName || !file}
                className="px-6 py-2 rounded-lg text-sm font-bold flex items-center gap-2"
                style={{
                  background: uploading || !caseName || !file ? 'var(--border2)' : 'var(--accent)',
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
            <p className="mt-3 text-xs text-red-400 flex items-center gap-1">
              <AlertCircle size={12} /> {error}
            </p>
          )}
        </div>

        {/* Case list */}
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
            <Loader2 size={24} className="animate-spin text-indigo-400" />
          </div>
        ) : cases.length === 0 ? (
          <div className="text-center py-16 flex flex-col items-center gap-3"
            style={{ color: 'var(--muted)' }}>
            <Scale size={32} className="opacity-20" />
            <p className="text-sm">No cases yet. Upload a PDF to get started.</p>
          </div>
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