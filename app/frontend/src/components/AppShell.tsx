import { useState, useEffect } from 'react'
import { useNavigate, useLocation, Outlet, Link, Navigate } from 'react-router-dom'
import { Scale, BookOpen, ChevronLeft, ChevronRight, Plus, User, LogOut } from 'lucide-react'
import { api } from '../lib/api'
import type { Case } from '../types'

type Mode = 'case' | 'law'

interface LawSession {
  id: string
  name: string
  updated_at: string
}

export default function AppShell() {
  const navigate   = useNavigate()
  const location   = useLocation()

  // Auth guard
  const isDev  = window.location.hostname === 'localhost'
  const token  = localStorage.getItem('token')
  if (!isDev && !token) return <Navigate to="/login" replace />

  const [collapsed, setCollapsed]   = useState(false)
  const [mode, setMode]             = useState<Mode>(
    location.pathname.startsWith('/law-research') ? 'law' : 'case'
  )
  const [cases, setCases]           = useState<Case[]>([])
  const [sessions, setSessions]     = useState<LawSession[]>([])
  const [loadingList, setLoadingList] = useState(false)

  const userName = localStorage.getItem('user_name') || ''

  // Load case list or session list when mode changes
  useEffect(() => {
    if (mode === 'case') {
      setLoadingList(true)
      api.getCases()
        .then(setCases)
        .catch(() => {})
        .finally(() => setLoadingList(false))
    } else {
      setLoadingList(true)
      api.getLawSessions()
        .then(setSessions)
        .catch(() => {})
        .finally(() => setLoadingList(false))
    }
  }, [mode])

  // Sync mode with route
  useEffect(() => {
    if (location.pathname.startsWith('/law-research')) setMode('law')
    else setMode('case')
  }, [location.pathname])

  async function newLawSession() {
    try {
      const session = await api.createLawSession()
      setSessions(prev => [session, ...prev])
      navigate(`/law-research/${session.id}`)
    } catch {}
  }

  const W = collapsed ? 48 : 240

  return (
    <div className="h-screen flex overflow-hidden" style={{ background: 'var(--bg)' }}>

      {/* ── Left panel ─────────────────────────────────────────────────────── */}
      <div
        className="flex-shrink-0 flex flex-col overflow-hidden transition-all duration-200"
        style={{
          width: W,
          background: 'var(--surface)',
          borderRight: '1px solid var(--border)',
        }}>

        {/* Logo + collapse toggle */}
        <div className="flex items-center justify-between flex-shrink-0 px-3"
          style={{ height: 52, borderBottom: '1px solid var(--border)' }}>
          {!collapsed && (
            <Link to="/" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 8 }}>
              <img src="/logo.png" alt="Verdictra" style={{ height: 24, width: 'auto' }} />
              <span className="font-bold text-sm tracking-wider"
                style={{ color: 'var(--accent)', fontFamily: 'Noto Serif, serif' }}>
                Verdictra
              </span>
            </Link>
          )}
          <button
            onClick={() => setCollapsed(c => !c)}
            className="w-7 h-7 flex items-center justify-center rounded transition-colors ml-auto"
            style={{ color: 'var(--muted)', background: 'transparent', border: 'none', cursor: 'pointer' }}>
            {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
          </button>
        </div>

        {/* Mode selector */}
        <div className="flex-shrink-0 p-2" style={{ borderBottom: '1px solid var(--border)' }}>
          {collapsed ? (
            // Icon-only mode buttons
            <div className="flex flex-col gap-1">
              <button
                onClick={() => { setMode('case'); navigate('/') }}
                className="w-8 h-8 flex items-center justify-center rounded-lg mx-auto"
                style={{
                  background: mode === 'case' ? 'var(--accent)' : 'transparent',
                  color: mode === 'case' ? '#fff' : 'var(--muted)',
                  border: 'none', cursor: 'pointer',
                }}
                title="Case Research">
                <Scale size={14} />
              </button>
              <button
                onClick={() => { setMode('law'); navigate('/law-research') }}
                className="w-8 h-8 flex items-center justify-center rounded-lg mx-auto"
                style={{
                  background: mode === 'law' ? 'var(--accent)' : 'transparent',
                  color: mode === 'law' ? '#fff' : 'var(--muted)',
                  border: 'none', cursor: 'pointer',
                }}
                title="Law Research">
                <BookOpen size={14} />
              </button>
            </div>
          ) : (
            <div className="flex flex-col gap-1">
              <button
                onClick={() => { setMode('case'); navigate('/') }}
                className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-semibold transition-colors text-left"
                style={{
                  background: mode === 'case' ? 'var(--accent)' : 'transparent',
                  color: mode === 'case' ? '#fff' : 'var(--muted2)',
                  border: 'none', cursor: 'pointer',
                }}>
                <Scale size={13} />
                Case Research
              </button>
              <button
                onClick={() => { setMode('law'); navigate('/law-research') }}
                className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs font-semibold transition-colors text-left"
                style={{
                  background: mode === 'law' ? 'var(--accent)' : 'transparent',
                  color: mode === 'law' ? '#fff' : 'var(--muted2)',
                  border: 'none', cursor: 'pointer',
                }}>
                <BookOpen size={13} />
                Law Research
              </button>
            </div>
          )}
        </div>

        {/* Context list */}
        <div className="flex-1 overflow-y-auto py-1">
          {!collapsed && (
            <>
              {/* List header + new button */}
              <div className="flex items-center justify-between px-3 py-2">
                <span className="text-xs font-semibold tracking-widest uppercase"
                  style={{ color: 'var(--muted)' }}>
                  {mode === 'case' ? 'Cases' : 'Sessions'}
                </span>
                {mode === 'case' ? (
                  <Link to="/">
                    <button
                      className="w-5 h-5 flex items-center justify-center rounded"
                      style={{ background: 'var(--surface2)', border: '1px solid var(--border2)', color: 'var(--muted2)', cursor: 'pointer' }}
                      title="New case">
                      <Plus size={11} />
                    </button>
                  </Link>
                ) : (
                  <button
                    onClick={newLawSession}
                    className="w-5 h-5 flex items-center justify-center rounded"
                    style={{ background: 'var(--surface2)', border: '1px solid var(--border2)', color: 'var(--muted2)', cursor: 'pointer' }}
                    title="New session">
                    <Plus size={11} />
                  </button>
                )}
              </div>

              {/* List items */}
              {loadingList ? (
                <div className="px-3 py-2 text-xs" style={{ color: 'var(--muted)' }}>Loading…</div>
              ) : mode === 'case' ? (
                cases.length === 0 ? (
                  <div className="px-3 py-2 text-xs" style={{ color: 'var(--muted)' }}>No cases yet</div>
                ) : (
                  cases.map(c => (
                    <button
                      key={c.case_id}
                      onClick={() => {
                        if (c.status === 'ready') navigate(`/query/${c.case_id}`)
                        else if (c.status === 'review') navigate(`/review/${c.case_id}`)
                        else navigate(`/processing/${c.case_id}`)
                      }}
                      className="w-full text-left px-3 py-2 text-xs transition-colors"
                      style={{
                        background: 'transparent',
                        border: 'none',
                        cursor: 'pointer',
                        color: 'var(--muted2)',
                        borderLeft: `2px solid transparent`,
                      }}
                      onMouseEnter={e => {
                        e.currentTarget.style.background = 'var(--surface2)'
                        e.currentTarget.style.borderLeftColor = 'var(--border2)'
                      }}
                      onMouseLeave={e => {
                        e.currentTarget.style.background = 'transparent'
                        e.currentTarget.style.borderLeftColor = 'transparent'
                      }}>
                      <div className="truncate font-medium">{c.case_name}</div>
                      <div className="text-xs mt-0.5" style={{ color: 'var(--muted)' }}>
                        {c.status === 'ready' ? '● Ready' : c.status === 'review' ? '● Review' : '○ Processing'}
                      </div>
                    </button>
                  ))
                )
              ) : (
                sessions.length === 0 ? (
                  <div className="px-3 py-2 text-xs" style={{ color: 'var(--muted)' }}>No sessions yet</div>
                ) : (
                  sessions.map(s => {
                    const active = location.pathname === `/law-research/${s.id}`
                    return (
                      <button
                        key={s.id}
                        onClick={() => navigate(`/law-research/${s.id}`)}
                        className="w-full text-left px-3 py-2 text-xs transition-colors"
                        style={{
                          background: active ? 'var(--surface2)' : 'transparent',
                          border: 'none',
                          cursor: 'pointer',
                          color: active ? 'var(--text)' : 'var(--muted2)',
                          borderLeft: `2px solid ${active ? 'var(--accent)' : 'transparent'}`,
                        }}
                        onMouseEnter={e => {
                          if (!active) e.currentTarget.style.background = 'var(--surface2)'
                        }}
                        onMouseLeave={e => {
                          if (!active) e.currentTarget.style.background = 'transparent'
                        }}>
                        <div className="truncate">{s.name}</div>
                      </button>
                    )
                  })
                )
              )}
            </>
          )}
        </div>

        {/* Bottom — user + profile + sign out */}
        <div className="flex-shrink-0 p-2" style={{ borderTop: '1px solid var(--border)' }}>
          {collapsed ? (
            <div className="flex flex-col gap-1">
              <button
                onClick={() => navigate('/profile')}
                className="w-8 h-8 flex items-center justify-center rounded-lg mx-auto"
                style={{ background: 'transparent', color: 'var(--muted)', border: 'none', cursor: 'pointer' }}
                title="Profile">
                <User size={13} />
              </button>
              <button
                onClick={() => { localStorage.clear(); navigate('/login') }}
                className="w-8 h-8 flex items-center justify-center rounded-lg mx-auto"
                style={{ background: 'transparent', color: 'var(--muted)', border: 'none', cursor: 'pointer' }}
                title="Sign out">
                <LogOut size={13} />
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-2 px-1">
              <div className="w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0 text-xs font-bold"
                style={{ background: 'var(--surface2)', border: '1px solid var(--border2)', color: 'var(--accent)' }}>
                {userName.charAt(0).toUpperCase()}
              </div>
              <span className="text-xs truncate flex-1" style={{ color: 'var(--muted2)' }}>{userName}</span>
              <button
                onClick={() => navigate('/profile')}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--muted)', padding: 2 }}
                title="Profile">
                <User size={12} />
              </button>
              <button
                onClick={() => { localStorage.clear(); navigate('/login') }}
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--muted)', padding: 2 }}
                title="Sign out">
                <LogOut size={12} />
              </button>
            </div>
          )}
        </div>
      </div>

      {/* ── Page content ───────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-hidden">
        <Outlet />
      </div>

    </div>
  )
}