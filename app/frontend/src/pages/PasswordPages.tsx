import { useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { AlertCircle, Eye, EyeOff } from 'lucide-react'

// ── Shared ───────────────────────────────────────────────────────────────────

const inputStyle: React.CSSProperties = {
  background: 'var(--bg)', border: '1px solid var(--border2)', borderRadius: 8,
  padding: '8px 12px', fontSize: 13, color: 'var(--text)', outline: 'none',
  width: '100%', fontFamily: 'inherit',
}

function PageShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen flex items-center justify-center p-5" style={{ background: 'var(--bg)' }}>
      <div className="w-full max-w-sm">
        <span className="font-bold text-lg tracking-wider block mb-8"
          style={{ color: 'var(--accent)', fontFamily: 'Noto Serif, serif' }}>Verdictra</span>
        {children}
      </div>
    </div>
  )
}


// ── ForgotPasswordPage ────────────────────────────────────────────────────────

export function ForgotPasswordPage() {
  const [email, setEmail]   = useState('')
  const [loading, setLoad]  = useState(false)
  const [error, setError]   = useState('')
  const [sent, setSent]     = useState(false)

  async function submit() {
    if (!email.trim()) { setError('Please enter your email address.'); return }
    setLoad(true); setError('')
    try {
      const r = await fetch('/api/auth/forgot-password', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: email.trim().toLowerCase() }),
      })
      if (!r.ok) { const d = await r.json(); setError(d.detail || 'Something went wrong.'); return }
      setSent(true)
    } catch { setError('Network error.') }
    finally { setLoad(false) }
  }

  return (
    <PageShell>
      <h1 className="text-xl font-bold mb-1">Forgot password?</h1>
      <p className="text-sm mb-7" style={{ color: 'var(--muted)' }}>
        Enter your email and we'll send a reset link.
      </p>

      {sent ? (
        <>
          <div className="rounded-lg p-4 mb-6 text-sm" style={{ background: 'var(--surface)', border: '1px solid var(--border2)', color: 'var(--muted2)' }}>
            If an account with that email exists, a reset link has been sent. Check your inbox and spam folder.
          </div>
          <Link to="/login" style={{ color: 'var(--accent2)', fontSize: 13 }}>← Back to sign in</Link>
        </>
      ) : (
        <>
          <div className="flex flex-col gap-1.5 mb-5">
            <label className="text-xs font-semibold" style={{ color: 'var(--muted2)' }}>Email address</label>
            <input type="email" value={email}
              onChange={e => { setEmail(e.target.value); setError('') }}
              onKeyDown={e => e.key === 'Enter' && submit()}
              placeholder="you@example.com" style={inputStyle} autoFocus />
          </div>
          {error && <p className="text-xs flex items-center gap-1.5 mb-3" style={{ color: '#f87171' }}><AlertCircle size={12} />{error}</p>}
          <button onClick={submit} disabled={loading}
            className="w-full py-2.5 rounded-lg text-sm font-bold mb-5"
            style={{ background: loading ? 'var(--border2)' : 'var(--accent)', color: '#fff', opacity: loading ? 0.7 : 1 }}>
            {loading ? 'Sending…' : 'Send reset link'}
          </button>
          <Link to="/login" className="text-xs" style={{ color: 'var(--muted)' }}>← Back to sign in</Link>
        </>
      )}
    </PageShell>
  )
}


// ── ResetPasswordPage ─────────────────────────────────────────────────────────

export function ResetPasswordPage() {
  const [searchParams] = useSearchParams()
  const token = searchParams.get('token') || ''

  const [password, setPassword]   = useState('')
  const [confirm, setConfirm]     = useState('')
  const [showPw, setShowPw]       = useState(false)
  const [loading, setLoad]        = useState(false)
  const [error, setError]         = useState('')
  const [done, setDone]           = useState(false)

  async function submit() {
    if (password.length < 8) { setError('Password must be at least 8 characters.'); return }
    if (password !== confirm) { setError("Passwords don't match."); return }
    setLoad(true); setError('')
    try {
      const r = await fetch('/api/auth/reset-password', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token, new_password: password }),
      })
      const d = await r.json()
      if (!r.ok) { setError(d.detail || 'Reset failed. Link may have expired.'); return }
      setDone(true)
    } catch { setError('Network error.') }
    finally { setLoad(false) }
  }

  if (!token) return (
    <PageShell>
      <h1 className="text-xl font-bold mb-3">Invalid link</h1>
      <p className="text-sm mb-6" style={{ color: 'var(--muted)' }}>This reset link is missing or malformed.</p>
      <Link to="/forgot-password" style={{ color: 'var(--accent2)', fontSize: 13 }}>Request a new link →</Link>
    </PageShell>
  )

  return (
    <PageShell>
      <h1 className="text-xl font-bold mb-1">{done ? 'Password updated' : 'Choose a new password'}</h1>

      {done ? (
        <>
          <p className="text-sm mb-7" style={{ color: 'var(--muted)' }}>You can now sign in with your new password.</p>
          <Link to="/login">
            <button className="w-full py-2.5 rounded-lg text-sm font-bold"
              style={{ background: 'var(--accent)', color: '#fff' }}>Sign in →</button>
          </Link>
        </>
      ) : (
        <>
          <p className="text-sm mb-7" style={{ color: 'var(--muted)' }}>Must be at least 8 characters.</p>

          <div className="flex flex-col gap-4 mb-5">
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold" style={{ color: 'var(--muted2)' }}>New password</label>
              <div className="relative">
                <input type={showPw ? 'text' : 'password'} value={password}
                  onChange={e => { setPassword(e.target.value); setError('') }}
                  placeholder="8+ characters"
                  style={{ ...inputStyle, paddingRight: 40 }} autoFocus />
                <button type="button" onClick={() => setShowPw(v => !v)}
                  style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--muted)' }}>
                  {showPw ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
              </div>
            </div>
            <div className="flex flex-col gap-1.5">
              <label className="text-xs font-semibold" style={{ color: 'var(--muted2)' }}>Confirm password</label>
              <input type={showPw ? 'text' : 'password'} value={confirm}
                onChange={e => { setConfirm(e.target.value); setError('') }}
                onKeyDown={e => e.key === 'Enter' && submit()}
                placeholder="Re-enter password" style={inputStyle} />
            </div>
          </div>

          {error && <p className="text-xs flex items-center gap-1.5 mb-3" style={{ color: '#f87171' }}><AlertCircle size={12} />{error}</p>}
          <button onClick={submit} disabled={loading}
            className="w-full py-2.5 rounded-lg text-sm font-bold mb-5"
            style={{ background: loading ? 'var(--border2)' : 'var(--accent)', color: '#fff', opacity: loading ? 0.7 : 1 }}>
            {loading ? 'Updating…' : 'Set new password'}
          </button>
          <Link to="/forgot-password" className="text-xs" style={{ color: 'var(--muted)' }}>Request a different link</Link>
        </>
      )}
    </PageShell>
  )
}