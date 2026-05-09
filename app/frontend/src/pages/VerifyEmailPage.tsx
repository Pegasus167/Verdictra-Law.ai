import { useEffect, useState } from 'react'
import { useSearchParams, Link } from 'react-router-dom'
import { Loader2, CheckCircle, AlertCircle } from 'lucide-react'

export default function VerifyEmailPage() {
  const [searchParams] = useSearchParams()
  const token = searchParams.get('token') || ''
  const [status, setStatus] = useState<'verifying' | 'success' | 'error'>('verifying')
  const [errorMsg, setErrorMsg] = useState('')

  useEffect(() => {
    if (!token) { setStatus('error'); setErrorMsg('Missing verification token. Please use the link from your email.'); return }
    fetch(`/api/auth/verify?token=${encodeURIComponent(token)}`)
      .then(r => r.json().then(d => ({ ok: r.ok, data: d })))
      .then(({ ok, data }) => {
        if (ok) setStatus('success')
        else { setStatus('error'); setErrorMsg(data.detail || 'Verification failed.') }
      })
      .catch(() => { setStatus('error'); setErrorMsg('Network error. Please try again.') })
  }, [token])

  return (
    <div className="min-h-screen flex items-center justify-center p-5" style={{ background: 'var(--bg)' }}>
      <div className="w-full max-w-sm text-center">
        <span className="font-bold text-lg tracking-wider block mb-8" style={{ color: 'var(--accent)', fontFamily: 'Noto Serif, serif' }}>Verdictra</span>

        {status === 'verifying' && (
          <>
            <Loader2 size={32} className="animate-spin mx-auto mb-4" style={{ color: 'var(--accent2)' }} />
            <p className="text-sm" style={{ color: 'var(--muted)' }}>Verifying your email…</p>
          </>
        )}

        {status === 'success' && (
          <>
            <CheckCircle size={40} className="mx-auto mb-4" style={{ color: '#34d399' }} />
            <h1 className="text-xl font-bold mb-3">Email verified</h1>
            <p className="text-sm mb-8" style={{ color: 'var(--muted)' }}>Your account is active. You can now sign in.</p>
            <Link to="/login">
              <button className="w-full py-2.5 rounded-lg text-sm font-bold"
                style={{ background: 'var(--accent)', color: '#fff' }}>
                Sign in →
              </button>
            </Link>
          </>
        )}

        {status === 'error' && (
          <>
            <AlertCircle size={40} className="mx-auto mb-4" style={{ color: '#f87171' }} />
            <h1 className="text-xl font-bold mb-3">Verification failed</h1>
            <p className="text-sm mb-8" style={{ color: 'var(--muted)' }}>{errorMsg}</p>
            <Link to="/signup">
              <button className="w-full py-2.5 rounded-lg text-sm font-bold mb-3"
                style={{ background: 'var(--accent)', color: '#fff' }}>
                Sign up again
              </button>
            </Link>
            <p className="text-xs" style={{ color: 'var(--muted)' }}>
              Or <a href="mailto:support@verdictra.ai" style={{ color: 'var(--accent2)' }}>contact support</a>
            </p>
          </>
        )}
      </div>
    </div>
  )
}