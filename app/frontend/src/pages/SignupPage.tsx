import { useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { AlertCircle, Eye, EyeOff, User, Building2 } from 'lucide-react'

type AccountType = 'individual' | 'firm'
type Step = 'type' | 'form' | 'done'

export default function SignupPage() {
  const [searchParams] = useSearchParams()
  const [accountType, setAccountType] = useState<AccountType>('individual')
  const [step, setStep]         = useState<Step>('type')
  const [name, setName]         = useState('')
  const [username, setUsername] = useState('')
  const [email, setEmail]       = useState('')
  const [password, setPassword] = useState('')
  const [firmName, setFirmName] = useState('')
  const [tosAgreed, setTos]     = useState(false)
  const [showPw, setShowPw]     = useState(false)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState('')

  async function handleSubmit() {
    setError('')
    if (!tosAgreed)          { setError('Please agree to the Terms of Service.'); return }
    if (password.length < 8) { setError('Password must be at least 8 characters.'); return }
    if (accountType === 'firm' && !firmName.trim()) { setError('Firm name is required.'); return }

    setLoading(true)
    try {
      const resp = await fetch('/api/auth/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username:     username.trim().toLowerCase(),
          email:        email.trim().toLowerCase(),
          name:         name.trim(),
          password,
          firm_name:    accountType === 'firm' ? firmName.trim() : undefined,
          account_type: accountType,
        }),
      })
      const data = await resp.json()
      if (!resp.ok) { setError(data.detail || 'Signup failed. Please try again.'); return }
      setStep('done')
    } catch {
      setError('Network error. Please check your connection.')
    } finally {
      setLoading(false)
    }
  }

  // ── Step 1: Choose account type ─────────────────────────────────────────
  if (step === 'type') {
    return (
      <div className="min-h-screen flex items-center justify-center p-5" style={{ background: 'var(--bg)' }}>
        <div className="w-full max-w-md">

          <div className="mb-8 text-center">
            <span className="font-bold text-lg tracking-wider" style={{ color: 'var(--accent)', fontFamily: 'Noto Serif, serif' }}>Verdictra</span>
            <h1 className="text-xl font-bold mt-4 mb-1">Create your account</h1>
            <p className="text-sm" style={{ color: 'var(--muted)' }}>How will you be using Verdictra?</p>
          </div>

          <div className="grid grid-cols-2 gap-3 mb-6">
            {([
              { type: 'individual' as AccountType, icon: <User size={20} />, title: 'Individual lawyer', desc: 'Solo advocate or chamber lawyer' },
              { type: 'firm' as AccountType,       icon: <Building2 size={20} />, title: 'Law firm', desc: 'Multiple lawyers, shared access' },
            ]).map(opt => (
              <button
                key={opt.type}
                onClick={() => setAccountType(opt.type)}
                className="rounded-xl p-5 text-left transition-all"
                style={{
                  background: accountType === opt.type ? 'var(--surface)' : 'var(--surface)',
                  border: `1.5px solid ${accountType === opt.type ? 'var(--accent)' : 'var(--border)'}`,
                  color: 'var(--text)',
                }}
              >
                <div className="mb-3" style={{ color: accountType === opt.type ? 'var(--accent)' : 'var(--muted)' }}>
                  {opt.icon}
                </div>
                <div className="font-semibold text-sm mb-1">{opt.title}</div>
                <div className="text-xs" style={{ color: 'var(--muted)' }}>{opt.desc}</div>
              </button>
            ))}
          </div>

          <button
            onClick={() => setStep('form')}
            className="w-full py-2.5 rounded-lg text-sm font-bold"
            style={{ background: 'var(--accent)', color: '#fff' }}
          >
            Continue →
          </button>

          <p className="text-center mt-5 text-xs" style={{ color: 'var(--muted)' }}>
            Already have an account? <Link to="/login" style={{ color: 'var(--accent2)' }}>Sign in</Link>
          </p>
        </div>
      </div>
    )
  }

  // ── Step 2: Form ─────────────────────────────────────────────────────────
  if (step === 'form') {
    return (
      <div className="min-h-screen flex items-center justify-center p-5" style={{ background: 'var(--bg)' }}>
        <div className="w-full max-w-md">

          <button onClick={() => setStep('type')} className="text-xs mb-6 flex items-center gap-1"
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--muted)' }}>
            ← Back
          </button>

          <div className="mb-7">
            <span className="font-bold text-lg tracking-wider" style={{ color: 'var(--accent)', fontFamily: 'Noto Serif, serif' }}>Verdictra</span>
            <h1 className="text-xl font-bold mt-4 mb-1">
              {accountType === 'firm' ? 'Set up your firm account' : 'Create your account'}
            </h1>
          </div>

          <div className="rounded-xl p-6 flex flex-col gap-4" style={{ background: 'var(--surface)', border: '1px solid var(--border2)' }}>

            <Field label={accountType === 'firm' ? 'Your name (account admin)' : 'Full name'}>
              <input value={name} onChange={e => { setName(e.target.value); setError('') }}
                placeholder="Rajveer Singh" style={inputStyle} />
            </Field>

            {accountType === 'firm' && (
              <Field label="Firm name">
                <input value={firmName} onChange={e => { setFirmName(e.target.value); setError('') }}
                  placeholder="Sharma & Associates LLP" style={inputStyle} />
              </Field>
            )}

            <Field label="Username" hint="Used to log in — lowercase, no spaces">
              <input value={username} onChange={e => { setUsername(e.target.value.toLowerCase()); setError('') }}
                placeholder="rajveer.singh" style={inputStyle} autoCapitalize="none" />
            </Field>

            <Field label="Email">
              <input type="email" value={email} onChange={e => { setEmail(e.target.value); setError('') }}
                placeholder="you@example.com" style={inputStyle} />
            </Field>

            <Field label="Password">
              <div className="relative">
                <input type={showPw ? 'text' : 'password'} value={password}
                  onChange={e => { setPassword(e.target.value); setError('') }}
                  placeholder="8+ characters" style={{ ...inputStyle, paddingRight: 40 }} />
                <button type="button" onClick={() => setShowPw(v => !v)}
                  style={{ position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--muted)' }}>
                  {showPw ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
              </div>
            </Field>

            <label className="flex items-start gap-2.5 cursor-pointer">
              <input type="checkbox" checked={tosAgreed} onChange={e => { setTos(e.target.checked); setError('') }}
                className="mt-0.5 flex-shrink-0" />
              <span className="text-xs" style={{ color: 'var(--muted)' }}>
                I agree to the{' '}
                <a href="/terms.html" target="_blank" style={{ color: 'var(--accent2)' }}>Terms of Service</a>
                {' '}and{' '}
                <a href="/privacy.html" target="_blank" style={{ color: 'var(--accent2)' }}>Privacy Policy</a>
              </span>
            </label>

            {error && (
              <p className="text-xs flex items-center gap-1.5" style={{ color: '#f87171' }}>
                <AlertCircle size={12} /> {error}
              </p>
            )}

            <button
              onClick={handleSubmit} disabled={loading}
              className="w-full py-2.5 rounded-lg text-sm font-bold mt-1"
              style={{ background: loading ? 'var(--border2)' : 'var(--accent)', color: '#fff', opacity: loading ? 0.7 : 1 }}
            >
              {loading ? 'Creating account…' : 'Create account'}
            </button>
          </div>

          <p className="text-center mt-5 text-xs" style={{ color: 'var(--muted)' }}>
            Already have an account? <Link to="/login" style={{ color: 'var(--accent2)' }}>Sign in</Link>
          </p>
        </div>
      </div>
    )
  }

  // ── Step 3: Email sent ────────────────────────────────────────────────────
  return (
    <div className="min-h-screen flex items-center justify-center p-5" style={{ background: 'var(--bg)' }}>
      <div className="w-full max-w-sm text-center">
        <div className="w-14 h-14 rounded-full flex items-center justify-center mx-auto mb-6"
          style={{ background: 'var(--surface)', border: '1px solid var(--border2)' }}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" stroke="var(--accent2)" strokeWidth="1.5"/>
            <path d="M22 6l-10 7L2 6" stroke="var(--accent2)" strokeWidth="1.5"/>
          </svg>
        </div>
        <h1 className="text-xl font-bold mb-3">Check your email</h1>
        <p className="text-sm mb-2" style={{ color: 'var(--muted)' }}>
          We've sent a verification link to <strong style={{ color: 'var(--text)' }}>{email}</strong>.
        </p>
        <p className="text-sm mb-8" style={{ color: 'var(--muted)' }}>
          Click it to activate your account, then come back to sign in.
        </p>
        <Link to="/login">
          <button className="w-full py-2.5 rounded-lg text-sm font-bold"
            style={{ background: 'var(--surface)', border: '1px solid var(--border2)', color: 'var(--text)' }}>
            Back to sign in
          </button>
        </Link>
        <p className="mt-4 text-xs" style={{ color: 'var(--muted)' }}>
          Didn't get it? Check spam, or{' '}
          <a href="mailto:support@verdictra.ai" style={{ color: 'var(--accent2)' }}>contact support</a>.
        </p>
      </div>
    </div>
  )
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-xs font-semibold" style={{ color: 'var(--muted2)' }}>{label}</label>
      {children}
      {hint && <span className="text-xs" style={{ color: 'var(--muted)' }}>{hint}</span>}
    </div>
  )
}

const inputStyle: React.CSSProperties = {
  background: 'var(--bg)', border: '1px solid var(--border2)', borderRadius: 8,
  padding: '8px 12px', fontSize: 13, color: 'var(--text)', outline: 'none',
  width: '100%', fontFamily: 'inherit',
}