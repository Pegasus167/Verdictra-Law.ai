import { useEffect, useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { Loader2, ChevronLeft } from 'lucide-react'
import { api } from '../lib/api'

interface Profile {
  username: string; name: string; email?: string; role: string
  plan: string; firm_name?: string; account_type?: string; created_at?: string
}

const PLAN_LABEL: Record<string, string> = {
  free: 'Free', starter: 'Starter', professional: 'Professional',
  firm_small: 'Professional', firm_mid: 'Firm', firm_large: 'Firm — Enterprise',
}
const PLAN_LIMIT: Record<string, number | null> = {
  free: 2, starter: 10, professional: 50, firm_small: 50, firm_mid: 100, firm_large: null,
}
const PLAN_UPGRADE: Record<string, string | null> = {
  free: 'Upgrade to Starter', starter: 'Upgrade to Professional',
  professional: 'Upgrade to Firm', firm_small: 'Upgrade to Firm',
  firm_mid: null, firm_large: null,
}

export default function ProfilePage() {
  const navigate = useNavigate()
  const [profile, setProfile] = useState<Profile | null>(null)
  const [caseCount, setCaseCount] = useState(0)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const token = localStorage.getItem('token')
    if (!token) { navigate('/login'); return }

    Promise.all([
      fetch('/api/auth/me', { headers: { Authorization: `Bearer ${token}` } }).then(r => {
        if (r.status === 401) { navigate('/login'); return null }
        return r.json()
      }),
      api.getCases().catch(() => [] as any[]),
    ]).then(([me, cases]) => {
      if (me) setProfile(me)
      setCaseCount(Array.isArray(cases) ? cases.length : 0)
    }).finally(() => setLoading(false))
  }, [navigate])

  if (loading) return (
    <div className="min-h-screen flex items-center justify-center" style={{ background: 'var(--bg)' }}>
      <Loader2 size={24} className="animate-spin" style={{ color: 'var(--accent2)' }} />
    </div>
  )

  if (!profile) return null

  const limit  = profile.role === 'admin' ? null : (PLAN_LIMIT[profile.plan] ?? 2)
  const pct    = limit ? Math.min(100, (caseCount / limit) * 100) : 0
  const joined = profile.created_at
    ? new Date(profile.created_at).toLocaleDateString('en-IN', { year: 'numeric', month: 'long' })
    : '—'

  return (
    <div className="min-h-screen" style={{ background: 'var(--bg)' }}>

      {/* Header */}
      <header style={{ background: 'var(--surface)', borderBottom: '1px solid var(--border)' }}
        className="h-14 flex items-center px-10 gap-3">
        <button onClick={() => navigate(-1)} className="flex items-center gap-1 text-xs"
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--muted)' }}>
          <ChevronLeft size={14} /> Back
        </button>
        <span className="font-bold tracking-wider text-sm" style={{ color: 'var(--accent)', fontFamily: 'Noto Serif, serif' }}>Verdictra</span>
      </header>

      <main className="mx-auto px-10 py-10" style={{ maxWidth: 680 }}>

        {/* User header */}
        <div className="flex items-center gap-4 mb-8">
          <div className="w-12 h-12 rounded-full flex items-center justify-center font-bold text-lg flex-shrink-0"
            style={{ background: 'var(--surface)', border: '1px solid var(--border2)', color: 'var(--accent)', fontFamily: 'Noto Serif, serif' }}>
            {profile.name.charAt(0).toUpperCase()}
          </div>
          <div>
            <h1 className="text-lg font-bold">{profile.name}</h1>
            <p className="text-xs mt-0.5" style={{ color: 'var(--muted)' }}>@{profile.username}</p>
          </div>
        </div>

        {/* Cards */}
        <div className="grid grid-cols-2 gap-4">

          {/* Account */}
          <div className="rounded-xl p-5" style={{ background: 'var(--surface)', border: '1px solid var(--border2)' }}>
            <div className="text-xs font-semibold tracking-widest uppercase mb-4" style={{ color: 'var(--muted2)' }}>Account</div>
            {[
              ['Email',        profile.email || '—'],
              profile.firm_name ? ['Firm', profile.firm_name] : null,
              ['Account type', profile.account_type || 'individual'],
              ['Role',         profile.role],
              ['Member since', joined],
            ].filter((row): row is [string, string] => row !== null).map(([k, v]) => (
              <div key={k as string} className="flex justify-between items-center mb-3 last:mb-0">
                <span className="text-xs" style={{ color: 'var(--muted)' }}>{k}</span>
                <span className="text-xs font-medium capitalize" style={{ color: 'var(--text)' }}>{v}</span>
              </div>
            ))}
          </div>

          {/* Plan */}
          <div className="rounded-xl p-5" style={{ background: 'var(--surface)', border: '1px solid var(--border2)' }}>
            <div className="text-xs font-semibold tracking-widest uppercase mb-4" style={{ color: 'var(--muted2)' }}>Plan & Usage</div>

            <div className="flex items-center gap-2 mb-4">
              <div className="w-2 h-2 rounded-full" style={{ background: 'var(--accent)' }} />
              <span className="text-sm font-semibold" style={{ color: 'var(--text)' }}>
                {PLAN_LABEL[profile.plan] || profile.plan}
              </span>
            </div>

            <div className="flex justify-between items-center mb-2">
              <span className="text-xs" style={{ color: 'var(--muted)' }}>Cases used</span>
              <span className="text-xs font-medium" style={{ color: 'var(--text)' }}>
                {caseCount} / {limit === null ? '∞' : limit}
              </span>
            </div>

            {limit !== null && (
              <div className="h-1 rounded-full mb-3" style={{ background: 'var(--border2)' }}>
                <div className="h-full rounded-full transition-all"
                  style={{ width: `${pct}%`, background: pct >= 90 ? '#f87171' : 'var(--accent)' }} />
              </div>
            )}

            {pct >= 80 && limit !== null && (
              <p className="text-xs mb-3" style={{ color: '#fbbf24' }}>
                {pct >= 100 ? 'Case limit reached.' : `Approaching limit (${caseCount}/${limit}).`}
              </p>
            )}

            {PLAN_UPGRADE[profile.plan] && (
              <a href="mailto:support@verdictra.ai?subject=Plan upgrade"
                className="text-xs font-semibold" style={{ color: 'var(--accent2)', textDecoration: 'none' }}>
                {PLAN_UPGRADE[profile.plan]} →
              </a>
            )}

            <div className="flex justify-between items-center mt-4">
              <span className="text-xs" style={{ color: 'var(--muted)' }}>Billing</span>
              <span className="text-xs" style={{ color: 'var(--muted2)' }}>UPI / bank transfer</span>
            </div>
          </div>
        </div>

        {/* Footer links */}
        <div className="flex items-center gap-4 mt-6">
          {[
            { label: 'Change password', href: '/forgot-password' },
            { label: 'Contact support', href: 'mailto:support@verdictra.ai' },
            { label: 'Privacy policy',  href: '/privacy.html', external: true },
          ].map((l, i) => (
            <span key={i} className="flex items-center gap-4">
              {i > 0 && <span style={{ color: 'var(--border2)' }}>·</span>}
              <a href={l.href} target={l.external ? '_blank' : undefined} rel="noopener noreferrer"
                className="text-xs" style={{ color: 'var(--muted)', textDecoration: 'none' }}>
                {l.label}
              </a>
            </span>
          ))}
        </div>

      </main>
    </div>
  )
}