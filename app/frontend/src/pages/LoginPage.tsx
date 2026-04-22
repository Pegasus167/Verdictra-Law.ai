import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Scale, Loader2, AlertCircle } from 'lucide-react'

const BASE = ''

export default function LoginPage() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState('')
  const navigate                = useNavigate()

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault()
    if (!username.trim() || !password.trim()) return
    setLoading(true)
    setError('')

    try {
      const res = await fetch(`${BASE}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: username.trim(), password }),
      })

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || 'Login failed')
      }

      const data = await res.json()
      localStorage.setItem('token', data.access_token)
      localStorage.setItem('user_name', data.name)
      localStorage.setItem('user_role', data.role)
      navigate('/')
    } catch (err: any) {
      setError(err.message || 'Login failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-8"
      style={{ background: 'var(--bg)' }}>
      <div className="w-full max-w-sm">

        <div className="flex items-center justify-center gap-3 mb-10">
          <Scale size={28} className="text-indigo-400" />
          <span className="text-2xl font-bold text-indigo-300 tracking-wider">Verdictra</span>
        </div>

        <div className="rounded-2xl p-8"
          style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
          <h1 className="text-lg font-bold mb-1">Sign in</h1>
          <p className="text-xs mb-6" style={{ color: 'var(--muted)' }}>
            Legal Intelligence Platform
          </p>

          <form onSubmit={handleLogin} className="flex flex-col gap-4">
            <div className="flex flex-col gap-1.5">
              <label className="text-xs" style={{ color: 'var(--muted)' }}>Username</label>
              <input
                type="text"
                value={username}
                onChange={e => setUsername(e.target.value)}
                placeholder="Enter username"
                required
                autoFocus
                className="rounded-lg px-3 py-2.5 text-sm outline-none"
                style={{ background: 'var(--bg)', border: '1px solid var(--border2)', color: 'var(--text)' }}
              />
            </div>

            <div className="flex flex-col gap-1.5">
              <label className="text-xs" style={{ color: 'var(--muted)' }}>Password</label>
              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder="Enter password"
                required
                className="rounded-lg px-3 py-2.5 text-sm outline-none"
                style={{ background: 'var(--bg)', border: '1px solid var(--border2)', color: 'var(--text)' }}
              />
            </div>

            {error && (
              <div className="flex items-center gap-2 text-xs text-red-400 px-3 py-2 rounded"
                style={{ background: '#1c0a0a', border: '1px solid #7f1d1d' }}>
                <AlertCircle size={12} /> {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading || !username.trim() || !password.trim()}
              className="flex items-center justify-center gap-2 rounded-lg py-2.5 text-sm font-bold mt-2"
              style={{
                background: loading || !username.trim() || !password.trim() ? 'var(--border2)' : 'var(--accent)',
                color: 'white',
              }}>
              {loading && <Loader2 size={14} className="animate-spin" />}
              {loading ? 'Signing in...' : 'Sign in →'}
            </button>
          </form>
        </div>

        <p className="text-center text-xs mt-6" style={{ color: 'var(--muted)' }}>
          Verdictra — Legal Intelligence Platform
        </p>
      </div>
    </div>
  )
}