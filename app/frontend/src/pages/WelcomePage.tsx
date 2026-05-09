/**
 * WelcomePage.tsx
 * ---------------
 * Full-screen modal overlay shown on first login.
 * Not skippable until the user reaches the last step.
 *
 * USAGE in CasesPage.tsx:
 *
 *   import WelcomePage from './WelcomePage'
 *
 *   // Inside CasesPage component, add state:
 *   const [showWelcome, setShowWelcome] = useState(
 *     localStorage.getItem('first_login') === 'true'
 *   )
 *
 *   // At the top of the return JSX:
 *   {showWelcome && (
 *     <WelcomePage
 *       userName={localStorage.getItem('user_name') || ''}
 *       onDismiss={() => {
 *         setShowWelcome(false)
 *         localStorage.removeItem('first_login')
 *       }}
 *     />
 *   )}
 */

import { useState } from 'react'

interface Props {
  userName: string
  onDismiss: () => void
}

const STEPS = [
  {
    n: '01',
    title: 'Upload your first case',
    body: 'Click New Case, give it a name, and upload your documents — PDF, DOCX, scanned orders, or email attachments. Multiple files per case are supported.',
    icon: (
      <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
        <rect x="7" y="9" width="22" height="20" rx="2" stroke="var(--accent2)" strokeWidth="1.5"/>
        <path d="M13 9V7a5 5 0 0110 0v2" stroke="var(--accent2)" strokeWidth="1.5" strokeLinecap="round"/>
        <path d="M18 17v7M15 20l3-3 3 3" stroke="var(--text)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    ),
  },
  {
    n: '02',
    title: 'Review extracted entities',
    body: 'After upload, every party, date, amount, and court order is shown with a page citation. Confirm or correct them before your first query — this is your quality gate.',
    icon: (
      <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
        <circle cx="18" cy="18" r="12" stroke="var(--accent2)" strokeWidth="1.5"/>
        <path d="M12 18l4 4 8-8" stroke="var(--text)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    ),
  },
  {
    n: '03',
    title: 'Ask your first question',
    body: '"What did the court direct on 15 January?" "Is there a payment default clause?" Every answer is cited to the exact page number in your document.',
    icon: (
      <svg width="36" height="36" viewBox="0 0 36 36" fill="none">
        <rect x="5" y="8" width="22" height="14" rx="2" stroke="var(--accent2)" strokeWidth="1.5"/>
        <path d="M10 14h12M10 18h8" stroke="var(--text)" strokeWidth="1.5" strokeLinecap="round"/>
        <path d="M5 24l4-4h22" stroke="var(--accent2)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    ),
  },
]

export default function WelcomePage({ userName, onDismiss }: Props) {
  const [step, setStep] = useState(0)
  const isLast = step === STEPS.length - 1
  const cur = STEPS[step]

  async function handleDone() {
    try {
      const token = localStorage.getItem('token')
      await fetch('/api/auth/welcome-dismissed', {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
      })
    } catch { /* non-critical */ }
    onDismiss()
  }

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 999,
      background: 'rgba(0,0,0,0.7)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: 20,
    }}>
      <div style={{
        background: 'var(--surface)', border: '1px solid var(--border2)',
        borderRadius: 12, maxWidth: 480, width: '100%',
        padding: '32px 36px 28px',
        boxShadow: '0 24px 64px rgba(0,0,0,0.5)',
      }}>

        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <span style={{ fontFamily: 'Noto Serif, serif', fontSize: 15, fontWeight: 700, color: 'var(--accent)' }}>
            Verdictra
          </span>
          <span style={{ fontSize: 11, color: 'var(--muted)', fontWeight: 600, letterSpacing: '0.05em' }}>
            {step + 1} of {STEPS.length}
          </span>
        </div>

        {/* Progress bar */}
        <div style={{ display: 'flex', gap: 4, marginBottom: 28 }}>
          {STEPS.map((_, i) => (
            <div key={i} style={{
              flex: 1, height: 3, borderRadius: 2,
              background: i <= step ? 'var(--accent)' : 'var(--border2)',
              transition: 'background 300ms',
            }} />
          ))}
        </div>

        {/* Greeting on first step */}
        {step === 0 && (
          <p style={{ fontSize: 12, color: 'var(--accent2)', marginBottom: 16, fontStyle: 'italic' }}>
            Welcome, {userName.split(' ')[0]}.
          </p>
        )}

        {/* Step content */}
        <div style={{ marginBottom: 28 }}>
          <div style={{ marginBottom: 16 }}>{cur.icon}</div>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--muted)', marginBottom: 8 }}>
            {cur.n}
          </div>
          <h2 style={{ fontFamily: 'Noto Serif, serif', fontSize: 20, fontWeight: 600, color: 'var(--text)', marginBottom: 10, lineHeight: 1.3 }}>
            {cur.title}
          </h2>
          <p style={{ fontSize: 13, color: 'var(--muted2)', lineHeight: 1.65 }}>
            {cur.body}
          </p>
        </div>

        {/* Navigation */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
          {step > 0 ? (
            <button onClick={() => setStep(s => s - 1)}
              style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: 12, color: 'var(--muted)' }}>
              ← Back
            </button>
          ) : <div />}

          <button
            onClick={isLast ? handleDone : () => setStep(s => s + 1)}
            className="rounded-lg px-5 py-2 text-sm font-bold"
            style={{ background: 'var(--accent)', color: '#fff', border: 'none', cursor: 'pointer' }}
          >
            {isLast ? 'Take me to Verdictra →' : 'Next →'}
          </button>
        </div>

        <p style={{ fontSize: 11, color: 'var(--muted)', textAlign: 'center' }}>
          Questions?{' '}
          <a href="mailto:support@verdictra.ai" style={{ color: 'var(--accent2)', textDecoration: 'none' }}>
            support@verdictra.ai
          </a>
        </p>

      </div>
    </div>
  )
}