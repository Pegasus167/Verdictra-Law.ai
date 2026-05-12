import { useState, useEffect, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { BookOpen, Send, Loader2, Pencil, Check, X } from 'lucide-react'
import { api } from '../lib/api'

const BASE = ''

const SUGGESTED = [
  'What does Section 138 of the Negotiable Instruments Act say?',
  'What are the grounds for filing a writ petition under Article 226?',
  'What is the procedure under Section 13(2) of SARFAESI Act?',
  'What is the doctrine of res judicata under CPC?',
  'What did the Supreme Court hold in Maneka Gandhi v Union of India?',
  'What is the limitation period for filing a civil suit under the Limitation Act?',
]

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1 py-1">
      {[0, 1, 2].map(i => (
        <div key={i} style={{
          width: 6, height: 6, borderRadius: '50%',
          background: 'var(--muted2)',
          animation: 'bounce 1.2s infinite',
          animationDelay: `${i * 0.2}s`,
        }} />
      ))}
      <style>{`
        @keyframes bounce {
          0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
          30% { transform: translateY(-5px); opacity: 1; }
        }
      `}</style>
    </div>
  )
}

function LawMarkdown({ content }: { content: string }) {
  const lines = content.split('\n')
  return (
    <div className="flex flex-col gap-1.5 text-xs" style={{ lineHeight: 1.75 }}>
      {lines.map((line, i) => {
        if (!line.trim()) return <div key={i} style={{ height: 6 }} />
        if (line.startsWith('## ')) {
          return (
            <h2 key={i} className="text-sm font-bold mt-2 mb-1 pb-1"
              style={{ color: 'var(--accent2)', borderBottom: '1px solid var(--border)' }}>
              {line.slice(3)}
            </h2>
          )
        }
        if (line.startsWith('**') && line.endsWith('**')) {
          return (
            <div key={i} className="font-semibold text-xs" style={{ color: 'var(--text)' }}>
              {line.slice(2, -2)}
            </div>
          )
        }
        const parts = line.split(/\*\*(.*?)\*\*/g)
        return (
          <div key={i}>
            {parts.map((part, j) =>
              j % 2 === 1
                ? <strong key={j} style={{ color: 'var(--text)', fontWeight: 600 }}>{part}</strong>
                : <span key={j} style={{ color: 'var(--muted2)' }}>{part}</span>
            )}
          </div>
        )
      })}
    </div>
  )
}

export default function LawResearchPage() {
  const { sessionId } = useParams<{ sessionId: string }>()
  const navigate = useNavigate()

  const [sessionName, setSessionName]   = useState('New Research')
  const [editingName, setEditingName]   = useState(false)
  const [nameInput, setNameInput]       = useState('')
  const [messages, setMessages]         = useState<Message[]>([])
  const [input, setInput]               = useState('')
  const [streaming, setStreaming]       = useState(false)
  const [loading, setLoading]           = useState(false)

  const messagesEnd  = useRef<HTMLDivElement>(null)
  const textareaRef  = useRef<HTMLTextAreaElement>(null)
  const abortRef     = useRef<AbortController | null>(null)

  // Load session if sessionId provided
  useEffect(() => {
    if (!sessionId) return
    setLoading(true)
    api.getLawSession(sessionId)
      .then(s => {
        setSessionName(s.name)
        setMessages(s.messages.map((m: any, i: number) => ({
          id: `m-${i}`,
          role: m.role,
          content: m.content,
        })))
      })
      .catch(() => navigate('/law-research'))
      .finally(() => setLoading(false))
  }, [sessionId, navigate])

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  function autoResize() {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 120) + 'px'
  }

  async function startEditing() {
    setNameInput(sessionName)
    setEditingName(true)
  }

  async function saveName() {
    if (!sessionId || !nameInput.trim()) { setEditingName(false); return }
    try {
      await api.renameLawSession(sessionId, nameInput.trim())
      setSessionName(nameInput.trim())
    } catch {}
    setEditingName(false)
  }

  async function send(question?: string) {
    const q = (question ?? input).trim()
    if (!q || streaming) return

    // If no session yet, create one first
    let activeSessionId = sessionId
    if (!activeSessionId) {
      try {
        const session = await api.createLawSession()
        activeSessionId = session.id
        navigate(`/law-research/${session.id}`, { replace: true })
      } catch { return }
    }

    setInput('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
    setStreaming(true)

    const userMsg: Message = { id: `u-${Date.now()}`, role: 'user', content: q }
    const assistantId = `a-${Date.now()}`
    const assistantMsg: Message = { id: assistantId, role: 'assistant', content: '' }
    setMessages(prev => [...prev, userMsg, assistantMsg])

    const token = localStorage.getItem('token')
    const controller = new AbortController()
    abortRef.current = controller

    try {
      const res = await fetch(`${BASE}/api/law-research/sessions/${activeSessionId}/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ question: q }),
        signal: controller.signal,
      })

      if (!res.body) throw new Error('No response body')
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let autoNamed = false

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const chunk = JSON.parse(line.slice(6))
            if (chunk.type === 'word') {
              setMessages(prev => prev.map(m =>
                m.id === assistantId ? { ...m, content: m.content + chunk.content } : m
              ))
            } else if (chunk.type === 'name' && !autoNamed) {
              autoNamed = true
              setSessionName(chunk.content)
            } else if (chunk.type === 'error') {
              setMessages(prev => prev.map(m =>
                m.id === assistantId ? { ...m, content: `Error: ${chunk.content}` } : m
              ))
            }
          } catch {}
        }
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        setMessages(prev => prev.map(m =>
          m.id === assistantId ? { ...m, content: 'Request failed. Please try again.' } : m
        ))
      }
    } finally {
      setStreaming(false)
    }
  }

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center" style={{ background: 'var(--bg)' }}>
        <Loader2 size={20} className="animate-spin" style={{ color: 'var(--accent2)' }} />
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col overflow-hidden" style={{ background: 'var(--bg)' }}>

      {/* Header */}
      <header className="flex items-center justify-between px-6 flex-shrink-0"
        style={{ background: 'var(--surface)', borderBottom: '1px solid var(--border)', height: 52 }}>
        <div className="flex items-center gap-3 min-w-0">
          <BookOpen size={15} style={{ color: 'var(--accent2)', flexShrink: 0 }} />
          {editingName ? (
            <div className="flex items-center gap-2">
              <input
                autoFocus
                value={nameInput}
                onChange={e => setNameInput(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') saveName(); if (e.key === 'Escape') setEditingName(false) }}
                className="text-sm outline-none rounded px-2 py-1"
                style={{ background: 'var(--bg)', border: '1px solid var(--border2)', color: 'var(--text)', minWidth: 200 }}
              />
              <button onClick={saveName} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--green)' }}>
                <Check size={13} />
              </button>
              <button onClick={() => setEditingName(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--muted)' }}>
                <X size={13} />
              </button>
            </div>
          ) : (
            <button
              onClick={startEditing}
              className="flex items-center gap-2 text-sm font-semibold truncate group"
              style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text)' }}>
              <span className="truncate">{sessionName}</span>
              <Pencil size={11} className="opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0"
                style={{ color: 'var(--muted)' }} />
            </button>
          )}
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          <span className="text-xs px-2.5 py-1 rounded"
            style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--muted)' }}>
            Law Research · GPT-4o mini
          </span>
        </div>
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-8 py-7 flex flex-col gap-6">
        {messages.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center gap-5 text-center py-12">
            <div className="w-14 h-14 rounded-full flex items-center justify-center"
              style={{ background: 'var(--surface)', border: '1px solid var(--border2)' }}>
              <BookOpen size={24} style={{ color: 'var(--accent2)', opacity: 0.6 }} />
            </div>
            <div>
              <h2 className="font-semibold text-base mb-2" style={{ fontFamily: 'Noto Serif, serif' }}>
                Law Research
              </h2>
              <p className="text-sm max-w-md" style={{ color: 'var(--muted)', lineHeight: 1.65 }}>
                Ask about Indian statutes, constitutional provisions, procedural law, or landmark judgments.
                Answers are based on Indian law as of 2024.
              </p>
            </div>
            <div className="flex flex-col gap-2 w-full max-w-lg mt-2">
              {SUGGESTED.map(q => (
                <button key={q} onClick={() => send(q)}
                  className="text-left px-4 py-2.5 rounded-lg text-xs transition-colors"
                  style={{ background: 'var(--surface)', border: '1px solid var(--border2)', color: 'var(--muted2)' }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--accent2)'; e.currentTarget.style.color = 'var(--text)' }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border2)'; e.currentTarget.style.color = 'var(--muted2)' }}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map(msg => (
            <div key={msg.id} className={`flex flex-col gap-2 ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
              <div className="text-xs uppercase tracking-widest px-1" style={{ color: 'var(--muted)' }}>
                {msg.role === 'user' ? 'You' : 'Law Research'}
              </div>
              {msg.role === 'user' ? (
                <div className="px-4 py-2.5 rounded-xl text-xs max-w-lg"
                  style={{ background: 'var(--accent)', color: 'white', borderRadius: '12px 12px 2px 12px' }}>
                  {msg.content}
                </div>
              ) : (
                <div className="px-5 py-4 rounded-xl max-w-3xl w-full"
                  style={{
                    background: 'var(--surface)',
                    border: '1px solid var(--border)',
                    borderRadius: '2px 12px 12px 12px',
                  }}>
                  {msg.content === ''
                    ? <TypingIndicator />
                    : <LawMarkdown content={msg.content} />
                  }
                  {msg.content !== '' && (
                    <p className="text-xs mt-3 pt-2"
                      style={{ color: 'var(--muted)', borderTop: '1px solid var(--border)' }}>
                      This is legal information, not legal advice. Verify recent developments independently.
                    </p>
                  )}
                </div>
              )}
            </div>
          ))
        )}
        <div ref={messagesEnd} />
      </div>

      {/* Input */}
      <div className="px-6 py-4 flex-shrink-0"
        style={{ background: 'var(--surface)', borderTop: '1px solid var(--border)' }}>
        <div className="flex items-end gap-2.5 rounded-xl px-4 py-2.5"
          style={{ background: 'var(--bg)', border: '1px solid var(--border2)' }}>
          <textarea
            ref={textareaRef}
            value={input}
            onChange={e => { setInput(e.target.value); autoResize() }}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
            placeholder="Ask about Indian law — statutes, sections, judgments, legal principles..."
            rows={1}
            disabled={streaming}
            className="flex-1 bg-transparent outline-none resize-none text-xs"
            style={{ color: 'var(--text)', maxHeight: 120, minHeight: 22, lineHeight: 1.5 }}
          />
          <button
            onClick={() => send()}
            disabled={streaming || !input.trim()}
            className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
            style={{
              background: (streaming || !input.trim()) ? 'var(--border2)' : 'var(--accent)',
              color: 'white', border: 'none', cursor: streaming || !input.trim() ? 'not-allowed' : 'pointer',
            }}>
            {streaming ? <Loader2 size={13} className="animate-spin" /> : <Send size={13} />}
          </button>
        </div>
        <p className="text-xs mt-1.5 px-1" style={{ color: 'var(--muted)' }}>
          Enter to send · Shift+Enter for new line · Covers Indian law as of 2024
        </p>
      </div>
    </div>
  )
}