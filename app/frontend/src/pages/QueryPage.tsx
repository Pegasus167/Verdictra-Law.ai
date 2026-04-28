import { useState, useEffect, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Scale, ArrowLeft, Zap, Cpu, Send, StickyNote, Trash2, ChevronUp, ChevronDown, BookOpen, Loader2 } from 'lucide-react'
import { api, askQuestion } from '../lib/api'
import type { Message, Citation } from '../types'

const SUGGESTED = [
  'Is MIDC justified in demanding ULC charges from CELIR?',
  'What is the relationship between Bafna Motors and Union Bank of India?',
  'What did the Supreme Court order on 21 September 2023?',
  'Who are all the parties involved and what are their roles?',
  'What is the total amount demanded by MIDC and what does it consist of?',
]

const NOTE_COLORS = ['#fef08a', '#86efac', '#93c5fd', '#f9a8d4', '#fdba74']
const BASE = '' // '' for production, 'http://localhost:8000' for development

interface Annotation {
  id: string
  page: number
  pdf: string
  note: string
  position: { x: number; y: number }
  anchor_text: string
  color: string
  created_at: string
}

// Render markdown-style headers and content in deep research output
function DeepResearchContent({ content }: { content: string }) {
  const lines = content.split('\n')
  return (
    <div className="flex flex-col gap-1">
      {lines.map((line, i) => {
        if (line.startsWith('# ')) {
          return <h1 key={i} className="text-base font-bold mt-3 mb-1" style={{ color: 'var(--text)' }}>{line.slice(2)}</h1>
        }
        if (line.startsWith('## ')) {
          return <h2 key={i} className="text-sm font-bold mt-3 mb-1 pb-1" style={{ color: 'var(--accent2)', borderBottom: '1px solid var(--border)' }}>{line.slice(3)}</h2>
        }
        if (line.startsWith('### ')) {
          return <h3 key={i} className="text-xs font-bold mt-2" style={{ color: 'var(--muted2)' }}>{line.slice(4)}</h3>
        }
        if (line.trim() === '') {
          return <div key={i} className="h-1" />
        }
        return <p key={i} className="text-xs leading-relaxed" style={{ color: 'var(--text)' }}>{line}</p>
      })}
    </div>
  )
}

export default function QueryPage() {
  const { caseId } = useParams<{ caseId: string }>()
  const navigate   = useNavigate()

  const [caseName, setCaseName]   = useState('')
  const [kgeStatus, setKgeStatus] = useState<string>('not_started')
  const [messages, setMessages]   = useState<Message[]>([])
  const [input, setInput]         = useState('')
  const [streaming, setStreaming] = useState(false)
  const [deepResearching, setDeepResearching] = useState(false)
  const [uploadingDoc, setUploadingDoc]   = useState(false)
  const [docUploadMsg, setDocUploadMsg]   = useState('')
  const docUploadRef                      = useRef<HTMLInputElement>(null)
  const [pdfOpen, setPdfOpen]     = useState(false)
  const [pdfPage, setPdfPage]     = useState(1)
  const [pdfFile, setPdfFile]     = useState('')
  const [activeCite, setActiveCite] = useState<string | null>(null)

  // Post-it state
  const [annotations, setAnnotations]     = useState<Annotation[]>([])
  const [noteOpen, setNoteOpen]           = useState(false)
  const [notePageInput, setNotePageInput] = useState('')
  const [noteText, setNoteText]           = useState('')
  const [noteColor, setNoteColor]         = useState(NOTE_COLORS[0])
  const [indexOpen, setIndexOpen]         = useState(false)

  const messagesEnd = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const abortRef    = useRef<AbortController | null>(null)

  useEffect(() => {
    if (!caseId) return
    api.getCase(caseId).then(c => {
      setCaseName(c.case_name)
      setPdfFile(c.pdf_filename)
    })
    api.getKgeStatus(caseId).then(s => setKgeStatus(s.kge_status))
    api.getConversationHistory(caseId).then(history => {
      if (history.length > 0) {
        setMessages(history.map((h: any, i: number) => [
          { id: `u-${i}`, role: 'user' as const, content: h.question, timestamp: h.asked_at },
          {
            id: `a-${i}`, role: 'assistant' as const, content: h.answer,
            citations: h.citations, confidence: h.confidence,
            answer_type: h.answer_type, hops: h.hops, timestamp: h.asked_at,
          },
        ]).flat())
      }
    })

    fetch(`${BASE}/annotations/${caseId}`)
      .then(r => r.ok ? r.json() : [])
      .then(setAnnotations)
      .catch(() => {})

    const interval = setInterval(async () => {
      const s = await api.getKgeStatus(caseId!)
      setKgeStatus(s.kge_status)
      if (['ready', 'failed', 'not_started'].includes(s.kge_status)) clearInterval(interval)
    }, 15_000)
    return () => clearInterval(interval)
  }, [caseId])

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    if (pdfOpen) setNotePageInput(String(pdfPage))
  }, [pdfPage, pdfOpen])

  function autoResize() {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 120) + 'px'
  }

  // ── Normal query ──────────────────────────────────────────────────────────

  async function send(question?: string) {
    const q = (question ?? input).trim()
    if (!q || streaming || deepResearching || !caseId) return
    setInput('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
    setStreaming(true)

    const userMsg: Message = {
      id: `u-${Date.now()}`, role: 'user', content: q, timestamp: new Date().toISOString(),
    }
    const assistantId = `a-${Date.now()}`
    const assistantMsg: Message = {
      id: assistantId, role: 'assistant', content: '', timestamp: new Date().toISOString(),
    }
    setMessages(prev => [...prev, userMsg, assistantMsg])

    abortRef.current = askQuestion(
      caseId, q,
      (chunk) => {
        if (chunk.type === 'word') {
          setMessages(prev => prev.map(m =>
            m.id === assistantId ? { ...m, content: m.content + chunk.content } : m
          ))
        } else if (chunk.type === 'done') {
          setMessages(prev => prev.map(m =>
            m.id === assistantId ? {
              ...m,
              citations: chunk.citations,
              confidence: chunk.confidence,
              answer_type: chunk.answer_type as Message['answer_type'],
              hops: chunk.hops,
            } : m
          ))
        } else if (chunk.type === 'error') {
          setMessages(prev => prev.map(m =>
            m.id === assistantId ? { ...m, content: `Error: ${chunk.content}` } : m
          ))
        }
      },
      () => setStreaming(false),
      (err) => { setStreaming(false); console.error(err) }
    )
  }

  // ── Deep Research ─────────────────────────────────────────────────────────

  async function startDeepResearch() {
    if (streaming || deepResearching || !caseId) return
    const q = input.trim() || 'Provide a comprehensive analysis of this case'
    setInput('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
    setDeepResearching(true)

    const userMsg: Message = {
      id: `u-${Date.now()}`,
      role: 'user',
      content: `🔬 Deep Research: ${q}`,
      timestamp: new Date().toISOString(),
    }
    const assistantId = `a-${Date.now()}`
    const assistantMsg: Message = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
    }
    setMessages(prev => [...prev, userMsg, assistantMsg])

    try {
      const token = localStorage.getItem('token')
      const res = await fetch(`${BASE}/deep-research/${caseId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json',
          ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
         },
        body: JSON.stringify({ query: q }),
      })

      if (!res.body) throw new Error('No response body')
      const reader = res.body.getReader()
      const decoder = new TextDecoder()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const text = decoder.decode(value)
        const lines = text.split('\n').filter(l => l.startsWith('data: '))
        for (const line of lines) {
          try {
            const chunk = JSON.parse(line.slice(6))
            if (chunk.type === 'word') {
              setMessages(prev => prev.map(m =>
                m.id === assistantId ? { ...m, content: m.content + chunk.content } : m
              ))
            } else if (chunk.type === 'done') {
              setMessages(prev => prev.map(m =>
                m.id === assistantId ? {
                  ...m,
                  citations: chunk.citations,
                  confidence: chunk.confidence,
                  answer_type: 'DEEP_RESEARCH' as Message['answer_type'],
                  hops: chunk.hops,
                } : m
              ))
            } else if (chunk.type === 'error') {
              setMessages(prev => prev.map(m =>
                m.id === assistantId ? { ...m, content: `Error: ${chunk.content}` } : m
              ))
            }
          } catch {}
        }
      }
    } catch (e) {
      console.error('Deep research failed:', e)
      setMessages(prev => prev.map(m =>
        m.id === assistantId ? { ...m, content: `Deep research failed: ${e}` } : m
      ))
    } finally {
      setDeepResearching(false)
    }
  }

  function openCitation(cite: Citation, key: string) {
    setActiveCite(key)
    setPdfPage(cite.page)
    setPdfFile(cite.pdf)
    setPdfOpen(true)
  }

  // ── Annotations ───────────────────────────────────────────────────────────

  async function saveNote() {
    if (!noteText.trim() || !caseId) return
    const page = parseInt(notePageInput) || pdfPage
    try {
      const res = await fetch(`${BASE}/annotations/${caseId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          page, pdf: pdfFile, note: noteText.trim(),
          position: { x: 0.05, y: 0.05 }, anchor_text: '', color: noteColor,
        }),
      })
      if (res.ok) {
        const ann = await res.json()
        setAnnotations(prev => [...prev, ann])
        setNoteText('')
        setNoteOpen(false)
      }
    } catch {}
  }

  async function deleteAnnotation(id: string) {
    if (!caseId) return
    try {
      await fetch(`${BASE}/annotations/${caseId}/${id}`, { method: 'DELETE' })
      setAnnotations(prev => prev.filter(a => a.id !== id))
    } catch {}
  }
  async function handleDocUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const selectedFiles = Array.from(e.target.files ?? [])
    if (!selectedFiles.length || !caseId) return
    setUploadingDoc(true)
    setDocUploadMsg(`Uploading ${selectedFiles.length} file${selectedFiles.length > 1 ? 's' : ''}...`)
    try {
      const form = new FormData()
      for (const f of selectedFiles) form.append('files', f)
      const token = localStorage.getItem('token')
      const res = await fetch(`/api/cases/${caseId}/documents`, {
        method: 'POST',
        headers: token ? { Authorization: `Bearer ${token}` } : {},
        body: form,
      })
      if (!res.ok) throw new Error('Upload failed')
      setDocUploadMsg(`✓ ${selectedFiles.length} file${selectedFiles.length > 1 ? 's' : ''} added — processing...`)
      setTimeout(() => setDocUploadMsg(''), 4000)
    } catch (err: any) {
      setDocUploadMsg(`✗ ${err.message || 'Upload failed'}`)
      setTimeout(() => setDocUploadMsg(''), 4000)
    } finally {
      setUploadingDoc(false)
      if (docUploadRef.current) docUploadRef.current.value = ''
    }
  }

  function navigateToNote(ann: Annotation) {
    setPdfPage(ann.page); setPdfFile(ann.pdf); setPdfOpen(true); setIndexOpen(false)
  }

  const pdfAnnotations = annotations.filter(a => a.pdf === pdfFile)

  const typeColor: Record<string, string> = {
    DIRECT: '#10b981', PARTIAL: '#f59e0b', INFERRED: '#818cf8', DEEP_RESEARCH: '#a78bfa',
  }
  const typeBg: Record<string, string> = {
    DIRECT: '#05261a', PARTIAL: '#1c1003', INFERRED: '#0f1a3a', DEEP_RESEARCH: '#1a0a2e',
  }

  const isDisabled = streaming || deepResearching

  return (
    <div className="h-screen flex flex-col overflow-hidden" style={{ background: 'var(--bg)' }}>

      {/* Header */}
      <header className="flex items-center justify-between px-5 flex-shrink-0"
        style={{ background: 'var(--surface)', borderBottom: '1px solid var(--border)', height: 52 }}>
        <div className="flex items-center gap-3">
          <button onClick={() => navigate('/')}
            className="flex items-center gap-1 text-xs transition-colors px-2 py-1 rounded"
            style={{ color: 'var(--muted)', border: '1px solid var(--border)' }}>
            <ArrowLeft size={12} /> Cases
          </button>
          <Scale size={14} className="text-indigo-400" />
          <span className="text-sm font-bold text-indigo-300">LAW.ai</span>
          <span className="text-xs" style={{ color: 'var(--muted)' }}>/ {caseName}</span>
        </div>
        <div className="flex items-center gap-2">
          {deepResearching && (
            <span className="flex items-center gap-1 text-xs px-2 py-1 rounded"
              style={{ background: '#1a0a2e', border: '1px solid #7e22ce', color: '#a78bfa' }}>
              <Loader2 size={10} className="animate-spin" /> Deep Research running...
            </span>
          )}
          {kgeStatus === 'training' && (
            <span className="flex items-center gap-1 text-xs px-2 py-1 rounded"
              style={{ background: '#1c1003', border: '1px solid #78350f', color: '#f59e0b' }}>
              <Cpu size={10} className="animate-pulse" /> KGE training...
            </span>
          )}
          {kgeStatus === 'ready' && (
            <span className="flex items-center gap-1 text-xs px-2 py-1 rounded"
              style={{ background: '#05261a', border: '1px solid #065f46', color: '#10b981' }}>
              <Zap size={10} /> KGE enhanced
            </span>
          )}
          {(kgeStatus === 'not_started' || kgeStatus === 'failed') && !deepResearching && (
            <span className="text-xs px-2 py-1 rounded"
              style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--muted)' }}>
              Standard mode
            </span>
          )}
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">

        {/* Chat panel */}
        <div className="flex-1 flex flex-col overflow-hidden min-w-0">
          <div className="flex-1 overflow-y-auto px-8 py-7 flex flex-col gap-7">
            {messages.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center gap-4 text-center py-16">
                <Scale size={32} className="opacity-20" style={{ color: 'var(--muted)' }} />
                <div className="font-semibold text-sm" style={{ color: 'var(--muted2)' }}>{caseName}</div>
                <div className="text-xs max-w-md" style={{ color: 'var(--muted)', lineHeight: 1.7 }}>
                  Ask any question about this case, or use Deep Research for a comprehensive analysis.
                </div>
                <div className="flex flex-col gap-2 w-full max-w-lg mt-2">
                  {SUGGESTED.map(q => (
                    <button key={q} onClick={() => send(q)}
                      className="text-left px-4 py-2.5 rounded-lg text-xs transition-colors"
                      style={{ background: 'var(--surface)', border: '1px solid var(--border2)', color: 'var(--muted2)' }}
                      onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--accent)'; e.currentTarget.style.color = 'var(--text)' }}
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
                    {msg.role === 'user' ? 'You' : 'LAW.ai'}
                  </div>
                  {msg.role === 'user' ? (
                    <div className="px-4 py-2.5 rounded-xl text-xs max-w-lg"
                      style={{ background: 'var(--accent)', color: 'white', borderRadius: '12px 12px 2px 12px' }}>
                      {msg.content}
                    </div>
                  ) : (
                    <div className="px-5 py-4 rounded-xl max-w-4xl"
                      style={{
                        background: msg.answer_type === 'DEEP_RESEARCH' ? '#0f0a1e' : 'var(--surface)',
                        border: `1px solid ${msg.answer_type === 'DEEP_RESEARCH' ? '#4c1d95' : 'var(--border)'}`,
                        lineHeight: 1.85,
                        borderRadius: '2px 12px 12px 12px',
                        color: 'var(--text)',
                      }}>

                      {/* Deep research header */}
                      {msg.answer_type === 'DEEP_RESEARCH' && (
                        <div className="flex items-center gap-2 mb-4 pb-3"
                          style={{ borderBottom: '1px solid #4c1d95' }}>
                          <BookOpen size={14} style={{ color: '#a78bfa' }} />
                          <span className="text-xs font-bold" style={{ color: '#a78bfa' }}>
                            Deep Research Report
                          </span>
                        </div>
                      )}

                      {/* Content — formatted for deep research, plain for normal */}
                      {msg.answer_type === 'DEEP_RESEARCH'
                        ? <DeepResearchContent content={msg.content} />
                        : <div className="text-xs">{msg.content}</div>
                      }

                      {msg.answer_type && (
                        <div className="flex items-center gap-2 mt-3 flex-wrap">
                          <span className="px-2 py-0.5 rounded text-xs font-bold"
                            style={{
                              background: typeBg[msg.answer_type] ?? 'var(--surface2)',
                              color: typeColor[msg.answer_type] ?? 'var(--muted2)',
                              border: `1px solid ${typeColor[msg.answer_type] ?? 'var(--border2)'}`,
                            }}>
                            {msg.answer_type === 'DEEP_RESEARCH' ? '🔬 DEEP RESEARCH' : msg.answer_type}
                          </span>
                          <span className="text-xs px-2 py-0.5 rounded"
                            style={{ background: 'var(--surface2)', color: 'var(--muted2)', border: '1px solid var(--border2)' }}>
                            {Math.round((msg.confidence ?? 0) * 100)}% confidence
                          </span>
                          <span className="text-xs px-2 py-0.5 rounded"
                            style={{ background: 'var(--surface2)', color: 'var(--muted)', border: '1px solid var(--border2)' }}>
                            {msg.hops} hop{msg.hops !== 1 ? 's' : ''}
                          </span>
                        </div>
                      )}
                      {msg.citations && msg.citations.length > 0 && (
                        <div className="flex flex-wrap gap-1.5 mt-3">
                          {msg.citations.map((c, i) => {
                            const key = `${msg.id}-${i}`
                            return (
                              <button key={key}
                                onClick={() => openCitation(c, key)}
                                className="flex items-center gap-1.5 px-2.5 py-1 rounded text-xs transition-colors"
                                style={{
                                  background: activeCite === key ? '#1a1f35' : 'var(--surface2)',
                                  border: `1px solid ${activeCite === key ? 'var(--accent)' : 'var(--border2)'}`,
                                  color: activeCite === key ? 'var(--accent2)' : 'var(--muted2)',
                                }}>
                                📄 Page {c.page}
                              </button>
                            )
                          })}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))
            )}
            <div ref={messagesEnd} />
          </div>

          {/* Input bar */}
          <div className="px-6 py-4 flex-shrink-0"
            style={{ background: 'var(--surface)', borderTop: '1px solid var(--border)' }}>
            <div className="flex items-end gap-2.5 rounded-xl px-4 py-2.5"
              style={{ background: 'var(--bg)', border: '1px solid var(--border2)' }}>
              {/* Add document button */}
              <div className="relative flex-shrink-0">
                <button
                  type="button"
                  onClick={() => docUploadRef.current?.click()}
                  disabled={isDisabled || uploadingDoc}
                  className="w-7 h-7 rounded-lg flex items-center justify-center transition-colors"
                  style={{
                    background: 'var(--surface2)',
                    border: '1px solid var(--border2)',
                    color: uploadingDoc ? 'var(--muted)' : 'var(--muted2)',
                  }}
                  title="Add document to this case">
                  {uploadingDoc ? <Loader2 size={12} className="animate-spin" /> : <span style={{ fontSize: 16, lineHeight: 1 }}>+</span>}
                </button>
                <input
                  ref={docUploadRef}
                  type="file"
                  accept=".pdf,.docx,.doc,.eml,.txt,.jpg,.jpeg,.png"
                  multiple
                  className="hidden"
                  onChange={handleDocUpload}
                />
              </div>
              <textarea
                ref={textareaRef}
                value={input}
                onChange={e => { setInput(e.target.value); autoResize() }}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
                placeholder="Ask a question, or use Deep Research for full case analysis..."
                rows={1}
                disabled={isDisabled}
                className="flex-1 bg-transparent outline-none resize-none text-xs"
                style={{ color: 'var(--text)', maxHeight: 120, minHeight: 22, lineHeight: 1.5 }}
              />

              {/* Deep Research button */}
              <button
                onClick={startDeepResearch}
                disabled={isDisabled}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold flex-shrink-0 transition-colors"
                style={{
                  background: isDisabled ? 'var(--border2)' : '#1a0a2e',
                  border: `1px solid ${isDisabled ? 'var(--border)' : '#7e22ce'}`,
                  color: isDisabled ? 'var(--muted)' : '#a78bfa',
                }}
                title="Deep Research — comprehensive analysis of the entire case">
                {deepResearching
                  ? <Loader2 size={11} className="animate-spin" />
                  : <BookOpen size={11} />
                }
                {deepResearching ? 'Researching...' : 'Deep Research'}
              </button>

              {/* Send button */}
              <button onClick={() => send()} disabled={isDisabled || !input.trim()}
                className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 transition-colors"
                style={{ background: (isDisabled || !input.trim()) ? 'var(--border2)' : 'var(--accent)', color: 'white' }}>
                <Send size={13} />
              </button>
            </div>
            {docUploadMsg && (
              <p className="text-xs mt-1 px-1" style={{ color: docUploadMsg.startsWith('✓') ? '#10b981' : docUploadMsg.startsWith('✗') ? '#ef4444' : 'var(--muted2)' }}>
                {docUploadMsg}
              </p>
            )}
            <p className="text-xs mt-1.5 px-1" style={{ color: 'var(--muted)' }}>
              Enter to send · Shift+Enter for new line · Deep Research for full case analysis
            </p>
          </div>
        </div>

        {/* PDF panel */}
        {pdfOpen && (
          <div className="w-96 flex-shrink-0 flex flex-col overflow-hidden"
            style={{ background: 'var(--surface)', borderLeft: '1px solid var(--border)' }}>

            {/* PDF toolbar */}
            <div className="flex items-center justify-between px-4 py-2.5 flex-shrink-0"
              style={{ borderBottom: '1px solid var(--border)' }}>
              <span className="text-xs uppercase tracking-widest" style={{ color: 'var(--muted)' }}>
                PDF Viewer
              </span>
              <div className="flex items-center gap-2">
                <button onClick={() => setIndexOpen(o => !o)}
                  className="flex items-center gap-1 px-2 py-1 rounded text-xs"
                  style={{
                    background: indexOpen ? '#1a1f35' : 'var(--surface2)',
                    border: `1px solid ${indexOpen ? 'var(--accent)' : 'var(--border)'}`,
                    color: indexOpen ? 'var(--accent2)' : 'var(--muted)',
                  }}>
                  <StickyNote size={11} />
                  {pdfAnnotations.length > 0 && <span className="font-bold">{pdfAnnotations.length}</span>}
                </button>
                <span className="text-xs" style={{ color: 'var(--accent2)' }}>
                  {pdfFile} · p.{pdfPage}
                </span>
                <button onClick={() => { setPdfOpen(false); setActiveCite(null); setNoteOpen(false); setIndexOpen(false) }}
                  className="text-xs px-2 py-1 rounded"
                  style={{ color: 'var(--muted)', border: '1px solid var(--border)' }}>
                  ✕
                </button>
              </div>
            </div>

            {/* Notes index */}
            {indexOpen && (
              <div className="flex-shrink-0 overflow-y-auto"
                style={{ maxHeight: 240, borderBottom: '1px solid var(--border)', background: 'var(--bg)' }}>
                <div className="px-3 py-2 text-xs uppercase tracking-widest"
                  style={{ color: 'var(--muted)', borderBottom: '1px solid var(--border)' }}>
                  All notes — {pdfFile}
                </div>
                {pdfAnnotations.length === 0 ? (
                  <div className="px-3 py-4 text-xs text-center" style={{ color: 'var(--muted)' }}>No notes yet</div>
                ) : (
                  pdfAnnotations.sort((a, b) => a.page - b.page).map(ann => (
                    <div key={ann.id}
                      className="flex items-start gap-2 px-3 py-2 cursor-pointer"
                      style={{ borderBottom: '1px solid var(--border)', background: 'transparent' }}
                      onMouseEnter={e => (e.currentTarget.style.background = 'var(--surface)')}
                      onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                      onClick={() => navigateToNote(ann)}>
                      <div className="w-3 h-3 rounded-sm flex-shrink-0 mt-0.5" style={{ background: ann.color }} />
                      <div className="flex-1 min-w-0">
                        <div className="text-xs font-medium mb-0.5" style={{ color: 'var(--accent2)' }}>p.{ann.page}</div>
                        <div className="text-xs truncate" style={{ color: 'var(--muted2)' }}>{ann.note}</div>
                      </div>
                      <button onClick={e => { e.stopPropagation(); deleteAnnotation(ann.id) }}
                        className="w-5 h-5 flex items-center justify-center rounded flex-shrink-0"
                        style={{ color: 'var(--muted)' }}>
                        <Trash2 size={10} />
                      </button>
                    </div>
                  ))
                )}
              </div>
            )}

            {/* iframe */}
            <iframe
              key={`${pdfFile}-${pdfPage}`}
              src={`/pdf/${caseId}/${pdfFile}#page=${pdfPage}`}
              className="flex-1 w-full"
              style={{ border: 'none', background: '#111' }}
              title="PDF Viewer"
            />

            {/* Post-it panel */}
            <div className="flex-shrink-0" style={{ borderTop: '1px solid var(--border)', background: 'var(--surface)' }}>
              <button
                onClick={() => setNoteOpen(o => !o)}
                className="w-full flex items-center justify-between px-4 py-2.5 text-xs"
                style={{ color: 'var(--muted2)' }}
                onMouseEnter={e => (e.currentTarget.style.background = 'var(--surface2)')}
                onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}>
                <div className="flex items-center gap-2">
                  <StickyNote size={12} />
                  <span>Add a note</span>
                </div>
                {noteOpen ? <ChevronDown size={12} /> : <ChevronUp size={12} />}
              </button>

              {noteOpen && (
                <div className="px-4 pb-4 flex flex-col gap-2.5">
                  <div className="flex items-center gap-2">
                    <span className="text-xs" style={{ color: 'var(--muted)' }}>Page:</span>
                    <input type="number" value={notePageInput} onChange={e => setNotePageInput(e.target.value)}
                      min={1} className="w-16 rounded px-2 py-1 text-xs outline-none"
                      style={{ background: 'var(--bg)', border: '1px solid var(--border2)', color: 'var(--text)' }} />
                    <span className="text-xs" style={{ color: 'var(--muted)' }}>(viewing p.{pdfPage})</span>
                  </div>
                  <textarea autoFocus value={noteText} onChange={e => setNoteText(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); saveNote() }
                      if (e.key === 'Escape') { setNoteOpen(false); setNoteText('') }
                    }}
                    placeholder="Type your note... (Enter to save, Esc to cancel)"
                    rows={3} className="w-full rounded-md px-2.5 py-1.5 text-xs outline-none resize-none"
                    style={{ background: 'var(--bg)', border: '1px solid var(--border2)', color: 'var(--text)', fontFamily: 'monospace' }} />
                  <div className="flex items-center gap-2">
                    <span className="text-xs" style={{ color: 'var(--muted)' }}>Color:</span>
                    {NOTE_COLORS.map(c => (
                      <button key={c} onClick={() => setNoteColor(c)}
                        style={{ width: 16, height: 16, background: c, borderRadius: 3, border: noteColor === c ? '2px solid white' : '1px solid transparent', cursor: 'pointer' }} />
                    ))}
                    <div className="ml-auto flex gap-2">
                      <button onClick={() => { setNoteOpen(false); setNoteText('') }}
                        className="px-2 py-1 rounded text-xs"
                        style={{ color: 'var(--muted)', border: '1px solid var(--border)' }}>Cancel</button>
                      <button onClick={saveNote} disabled={!noteText.trim()}
                        className="px-2 py-1 rounded text-xs font-bold"
                        style={{ background: noteText.trim() ? 'var(--accent)' : 'var(--border2)', color: 'white' }}>Save</button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}