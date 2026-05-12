import type { Case, ResolutionState, StagedDecision } from '../types'

const BASE = '/api'

// ── Auth helpers ───────────────────────────────────────────────────────────────

function authFetch(url: string, options: RequestInit = {}): Promise<Response> {
  const token = localStorage.getItem('token')
  const headers: Record<string, string> = {}

  if (!(options.body instanceof FormData)) {
    headers['Content-Type'] = 'application/json'
  }

  if (token) headers['Authorization'] = `Bearer ${token}`

  const existingHeaders = (options.headers as Record<string, string>) ?? {}
  const { 'Content-Type': _, 'Authorization': __, ...otherHeaders } = existingHeaders

  return fetch(url, {
    ...options,
    headers: {
      ...otherHeaders,
      ...headers,
    },
  })
}

// ── Cases ──────────────────────────────────────────────────────────────────────

export const api = {
  async getCases(): Promise<Case[]> {
    const res = await authFetch(`${BASE}/cases`)
    if (!res.ok) throw new Error('Failed to fetch cases')
    return res.json()
  },

  async getCase(caseId: string): Promise<Case> {
    const res = await authFetch(`${BASE}/cases/${caseId}`)
    if (!res.ok) throw new Error('Failed to fetch case')
    return res.json()
  },

  async uploadCase(
    caseName: string,
    files: File[],
    domain: string = 'constitutional',
  ): Promise<{ case_id: string; doc_count: number; documents: Array<{ doc_id: string; filename: string }> }> {
    const form = new FormData()
    form.append('case_name', caseName)
    form.append('domain', domain)
    for (const file of files) {
      form.append('pdf_files', file)
    }
    const res = await authFetch(`${BASE}/upload`, { method: 'POST', body: form })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.detail || err.error || 'Upload failed')
    }
    return res.json()
  },

  async getDomains(): Promise<Array<{ id: string; name: string; description: string }>> {
    const res = await authFetch(`${BASE}/domains`)
    if (!res.ok) return []
    return res.json()
  },

  async getKgeStatus(caseId: string): Promise<{ kge_status: string }> {
    const res = await authFetch(`${BASE}/kge-status/${caseId}`)
    if (!res.ok) throw new Error('Failed to fetch KGE status')
    return res.json()
  },

  async getResolutionState(caseId: string): Promise<ResolutionState> {
    const res = await authFetch(`${BASE}/resolution-state/${caseId}`)
    if (!res.ok) throw new Error('Failed to fetch resolution state')
    return res.json()
  },

  async getStagedDecisions(caseId: string): Promise<Record<string, StagedDecision>> {
    const res = await authFetch(`${BASE}/staged-status/${caseId}`)
    if (!res.ok) return {}
    return res.json()
  },

  async stageDecision(
    caseId: string,
    idx: number,
    groupId: string,
    decision: 'MERGE' | 'KEEP' | 'SKIP',
    buckets: Record<string, string[]>
  ): Promise<void> {
    const res = await authFetch(`${BASE}/stage/${caseId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ idx, group_id: groupId, decision, buckets }),
    })
    if (!res.ok) throw new Error('Failed to stage decision')
  },

  async confirmAll(caseId: string): Promise<{ status: string; count: number }> {
    const res = await authFetch(`${BASE}/confirm-all/${caseId}`, { method: 'POST' })
    if (!res.ok) throw new Error('Failed to confirm decisions')
    return res.json()
  },

  async getConversationHistory(caseId: string) {
    const res = await authFetch(`${BASE}/conversation/${caseId}`)
    if (!res.ok) return []
    return res.json()
  },

  async deleteCase(caseId: string): Promise<void> {
    const res = await authFetch(`${BASE}/cases/${caseId}`, { method: 'DELETE' })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.error || 'Delete failed')
    }
  },

  // ── Law Research sessions ──────────────────────────────────────────────────

  async getLawSessions(): Promise<Array<{ id: string; name: string; updated_at: string }>> {
    const res = await authFetch(`${BASE}/law-research/sessions`)
    if (!res.ok) return []
    return res.json()
  },

  async createLawSession(): Promise<{ id: string; name: string; updated_at: string }> {
    const res = await authFetch(`${BASE}/law-research/sessions`, { method: 'POST' })
    if (!res.ok) throw new Error('Failed to create session')
    return res.json()
  },

  async getLawSession(sessionId: string): Promise<{
    id: string; name: string; messages: Array<{ role: string; content: string }>
  }> {
    const res = await authFetch(`${BASE}/law-research/sessions/${sessionId}`)
    if (!res.ok) throw new Error('Session not found')
    return res.json()
  },

  async renameLawSession(sessionId: string, name: string): Promise<void> {
    const res = await authFetch(`${BASE}/law-research/sessions/${sessionId}/name`, {
      method: 'PUT',
      body: JSON.stringify({ name }),
    })
    if (!res.ok) throw new Error('Failed to rename session')
  },

  async deleteLawSession(sessionId: string): Promise<void> {
    const res = await authFetch(`${BASE}/law-research/sessions/${sessionId}`, { method: 'DELETE' })
    if (!res.ok) throw new Error('Failed to delete session')
  },
}

// ── SSE streaming for /ask ────────────────────────────────────────────────────

export interface AskChunk {
  type: 'word' | 'done' | 'error'
  content?: string
  citations?: Array<{ text: string; page: number; pdf: string }>
  confidence?: number
  answer_type?: string
  hops?: number
}

export function askQuestion(
  caseId: string,
  question: string,
  onChunk: (chunk: AskChunk) => void,
  onDone: () => void,
  onError: (err: string) => void
): AbortController {
  const controller = new AbortController()
  const token = localStorage.getItem('token')

  fetch(`/ask/${caseId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ question }),
    signal: controller.signal,
  })
    .then(async (res) => {
      if (!res.ok) throw new Error('Ask failed')
      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const data: AskChunk = JSON.parse(line.slice(6))
            onChunk(data)
            if (data.type === 'done') onDone()
          } catch {}
        }
      }
    })
    .catch((err) => {
      if (err.name !== 'AbortError') onError(err.message)
    })

  return controller
}