import type { Case, ResolutionState, StagedDecision } from '../types'

const BASE = '/api'

// ── Cases ──────────────────────────────────────────────────────────────────────

export const api = {
  async getCases(): Promise<Case[]> {
    const res = await fetch(`${BASE}/cases`)
    if (!res.ok) throw new Error('Failed to fetch cases')
    return res.json()
  },

  async getCase(caseId: string): Promise<Case> {
    const res = await fetch(`${BASE}/cases/${caseId}`)
    if (!res.ok) throw new Error('Failed to fetch case')
    return res.json()
  },

  // Upload new case — now accepts domain
  async uploadCase(
    caseName: string,
    file: File,
    domain: string = 'constitutional',
  ): Promise<{ case_id: string }> {
    const form = new FormData()
    form.append('case_name', caseName)
    form.append('pdf_file', file)
    form.append('domain', domain)
    const res = await fetch(`${BASE}/upload`, { method: 'POST', body: form })
    if (!res.ok) throw new Error('Upload failed')
    return res.json()
  },

  // Get available domains for upload dropdown
  async getDomains(): Promise<Array<{ id: string; name: string; description: string }>> {
    const res = await fetch(`${BASE}/domains`)
    if (!res.ok) return []
    return res.json()
  },

  async getKgeStatus(caseId: string): Promise<{ kge_status: string }> {
    const res = await fetch(`${BASE}/kge-status/${caseId}`)
    if (!res.ok) throw new Error('Failed to fetch KGE status')
    return res.json()
  },

  // ── Resolver ────────────────────────────────────────────────────────────────

  async getResolutionState(caseId: string): Promise<ResolutionState> {
    const res = await fetch(`${BASE}/resolution-state/${caseId}`)
    if (!res.ok) throw new Error('Failed to fetch resolution state')
    return res.json()
  },

  async getStagedDecisions(caseId: string): Promise<Record<string, StagedDecision>> {
    const res = await fetch(`${BASE}/staged-status/${caseId}`)
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
    const res = await fetch(`${BASE}/stage/${caseId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ idx, group_id: groupId, decision, buckets }),
    })
    if (!res.ok) throw new Error('Failed to stage decision')
  },

  async confirmAll(caseId: string): Promise<{ status: string; count: number }> {
    const res = await fetch(`${BASE}/confirm-all/${caseId}`, { method: 'POST' })
    if (!res.ok) throw new Error('Failed to confirm decisions')
    return res.json()
  },

  async getConversationHistory(caseId: string) {
    const res = await fetch(`${BASE}/conversation/${caseId}`)
    if (!res.ok) return []
    return res.json()
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

  fetch(`/ask/${caseId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
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