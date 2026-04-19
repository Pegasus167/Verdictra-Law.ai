import { useState, useEffect, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  ChevronLeft, ChevronRight, Plus, Minus,
  Check, X, SkipForward, Trash2, PanelLeftClose, PanelLeftOpen,
} from 'lucide-react'
import { api } from '../lib/api'
import PDFViewer from '../components/PDFViewer'
import type { Group, SidebarItem, StagedDecision, ResolutionState } from '../types'

const DOT_COLOR: Record<string, string> = {
  PENDING: '#243044',
  MERGE:   '#10b981',
  KEEP:    '#f59e0b',
  SKIP:    '#64748b',
}

// ── Per-group UI state persisted in a ref so navigation doesn't reset it ──────
interface GroupUIState {
  bucketNames: string[]
  assignments: Record<number, number>
  deletedCandidateIndices: Set<number>
}

export default function ReviewPage() {
  const { caseId } = useParams<{ caseId: string }>()
  const navigate = useNavigate()

  const [state, setState] = useState<ResolutionState | null>(null)
  const [staged, setStaged] = useState<Record<string, StagedDecision>>({})
  const [currentIdx, setCurrentIdx] = useState(0)
  const [confirming, setConfirming] = useState(false)
  const [pdfOpen, setPdfOpen] = useState(false)
  const [pdfPage, setPdfPage] = useState(1)
  const [pdfFile, setPdfFile] = useState('')
  const [activeCitation, setActiveCitation] = useState<string | undefined>()
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [deletedGroups, setDeletedGroups] = useState<Set<number>>(new Set())

  // Per-group UI state — stored in a ref so it survives re-renders
  // and navigation within the same component mount
  const groupUIRef = useRef<Record<number, GroupUIState>>({})

  function getGroupUI(idx: number, group: Group): GroupUIState {
    if (!groupUIRef.current[idx]) {
      groupUIRef.current[idx] = {
        bucketNames: [group.canonical_name || ''],
        assignments: {},
        deletedCandidateIndices: new Set(),
      }
    }
    return groupUIRef.current[idx]
  }

  function updateGroupUI(idx: number, update: Partial<GroupUIState>) {
  groupUIRef.current[idx] = { ...groupUIRef.current[idx], ...update }
  // Persist to localStorage — no server call
  const key = `law_ai_review_${caseId}`
  const toSave: Record<string, any> = {}
  Object.entries(groupUIRef.current).forEach(([i, state]) => {
    toSave[i] = {
      bucketNames: state.bucketNames,
      assignments: state.assignments,
      deletedCandidateIndices: Array.from(state.deletedCandidateIndices),
    }
  })
  localStorage.setItem(key, JSON.stringify(toSave))
  setRenderTick(t => t + 1)
}

  const [renderTick, setRenderTick] = useState(0)

  // Load state + staged decisions on mount
  useEffect(() => {
  if (!caseId) return
  Promise.all([
    api.getResolutionState(caseId),
    api.getStagedDecisions(caseId),
  ]).then(([s, d]) => {
    setState(s)
    setStaged(d)
    // Restore UI state from localStorage
    const key = `law_ai_review_${caseId}`
    const saved = localStorage.getItem(key)
    if (saved) {
      try {
        const parsed = JSON.parse(saved)
        Object.entries(parsed).forEach(([idx, state]: [string, any]) => {
          groupUIRef.current[Number(idx)] = {
            bucketNames: state.bucketNames ?? [''],
            assignments: state.assignments ?? {},
            deletedCandidateIndices: new Set(state.deletedCandidateIndices ?? []),
          }
        })
      } catch {}
    }
    const savedDeleted = localStorage.getItem(`law_ai_deleted_${caseId}`)
    if (savedDeleted) {
      try {
        setDeletedGroups(new Set(JSON.parse(savedDeleted)))
      } catch {}
    }
    setRenderTick(t => t + 1)
  })
}, [caseId])

  // Active group (excluding deleted groups)
  const activeNeeds = (state?.needs_review ?? []).filter((_, i) => !deletedGroups.has(i))
  const group: Group | null = state?.needs_review[currentIdx] ?? null
  const total = state?.needs_review.length ?? 0
  const activeTotal = activeNeeds.length

  // Current group UI state
  const ui = group ? getGroupUI(currentIdx, group) : null

  // Sidebar items
  const sidebar: SidebarItem[] = (state?.needs_review ?? []).map((g, i) => ({
    idx: i,
    label: g.canonical_name || g.schema_type || `Group ${i + 1}`,
    schema_type: g.schema_type,
    status: deletedGroups.has(i)
      ? 'SKIP'
      : (staged[String(i)]?.decision as SidebarItem['status']) ?? 'PENDING',
  }))

  const stagedCount = Object.keys(staged).length + deletedGroups.size
  const progress = total > 0 ? Math.round(stagedCount / total * 100) : 0

  // ── Bucket operations ────────────────────────────────────────────────────────

  function addBucket() {
    if (!ui) return
    updateGroupUI(currentIdx, {
      bucketNames: [...ui.bucketNames, ''],
    })
  }

  function removeBucket() {
    if (!ui || ui.bucketNames.length <= 1) return
    const removed = ui.bucketNames.length - 1
    const newAssignments = { ...ui.assignments }
    Object.keys(newAssignments).forEach(k => {
      if (newAssignments[Number(k)] === removed) delete newAssignments[Number(k)]
    })
    updateGroupUI(currentIdx, {
      bucketNames: ui.bucketNames.slice(0, -1),
      assignments: newAssignments,
    })
  }

  function setBucketName(i: number, value: string) {
    if (!ui) return
    const next = [...ui.bucketNames]
    next[i] = value
    updateGroupUI(currentIdx, { bucketNames: next })
  }

  function setAssignment(candidateIdx: number, bucketIdx: number | undefined) {
    if (!ui) return
    const next = { ...ui.assignments }
    if (bucketIdx === undefined) delete next[candidateIdx]
    else next[candidateIdx] = bucketIdx
    updateGroupUI(currentIdx, { assignments: next })
  }

  // ── Candidate deletion ───────────────────────────────────────────────────────

  function deleteCandidate(candidateIdx: number) {
    if (!ui) return
    const next = new Set(ui.deletedCandidateIndices)
    next.add(candidateIdx)
    const newAssignments = { ...ui.assignments }
    delete newAssignments[candidateIdx]
    updateGroupUI(currentIdx, {
      deletedCandidateIndices: next,
      assignments: newAssignments,
    })
  }

  // ── Group deletion ───────────────────────────────────────────────────────────

  async function deleteGroup() {
    if (!caseId || !group) return
    // Stage as SKIP then mark deleted
    await api.stageDecision(caseId, currentIdx, group.group_id, 'SKIP', {})
    const newStaged = {
      ...staged,
      [String(currentIdx)]: {
        group_id: group.group_id,
        decision: 'SKIP' as const,
        buckets: {},
        staged_at: new Date().toISOString(),
      }
    }
    setStaged(newStaged)
    const newDeleted = new Set([...deletedGroups, currentIdx])
    setDeletedGroups(newDeleted)
    localStorage.setItem(`law_ai_deleted_${caseId}`, JSON.stringify(Array.from(newDeleted)))
    // Advance to next non-deleted group
    if (currentIdx < total - 1) setCurrentIdx(i => i + 1)
  }

  // ── Build buckets for staging ────────────────────────────────────────────────

  function buildBuckets(): Record<string, string[]> {
    if (!group || !ui) return {}
    const buckets: Record<string, string[]> = {}
    ui.bucketNames.forEach(name => { if (name.trim()) buckets[name.trim()] = [] })
    // Only include non-deleted candidates
    group.candidates.forEach((c, i) => {
      if (ui.deletedCandidateIndices.has(i)) return
      const bi = ui.assignments[i]
      if (bi !== undefined) {
        const name = ui.bucketNames[bi]?.trim()
        if (name && buckets[name]) buckets[name].push(c.canonical_name)
      }
    })
    return buckets
  }

  // ── Stage decision ───────────────────────────────────────────────────────────

  async function stageDecision(decision: 'MERGE' | 'KEEP' | 'SKIP') {
    if (!caseId || !group) return
    const buckets = decision === 'MERGE' ? buildBuckets() : {}
    await api.stageDecision(caseId, currentIdx, group.group_id, decision, buckets)
    const newStaged = {
      ...staged,
      [String(currentIdx)]: {
        group_id: group.group_id,
        decision,
        buckets,
        staged_at: new Date().toISOString(),
      }
    }
    setStaged(newStaged)
    if (currentIdx < total - 1) setCurrentIdx(i => i + 1)
  }

  // ── Confirm all ──────────────────────────────────────────────────────────────

  async function confirmAll() {
  if (!caseId) return
  const count = Object.keys(staged).length
  if (count === 0) return alert('No staged decisions to confirm.')
  if (!confirm(`Confirm and apply ${count} decision(s) to Neo4j?`)) return
  setConfirming(true)
  try {
    await api.confirmAll(caseId)
    // Clear localStorage for this case — review is done
    localStorage.removeItem(`law_ai_review_${caseId}`)
    localStorage.removeItem(`law_ai_deleted_${caseId}`)
    navigate(`/complete/${caseId}`)
  } catch {
    alert('Failed to confirm decisions')
    setConfirming(false)
  }
}

  // ── Loading state ─────────────────────────────────────────────────────────────

  if (!state || !group || !ui) {
    return (
      <div className="h-screen flex items-center justify-center" style={{ background: 'var(--bg)' }}>
        <div className="text-sm" style={{ color: 'var(--muted)' }}>Loading...</div>
      </div>
    )
  }

  return (
    <div className="h-screen flex flex-col overflow-hidden" style={{ background: 'var(--bg)' }}>
      <div style={{ display: 'none' }}>{renderTick}</div>

      {/* Header */}
      <header className="flex items-center justify-between px-4 flex-shrink-0"
        style={{ background: 'var(--surface)', borderBottom: '1px solid var(--border)', height: 52 }}>
        <div className="flex items-center gap-3">
          <span className="font-bold text-indigo-300 text-sm tracking-wider">LAW.ai / RESOLVE</span>
          <div className="flex items-center gap-2">
            <div className="w-40 h-0.5 rounded-full overflow-hidden" style={{ background: 'var(--border2)' }}>
              <div className="h-full rounded-full transition-all duration-300"
                style={{ width: `${progress}%`, background: 'var(--accent)' }} />
            </div>
            <span className="text-xs" style={{ color: 'var(--muted)' }}>
              {stagedCount}/{total}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs px-2.5 py-1.5 rounded-md"
            style={{ background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--muted)' }}>
            {stagedCount} staged · {activeTotal} active
          </span>
          <button
            onClick={confirmAll}
            disabled={confirming || stagedCount === 0}
            className="px-4 py-1.5 rounded-md text-xs font-bold transition-colors"
            style={{ background: stagedCount === 0 ? 'var(--border2)' : 'var(--accent)', color: 'white' }}>
            {confirming ? 'Confirming...' : 'Confirm All Changes'}
          </button>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">

        {/* Sidebar */}
        <div
          className="flex-shrink-0 flex flex-col overflow-hidden transition-all duration-200"
          style={{
            width: sidebarCollapsed ? 40 : 224,
            background: 'var(--surface)',
            borderRight: '1px solid var(--border)',
          }}>

          {/* Sidebar header with collapse toggle */}
          <div className="flex items-center justify-between px-2 py-2 flex-shrink-0"
            style={{ borderBottom: '1px solid var(--border)', height: 36 }}>
            {!sidebarCollapsed && (
              <span className="text-xs tracking-widest uppercase px-1"
                style={{ color: 'var(--muted)' }}>
                Groups ({activeTotal})
              </span>
            )}
            <button
              onClick={() => setSidebarCollapsed(c => !c)}
              className="w-6 h-6 flex items-center justify-center rounded transition-colors ml-auto"
              style={{ color: 'var(--muted)', background: 'transparent' }}>
              {sidebarCollapsed
                ? <PanelLeftOpen size={14} />
                : <PanelLeftClose size={14} />}
            </button>
          </div>

          <div className="flex-1 overflow-y-auto py-1">
            {sidebar.map(item => (
              <button
                key={item.idx}
                onClick={() => setCurrentIdx(item.idx)}
                className="w-full flex items-center gap-2 py-1.5 text-left transition-colors"
                style={{
                  paddingLeft: sidebarCollapsed ? 12 : 12,
                  paddingRight: sidebarCollapsed ? 12 : 8,
                  borderLeft: `2px solid ${item.idx === currentIdx ? 'var(--accent)' : 'transparent'}`,
                  background: item.idx === currentIdx ? '#1a1f35' : 'transparent',
                  opacity: deletedGroups.has(item.idx) ? 0.4 : 1,
                }}
              >
                <span className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                  style={{ background: DOT_COLOR[item.status] ?? '#243044' }} />
                {!sidebarCollapsed && (
                  <>
                    <span className="text-xs truncate flex-1"
                      style={{ color: item.idx === currentIdx ? 'var(--text)' : 'var(--muted2)' }}>
                      {item.label.slice(0, 24)}
                    </span>
                    <span className="text-xs flex-shrink-0" style={{ color: 'var(--muted)' }}>
                      {item.idx + 1}
                    </span>
                  </>
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Center */}
        <div className="flex-1 overflow-y-auto p-5 min-w-0">

          {/* Group meta + delete group */}
          <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="px-2.5 py-1 rounded text-xs"
                style={{ background: '#1e1b4b', color: '#a5b4fc', border: '1px solid #3730a3' }}>
                {group.schema_type}
              </span>
              {group.llm_model_used && (
                <span className="px-2.5 py-1 rounded text-xs"
                  style={{ background: '#0f2a42', color: '#60a5fa', border: '1px solid #1e40af' }}>
                  {group.llm_model_used}
                </span>
              )}
              {group.is_complex && (
                <span className="px-2.5 py-1 rounded text-xs"
                  style={{ background: '#1a0a2e', color: '#c084fc', border: '1px solid #7e22ce' }}>
                  complex
                </span>
              )}
              <span className="text-xs" style={{ color: 'var(--muted)' }}>
                Group {currentIdx + 1} of {total}
              </span>
            </div>
            {/* Delete entire group */}
            <button
              onClick={deleteGroup}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-colors"
              style={{ background: '#1f0a0a', border: '1px solid #7f1d1d', color: '#f87171' }}
              title="Delete this entire group">
              <Trash2 size={11} /> Delete Group
            </button>
          </div>

          {/* Bucket controls */}
          <div className="rounded-xl p-4 mb-4"
            style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs tracking-widest uppercase" style={{ color: 'var(--muted)' }}>
                Canonical name buckets
              </span>
              <div className="flex items-center gap-2">
                <button onClick={removeBucket}
                  className="w-6 h-6 rounded flex items-center justify-center"
                  style={{ background: 'var(--surface2)', border: '1px solid var(--border2)', color: 'var(--muted2)' }}>
                  <Minus size={12} />
                </button>
                <span className="text-xs w-4 text-center">{ui.bucketNames.length}</span>
                <button onClick={addBucket}
                  className="w-6 h-6 rounded flex items-center justify-center"
                  style={{ background: 'var(--surface2)', border: '1px solid var(--border2)', color: 'var(--muted2)' }}>
                  <Plus size={12} />
                </button>
              </div>
            </div>
            <div className="flex flex-col gap-2">
              {ui.bucketNames.map((name, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span className="text-xs w-4 text-right" style={{ color: 'var(--muted)' }}>{i + 1}</span>
                  <input
                    value={name}
                    onChange={e => setBucketName(i, e.target.value)}
                    placeholder={`Canonical name ${i + 1}...`}
                    className="flex-1 rounded-md px-2.5 py-1.5 text-xs outline-none"
                    style={{ background: 'var(--bg)', border: '1px solid var(--border2)', color: 'var(--text)' }}
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Candidates */}
          <div className="flex flex-col gap-2.5 mb-4">
            {group.candidates.map((c, i) => {
              if (ui.deletedCandidateIndices.has(i)) return null
              const conf = c.llm_merge_confidence ?? 0
              const vote = c.llm_vote
              const assignedBucket = ui.assignments[i]
              return (
                <div key={c.canonical_name} className="rounded-xl p-4 relative"
                  style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>

                  {/* Delete candidate button */}
                  <button
                    onClick={() => deleteCandidate(i)}
                    className="absolute top-3 right-3 w-6 h-6 flex items-center justify-center rounded transition-colors"
                    style={{ color: 'var(--muted)', border: '1px solid var(--border)' }}
                    title="Remove this candidate from group">
                    <Trash2 size={11} />
                  </button>

                  <div className="font-bold text-sm mb-1 pr-8">{c.text}</div>
                  <div className="flex gap-3 text-xs mb-2" style={{ color: 'var(--muted)' }}>
                    <button
                      onClick={() => { setPdfPage(c.source_page); setPdfFile(c.source_pdf); setActiveCitation(c.context); setPdfOpen(true) }}
                      className="flex items-center gap-1 transition-colors"
                      style={{ color: 'var(--accent2)' }}
                      onMouseEnter={e => e.currentTarget.style.textDecoration = 'underline'}
                      onMouseLeave={e => e.currentTarget.style.textDecoration = 'none'}>
                      📄 Page {c.source_page}
                    </button>
                    <span>{c.source_pdf}</span>
                  </div>
                  {c.context && (
                    <div className="rounded-md p-2.5 text-xs mb-2 font-mono"
                      style={{ background: 'var(--bg)', border: '1px solid var(--border)', color: 'var(--muted2)' }}>
                      {c.context}
                    </div>
                  )}
                  {vote && (
                    <div className="rounded-md p-2 text-xs mb-2"
                      style={{
                        background: vote === 'YES' ? '#05261a' : vote === 'NO' ? '#1f0a0a' : '#1c1003',
                        border: `1px solid ${vote === 'YES' ? '#065f46' : vote === 'NO' ? '#7f1d1d' : '#78350f'}`,
                      }}>
                      <span style={{ color: vote === 'YES' ? '#10b981' : vote === 'NO' ? '#ef4444' : '#f59e0b', fontWeight: 700 }}>
                        {vote === 'YES' ? '✓ MERGE' : vote === 'NO' ? '✗ KEEP' : '? UNSURE'}
                      </span>
                      {' — '}{c.llm_reason}
                      <div className="flex items-center gap-2 mt-1">
                        <div className="flex-1 h-0.5 rounded overflow-hidden" style={{ background: 'var(--border)' }}>
                          <div className="h-full rounded"
                            style={{
                              width: `${Math.round(conf * 100)}%`,
                              background: conf >= 0.75 ? '#10b981' : conf >= 0.4 ? '#f59e0b' : '#ef4444',
                            }} />
                        </div>
                        <span style={{ color: 'var(--muted)' }}>{Math.round(conf * 100)}%</span>
                      </div>
                    </div>
                  )}

                  {/* Bucket assignment */}
                  <div className="flex items-center gap-2">
                    <span className="text-xs" style={{ color: 'var(--muted)' }}>Assign to:</span>
                    <select
                      value={assignedBucket !== undefined ? String(assignedBucket) : 'none'}
                      onChange={e => {
                        const val = e.target.value
                        setAssignment(i, val === 'none' ? undefined : Number(val))
                      }}
                      className="rounded text-xs px-2 py-1 outline-none"
                      style={{ background: 'var(--bg)', border: '1px solid var(--border2)', color: 'var(--text)' }}>
                      <option value="none">— none —</option>
                      {ui.bucketNames.map((name, bi) => (
                        <option key={bi} value={String(bi)}>
                          {name.trim() || `Canonical ${bi + 1}`}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              )
            })}
          </div>

          {/* Action bar */}
          <div className="rounded-xl p-4 flex items-center gap-3 mb-4 flex-wrap"
            style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
            <button onClick={() => stageDecision('MERGE')}
              className="px-5 py-2 rounded-lg text-xs font-bold transition-colors flex items-center gap-1.5"
              style={{ background: 'var(--accent)', color: 'white' }}>
              <Check size={12} /> Stage Merge
            </button>
            <button onClick={() => stageDecision('KEEP')}
              className="px-5 py-2 rounded-lg text-xs font-bold transition-colors flex items-center gap-1.5"
              style={{ background: 'var(--surface2)', color: 'var(--muted2)', border: '1px solid var(--border2)' }}>
              <X size={12} /> Stage Keep Separate
            </button>
            <button onClick={() => stageDecision('SKIP')}
              className="px-4 py-2 rounded-lg text-xs transition-colors flex items-center gap-1.5"
              style={{ background: 'transparent', color: 'var(--muted)', border: '1px solid var(--border)' }}>
              <SkipForward size={12} /> Skip
            </button>
            {staged[String(currentIdx)] && (
              <span className="ml-auto text-xs px-2.5 py-1 rounded"
                style={{ background: '#05261a', border: '1px solid #065f46', color: '#10b981' }}>
                ✓ staged: {staged[String(currentIdx)].decision.toLowerCase()}
              </span>
            )}
          </div>

          {/* Nav */}
          <div className="flex justify-between items-center pb-8">
            <button onClick={() => setCurrentIdx(i => Math.max(0, i - 1))}
              disabled={currentIdx === 0}
              className="flex items-center gap-1 text-xs px-3 py-1.5 rounded"
              style={{ color: 'var(--muted)', border: '1px solid transparent' }}>
              <ChevronLeft size={14} /> Previous
            </button>
            <button onClick={() => navigate('/')}
              className="text-xs" style={{ color: 'var(--muted)' }}>↩ Cases</button>
            <button onClick={() => setCurrentIdx(i => Math.min(total - 1, i + 1))}
              disabled={currentIdx === total - 1}
              className="flex items-center gap-1 text-xs px-3 py-1.5 rounded"
              style={{ color: 'var(--muted)', border: '1px solid transparent' }}>
              Next <ChevronRight size={14} />
            </button>
          </div>
        </div>

        {/* PDF panel */}
        {pdfOpen && (
          <PDFViewer
            caseId={caseId!}
            pdfFile={pdfFile}
            page={pdfPage}
            highlightText={activeCitation}
            onClose={() => setPdfOpen(false)}
          />
        )}
      </div>
    </div>
  )
}