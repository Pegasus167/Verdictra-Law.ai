/**
 * PDFViewer.tsx
 *
 * PDF.js viewer with:
 * - CSS transform zoom (instant, smooth, browser handles focal point)
 * - Debounced full re-render after zoom stops (keeps text sharp)
 * - Pinch-to-zoom (trackpad) + Ctrl+scroll (mouse)
 * - Text highlight on citation/context text (temporary)
 * - Post-it sticky notes (permanent, saved to backend)
 * - Notes index in toolbar
 */

import { useEffect, useRef, useState, useCallback } from 'react'
import { X, StickyNote, Trash2, ChevronLeft, ChevronRight } from 'lucide-react'

// ── Types ──────────────────────────────────────────────────────────────────────

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

interface PDFViewerProps {
  caseId: string
  pdfFile: string
  page: number
  highlightText?: string
  onClose: () => void
  width?: number
}

// ── Constants ──────────────────────────────────────────────────────────────────

const PDFJS_CDN     = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js'
const PDFJS_WORKER  = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js'
const BASE          = 'http://localhost:8000'
const NOTE_COLORS   = ['#fef08a', '#86efac', '#93c5fd', '#f9a8d4', '#fdba74']
const DEFAULT_SCALE = 1.4

// ── API helpers ────────────────────────────────────────────────────────────────

async function fetchAnnotations(caseId: string): Promise<Annotation[]> {
  try {
    const res = await fetch(`${BASE}/annotations/${caseId}`)
    if (!res.ok) return []
    return res.json()
  } catch { return [] }
}

async function saveAnnotation(
  caseId: string,
  ann: Omit<Annotation, 'id' | 'created_at'>,
): Promise<Annotation | null> {
  try {
    const res = await fetch(`${BASE}/annotations/${caseId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(ann),
    })
    if (!res.ok) return null
    return res.json()
  } catch { return null }
}

async function deleteAnnotation(caseId: string, id: string): Promise<void> {
  try {
    await fetch(`${BASE}/annotations/${caseId}/${id}`, { method: 'DELETE' })
  } catch {}
}

// ── Main Component ─────────────────────────────────────────────────────────────

export default function PDFViewer({
  caseId,
  pdfFile,
  page,
  highlightText,
  onClose,
  width = 480,
}: PDFViewerProps) {
  const canvasRef      = useRef<HTMLCanvasElement>(null)
  const overlayRef     = useRef<HTMLDivElement>(null)
  const scrollRef      = useRef<HTMLDivElement>(null)
  const canvasWrapRef  = useRef<HTMLDivElement>(null)
  const pdfRef         = useRef<any>(null)
  const renderTaskRef  = useRef<any>(null)
  const zoomTimerRef   = useRef<any>(null)

  const [currentPage, setCurrentPage] = useState(page)
  const [totalPages, setTotalPages]   = useState(0)
  const [renderScale, setRenderScale] = useState(DEFAULT_SCALE)  // actual PDF render scale
  const [cssScale, setCssScale]       = useState(1)               // live CSS zoom multiplier
  const [loading, setLoading]         = useState(true)
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [addingNote, setAddingNote]   = useState(false)
  const [pendingNote, setPendingNote] = useState<{ x: number; y: number } | null>(null)
  const [noteText, setNoteText]       = useState('')
  const [noteColor, setNoteColor]     = useState(NOTE_COLORS[0])
  const [showIndex, setShowIndex]     = useState(false)
  const [activeNote, setActiveNote]   = useState<string | null>(null)

  // Displayed zoom = renderScale × cssScale
  const displayZoom = Math.round(renderScale * cssScale * 100)

  // ── Load PDF.js ────────────────────────────────────────────────────────────

  useEffect(() => {
    if ((window as any).pdfjsLib) return
    const script    = document.createElement('script')
    script.src      = PDFJS_CDN
    script.onload   = () => {
      ;(window as any).pdfjsLib.GlobalWorkerOptions.workerSrc = PDFJS_WORKER
    }
    document.head.appendChild(script)
  }, [])

  // ── Load annotations ───────────────────────────────────────────────────────

  useEffect(() => {
    fetchAnnotations(caseId).then(setAnnotations)
  }, [caseId])

  // ── Sync page prop ─────────────────────────────────────────────────────────

  useEffect(() => {
    setCurrentPage(page)
  }, [page])

  // ── Load PDF document ──────────────────────────────────────────────────────

  useEffect(() => {
    const tryLoad = () => {
      const pdfjsLib = (window as any).pdfjsLib
      if (!pdfjsLib) { setTimeout(tryLoad, 500); return }
      setLoading(true)
      pdfjsLib.getDocument(`${BASE}/pdf/${caseId}/${pdfFile}`).promise
        .then((pdf: any) => {
          pdfRef.current = pdf
          setTotalPages(pdf.numPages)
        })
        .catch((e: any) => { console.error('PDF load failed:', e); setLoading(false) })
    }
    tryLoad()
  }, [caseId, pdfFile])

  // ── Re-render when page or renderScale changes ─────────────────────────────

  useEffect(() => {
    if (!pdfRef.current) return
    renderPage()
  }, [currentPage, renderScale, pdfRef.current])

  // ── Render one page ────────────────────────────────────────────────────────

  const renderPage = useCallback(async () => {
    const doc = pdfRef.current
    if (!doc || !canvasRef.current) return

    if (renderTaskRef.current) {
      try { renderTaskRef.current.cancel() } catch {}
    }

    setLoading(true)

    try {
      const pg       = await doc.getPage(currentPage)
      const viewport = pg.getViewport({ scale: renderScale })
      const canvas   = canvasRef.current
      const ctx      = canvas.getContext('2d')!

      canvas.width  = viewport.width
      canvas.height = viewport.height

      const task = pg.render({ canvasContext: ctx, viewport })
      renderTaskRef.current = task
      await task.promise

      setLoading(false)

      // Clear old highlights
      if (overlayRef.current) {
        overlayRef.current
          .querySelectorAll('.text-highlight')
          .forEach(el => el.remove())
      }

      if (highlightText) {
        await highlightTextOnPage(pg, viewport, highlightText)
      }
    } catch (e: any) {
      if (e?.name !== 'RenderingCancelledException') {
        console.error('Render failed:', e)
        setLoading(false)
      }
    }
  }, [currentPage, renderScale, highlightText])

  // ── Text highlighting ──────────────────────────────────────────────────────

  async function highlightTextOnPage(pg: any, viewport: any, searchText: string) {
    if (!overlayRef.current) return
    try {
      const textContent = await pg.getTextContent()
      const candidates  = [searchText, searchText.slice(0, 80), searchText.slice(0, 40)]
      for (const c of candidates) {
        if (c.length < 10) continue
        const found = tryHighlight(viewport, textContent, c)
        if (found) break
      }
    } catch (e) { console.error('Highlight failed:', e) }
  }

  function tryHighlight(viewport: any, textContent: any, searchText: string): boolean {
    if (!overlayRef.current) return false
    const items     = textContent.items
    const searchLow = searchText.toLowerCase().replace(/\s+/g, ' ').trim()

    let combined = ''
    const positions: Array<{ start: number; item: any }> = []
    for (const item of items) {
      positions.push({ start: combined.length, item })
      combined += item.str + ' '
    }

    const matchIdx = combined.toLowerCase().indexOf(searchLow)
    if (matchIdx === -1) return false
    const matchEnd = matchIdx + searchLow.length

    const matched = positions.filter(({ start, item }) => {
      const end = start + item.str.length
      return start < matchEnd && end > matchIdx
    })
    if (matched.length === 0) return false

    matched.forEach(({ item }) => {
      if (!item.transform) return
      const [, , , , tx, ty] = item.transform
      const pt   = viewport.convertToViewportPoint(tx, ty)
      const itemW = (item.width  ?? item.str.length * 6) * renderScale
      const itemH = (item.height ?? 12) * renderScale

      const hl         = document.createElement('div')
      hl.className     = 'text-highlight'
      hl.style.cssText = `
        position: absolute;
        left: ${pt[0]}px;
        top: ${pt[1] - itemH}px;
        width: ${itemW}px;
        height: ${itemH + 4}px;
        background: rgba(253,224,71,0.45);
        border-bottom: 2px solid #ca8a04;
        border-radius: 2px;
        pointer-events: none;
        mix-blend-mode: multiply;
        animation: hl-in 0.3s ease;
      `
      overlayRef.current!.appendChild(hl)
    })

    // Scroll to first highlighted item
    if (matched[0]?.item?.transform && scrollRef.current) {
      const [, , , , tx, ty] = matched[0].item.transform
      const pt = viewport.convertToViewportPoint(tx, ty)
      scrollRef.current.scrollTop = Math.max(0, pt[1] - 120)
    }

    return true
  }

  // ── Wheel zoom — CSS transform, debounce full re-render ───────────────────

  function handleWheel(e: React.WheelEvent<HTMLDivElement>) {
    if (!(e.ctrlKey || e.metaKey)) return
    e.preventDefault()

    const delta  = e.deltaY > 0 ? -0.05 : 0.05
    const newCss = Math.round(
      Math.min(2.5, Math.max(0.3, cssScale + delta)) * 100
    ) / 100
    setCssScale(newCss)

    // After 600ms of no zooming, bake CSS scale into renderScale and reset
    clearTimeout(zoomTimerRef.current)
    zoomTimerRef.current = setTimeout(() => {
      const baked = Math.round(
        Math.min(3, Math.max(0.5, renderScale * newCss)) * 10
      ) / 10
      setCssScale(1)
      setRenderScale(baked)
    }, 600)
  }

  // Reset zoom
  function resetZoom() {
    clearTimeout(zoomTimerRef.current)
    setCssScale(1)
    setRenderScale(DEFAULT_SCALE)
  }

  // ── Post-it interactions ───────────────────────────────────────────────────

  function handleCanvasClick(e: React.MouseEvent<HTMLDivElement>) {
    if (!addingNote || !canvasRef.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const x    = (e.clientX - rect.left)  / rect.width
    const y    = (e.clientY - rect.top)   / rect.height
    setPendingNote({ x, y })
    setAddingNote(false)
  }

  async function saveNote() {
    if (!pendingNote || !noteText.trim()) return
    const ann = await saveAnnotation(caseId, {
      page:        currentPage,
      pdf:         pdfFile,
      note:        noteText.trim(),
      position:    pendingNote,
      anchor_text: highlightText || '',
      color:       noteColor,
    })
    if (ann) setAnnotations(prev => [...prev, ann])
    setPendingNote(null)
    setNoteText('')
    setNoteColor(NOTE_COLORS[0])
  }

  async function removeAnnotation(id: string) {
    await deleteAnnotation(caseId, id)
    setAnnotations(prev => prev.filter(a => a.id !== id))
    if (activeNote === id) setActiveNote(null)
  }

  const pageAnnotations = annotations.filter(
    a => a.page === currentPage && a.pdf === pdfFile
  )

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div
      className="flex-shrink-0 flex flex-col overflow-hidden"
      style={{ width, background: 'var(--surface)', borderLeft: '1px solid var(--border)' }}
    >
      <style>{`
        @keyframes hl-in {
          from { opacity: 0; transform: scaleY(0.5); }
          to   { opacity: 1; transform: scaleY(1); }
        }
        @keyframes note-pop {
          from { opacity: 0; transform: scale(0.7) rotate(-3deg); }
          to   { opacity: 1; transform: scale(1) rotate(-1.5deg); }
        }
        .postit { animation: note-pop 0.2s cubic-bezier(0.34,1.56,0.64,1); }
      `}</style>

      {/* ── Toolbar ── */}
      <div
        className="flex items-center justify-between px-3 flex-shrink-0"
        style={{ borderBottom: '1px solid var(--border)', height: 44, gap: 6 }}
      >
        {/* Page nav */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
            disabled={currentPage <= 1}
            className="w-6 h-6 flex items-center justify-center rounded"
            style={{ color: 'var(--muted)', opacity: currentPage <= 1 ? 0.3 : 1 }}
          >
            <ChevronLeft size={13} />
          </button>
          <span
            className="text-xs px-1"
            style={{ color: 'var(--muted2)', minWidth: 64, textAlign: 'center' }}
          >
            p.{currentPage}{totalPages > 0 ? ` / ${totalPages}` : ''}
          </span>
          <button
            onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
            disabled={currentPage >= totalPages}
            className="w-6 h-6 flex items-center justify-center rounded"
            style={{ color: 'var(--muted)', opacity: currentPage >= totalPages ? 0.3 : 1 }}
          >
            <ChevronRight size={13} />
          </button>
        </div>

        {/* Zoom indicator + reset */}
        <div className="flex items-center gap-1.5">
          <span
            className="text-xs px-1.5 py-0.5 rounded"
            style={{
              color: 'var(--muted)',
              background: 'var(--bg)',
              border: '1px solid var(--border)',
              minWidth: 42,
              textAlign: 'center',
            }}
          >
            {displayZoom}%
          </span>
          <button
            onClick={resetZoom}
            className="text-xs px-1.5 py-0.5 rounded"
            style={{ color: 'var(--muted)', border: '1px solid var(--border)' }}
            title="Reset zoom — pinch trackpad or Ctrl+scroll to zoom"
          >
            ↺
          </button>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => { setAddingNote(a => !a); setShowIndex(false) }}
            className="flex items-center gap-1 px-2 py-1 rounded text-xs"
            style={{
              background: addingNote ? '#1c1003' : 'var(--surface2)',
              border:     `1px solid ${addingNote ? '#78350f' : 'var(--border)'}`,
              color:      addingNote ? '#f59e0b' : 'var(--muted)',
            }}
            title="Add sticky note"
          >
            <StickyNote size={11} />
            {addingNote ? 'Click page' : 'Note'}
          </button>

          <button
            onClick={() => { setShowIndex(s => !s); setAddingNote(false) }}
            className="flex items-center gap-1 px-2 py-1 rounded text-xs"
            style={{
              background: showIndex ? '#1a1f35' : 'var(--surface2)',
              border:     `1px solid ${showIndex ? 'var(--accent)' : 'var(--border)'}`,
              color:      showIndex ? 'var(--accent2)' : 'var(--muted)',
            }}
            title="All notes"
          >
            {annotations.filter(a => a.pdf === pdfFile).length > 0 && (
              <span className="font-bold mr-0.5">
                {annotations.filter(a => a.pdf === pdfFile).length}
              </span>
            )}
            Index
          </button>

          <button
            onClick={onClose}
            className="w-7 h-7 flex items-center justify-center rounded"
            style={{ color: 'var(--muted)', border: '1px solid var(--border)' }}
          >
            <X size={13} />
          </button>
        </div>
      </div>

      {/* ── Notes index ── */}
      {showIndex && (
        <div
          className="flex-shrink-0 overflow-y-auto"
          style={{
            maxHeight: 260,
            borderBottom: '1px solid var(--border)',
            background: 'var(--bg)',
          }}
        >
          <div
            className="px-3 py-2 text-xs uppercase tracking-widest"
            style={{ color: 'var(--muted)', borderBottom: '1px solid var(--border)' }}
          >
            Notes — {pdfFile}
          </div>
          {annotations.filter(a => a.pdf === pdfFile).length === 0 ? (
            <div className="px-3 py-4 text-xs text-center" style={{ color: 'var(--muted)' }}>
              No notes yet
            </div>
          ) : (
            annotations
              .filter(a => a.pdf === pdfFile)
              .sort((a, b) => a.page - b.page)
              .map(ann => (
                <div
                  key={ann.id}
                  className="flex items-start gap-2 px-3 py-2 cursor-pointer"
                  style={{ borderBottom: '1px solid var(--border)', background: 'transparent' }}
                  onMouseEnter={e => (e.currentTarget.style.background = 'var(--surface)')}
                  onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                  onClick={() => { setCurrentPage(ann.page); setShowIndex(false) }}
                >
                  <div
                    className="w-3 h-3 rounded-sm flex-shrink-0 mt-0.5"
                    style={{ background: ann.color }}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="text-xs font-medium mb-0.5" style={{ color: 'var(--accent2)' }}>
                      p.{ann.page}
                    </div>
                    <div className="text-xs truncate" style={{ color: 'var(--muted2)' }}>
                      {ann.note}
                    </div>
                  </div>
                  <button
                    onClick={e => { e.stopPropagation(); removeAnnotation(ann.id) }}
                    className="w-5 h-5 flex items-center justify-center rounded flex-shrink-0"
                    style={{ color: 'var(--muted)' }}
                  >
                    <Trash2 size={10} />
                  </button>
                </div>
              ))
          )}
        </div>
      )}

      {/* ── Canvas scroll area ── */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-auto"
        style={{
          background: '#1a1a1a',
          cursor: addingNote ? 'crosshair' : 'default',
        }}
        onWheel={handleWheel}
        onClick={handleCanvasClick}
      >
        {loading && (
          <div
            className="absolute inset-0 flex items-center justify-center z-10"
            style={{ background: 'rgba(0,0,0,0.5)', pointerEvents: 'none' }}
          >
            <div className="text-xs" style={{ color: 'var(--muted)' }}>
              {(window as any).pdfjsLib ? 'Rendering...' : 'Loading PDF.js...'}
            </div>
          </div>
        )}

        {/* Canvas wrapper — CSS scale applied here for smooth live zoom */}
        <div
          ref={canvasWrapRef}
          style={{
            position:        'relative',
            display:         'inline-block',
            minWidth:        '100%',
            transformOrigin: 'top left',
            transform:       `scale(${cssScale})`,
            // No transition — instant feel, matching native browser zoom
          }}
        >
          <canvas ref={canvasRef} style={{ display: 'block' }} />

          {/* Highlight + post-it overlay — inside wrapper so it scales too */}
          <div
            ref={overlayRef}
            style={{
              position:      'absolute',
              top: 0, left: 0,
              width:         '100%',
              height:        '100%',
              pointerEvents: 'none',
            }}
          >
            {pageAnnotations.map(ann => (
              <div
                key={ann.id}
                className="postit"
                style={{
                  position:     'absolute',
                  left:         `${ann.position.x * 100}%`,
                  top:          `${ann.position.y * 100}%`,
                  transform:    'rotate(-1.5deg)',
                  width:        160,
                  background:   ann.color,
                  boxShadow:    '2px 3px 8px rgba(0,0,0,0.4)',
                  borderRadius: '2px 8px 8px 2px',
                  padding:      '8px 10px',
                  pointerEvents: 'all',
                  cursor:       'pointer',
                  zIndex:       activeNote === ann.id ? 20 : 10,
                }}
                onClick={e => {
                  e.stopPropagation()
                  setActiveNote(activeNote === ann.id ? null : ann.id)
                }}
              >
                <div
                  style={{
                    fontSize:   11,
                    color:      '#1a1a1a',
                    lineHeight: 1.4,
                    fontFamily: 'monospace',
                    wordBreak:  'break-word',
                  }}
                >
                  {ann.note}
                </div>
                {activeNote === ann.id && (
                  <button
                    onClick={e => { e.stopPropagation(); removeAnnotation(ann.id) }}
                    style={{
                      position:     'absolute',
                      top: -8, right: -8,
                      width:        18,
                      height:       18,
                      background:   '#ef4444',
                      border:       'none',
                      borderRadius: '50%',
                      color:        'white',
                      fontSize:     10,
                      cursor:       'pointer',
                      display:      'flex',
                      alignItems:   'center',
                      justifyContent: 'center',
                    }}
                  >
                    ×
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Pending note form ── */}
      {pendingNote && (
        <div
          className="flex-shrink-0 p-3 flex flex-col gap-2"
          style={{ borderTop: '1px solid var(--border)', background: 'var(--surface)' }}
        >
          <div className="text-xs font-medium" style={{ color: 'var(--muted2)' }}>
            New note — p.{currentPage}
          </div>
          <textarea
            autoFocus
            value={noteText}
            onChange={e => setNoteText(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); saveNote() }
              if (e.key === 'Escape') { setPendingNote(null); setNoteText('') }
            }}
            placeholder="Type your note... (Enter to save, Esc to cancel)"
            rows={3}
            className="w-full rounded-md px-2.5 py-1.5 text-xs outline-none resize-none"
            style={{
              background:  'var(--bg)',
              border:      '1px solid var(--border2)',
              color:       'var(--text)',
              fontFamily:  'monospace',
            }}
          />
          <div className="flex items-center gap-2">
            <span className="text-xs" style={{ color: 'var(--muted)' }}>Color:</span>
            {NOTE_COLORS.map(c => (
              <button
                key={c}
                onClick={() => setNoteColor(c)}
                style={{
                  width:        16,
                  height:       16,
                  background:   c,
                  borderRadius: 3,
                  border:       noteColor === c ? '2px solid white' : '1px solid transparent',
                  cursor:       'pointer',
                }}
              />
            ))}
            <div className="ml-auto flex gap-2">
              <button
                onClick={() => { setPendingNote(null); setNoteText('') }}
                className="px-2 py-1 rounded text-xs"
                style={{ color: 'var(--muted)', border: '1px solid var(--border)' }}
              >
                Cancel
              </button>
              <button
                onClick={saveNote}
                disabled={!noteText.trim()}
                className="px-2 py-1 rounded text-xs font-bold"
                style={{
                  background: noteText.trim() ? 'var(--accent)' : 'var(--border2)',
                  color:      'white',
                }}
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}