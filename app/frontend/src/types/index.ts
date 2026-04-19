export interface Case {
  case_id: string
  case_name: string
  pdf_filename: string
  status: 'processing' | 'review' | 'ready'
  created_at: string
  pages: number | null
  kge_status?: 'not_started' | 'training' | 'ready' | 'failed'
  has_tree?: boolean
}

export interface Citation {
  text: string
  page: number
  pdf: string
  source?: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  confidence?: number
  answer_type?: 'DIRECT' | 'PARTIAL' | 'INFERRED' | 'DEEP_RESEARCH'
  hops?: number
  timestamp: string
}

export interface Candidate {
  canonical_name: string
  text: string
  schema_type: string
  source_pdf: string
  source_page: number
  source_line?: number
  context?: string
  confidence: number
  llm_vote?: 'YES' | 'NO' | 'UNSURE'
  llm_merge_confidence?: number
  llm_reason?: string
}

export interface Group {
  group_id: string
  schema_type: string
  canonical_name: string
  llm_model_used?: string
  is_complex?: boolean
  candidates: Candidate[]
  human_decision?: string
}

export interface ResolutionState {
  created_at: string
  summary: {
    total_groups: number
    auto_merge_count: number
    needs_review_count: number
    auto_keep_count: number
  }
  auto_merge: Group[]
  needs_review: Group[]
  auto_keep: Group[]
}

export interface SidebarItem {
  idx: number
  label: string
  schema_type: string
  status: 'PENDING' | 'MERGE' | 'KEEP' | 'SKIP'
}

export interface StagedDecision {
  group_id: string
  decision: 'MERGE' | 'KEEP' | 'SKIP'
  buckets: Record<string, string[]>
  staged_at: string
}