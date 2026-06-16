"""
retrieval/agent.py
-------------------
LLM Reasoning Agent — the final stage of the retrieval pipeline.

Two modes:
    Normal mode  — standard query answering with citations
    Deep Research — all nodes, all relationships, all passages fed to GPT-4o
                    for a comprehensive structured report

Deep Research output structure:
    1. Executive Summary
    2. Key Parties and Their Roles
    3. Sequence of Events (chronological)
    4. Legal Issues and Arguments
    5. Court Orders and Directions
    6. Financial Details
    7. Key Relationships
    8. Conclusion and Legal Standing
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field

import query
from retrieval import graph_retriever

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from config import settings
from retrieval.fusion import FusedContext, ConfidenceLevel

logger = logging.getLogger(__name__)

MAX_HOPS = 3
SUFFICIENCY_CONFIDENCE_THRESHOLD = 0.80
MIN_TYPED_TRIPLE_RATIO = 0.10
FAST_PATH_HIGH_CONFIDENCE_MIN = 5


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class SufficiencyCheck:
    is_sufficient: bool
    missing_information: list[str]
    expand_entities: list[str]
    confidence: float
    reason: str = ""


@dataclass
class FinalAnswer:
    query: str
    answer: str
    citations: list[dict]
    confidence: float
    hops_taken: int
    answer_type: str
    is_deep_research: bool = False


# ── Agent ──────────────────────────────────────────────────────────────────────

class ReasoningAgent:

    SUFFICIENCY_PROMPT = """You are a legal document reasoning assistant.

You have been given a query and retrieved context from a legal case document.
Determine if the context is sufficient to answer the query accurately.

The context includes graph relationships annotated as:
  [typed]  — semantic relationship with actual meaning, usable as evidence
  [weak]   — co-occurrence only (two entities appeared on the same page),
             NOT evidence of any relationship between them

Query: {query}

Retrieved Context:
{context}

Evaluation rules:
- Weak [co-occurrence] relationships alone are NEVER sufficient evidence.
- You need either: (a) at least one [typed] graph relationship relevant to
  the query, OR (b) direct document passages that explicitly address the query.
- If only weak relationships exist and document passages are vague,
  mark as NOT sufficient and request expansion on specific missing entities.

Respond in this exact JSON format:
{{
  "is_sufficient": true | false,
  "confidence": 0.0-1.0,
  "missing_information": ["what specific information is missing"],
  "expand_entities": ["entity names to search for more context"],
  "reasoning": "brief explanation referencing typed vs weak evidence"
}}"""

    ANSWER_PROMPT = """You are an expert legal analyst answering questions about a legal document.

Answer the query based ONLY on the provided context. Never fabricate facts.

Query: {query}

Context:
{context}

Rules:
1. Cite every factual claim — use [Graph: Entity→RELATIONSHIP→Entity] or [Document: Page N].
2. [weak] and [RELATED_TO] relationships are NOT evidence — never use them to support a claim.
3. Only [typed] relationships carry semantic meaning and can be cited as evidence.
4. If information is missing from context, say so specifically.
5. Format your response to match the query type:
   - Factual question → direct answer in 2-3 sentences
   - List/relationship question → bullet points, one item per line
   - Analysis/argument question → structured paragraphs with clear headings
   - Use **bold** for entity names and key terms within your answer

Respond in this exact JSON format:
{{
  "answer": "your answer here",
  "citations": [
    {{"text": "key claim", "source": "Graph|Document", "page": 0, "detail": "relationship or section"}}
  ],
  "confidence": 0.0-1.0,
  "answer_type": "DIRECT|INFERRED|PARTIAL|COMPLEX",
  "reasoning": "brief explanation of evidence quality"
}}"""

    DEEP_RESEARCH_PROMPT = """You are a senior legal analyst preparing a comprehensive research report on a legal case.

You have been given ALL available information extracted from the case document:
- Every relevant entity and its relationships from the knowledge graph
- Every relevant document passage with page citations
- All typed (semantic) relationships between parties, courts, and events
- Graph-augmented evidence grounded in specific document pages

Your task is to produce a COMPREHENSIVE, WELL-STRUCTURED research report covering
everything that can be known about this case from the provided information.

CRITICAL: Do NOT invent disputes, allegations, or legal proceedings that are not
explicitly stated in the provided context. If the document is a contract with no
active dispute, describe it as a governing agreement — not a dispute. Do not assume
non-compliance or breach unless the context explicitly states it. Report only what
is present in the evidence, not what clauses say could happen hypothetically.

Query / Research Topic: {query}

Complete Case Context:
{context}

Produce a detailed research report with the following structure.
Every factual claim MUST include a citation [Graph: ...] or [Document: Page N].
Be exhaustive — include every relevant detail, number, date, and legal reference.

IMPORTANT: Adapt the section content to the TYPE of document being analysed.
- For litigation: focus on court orders, legal challenges, judicial history
- For contracts: focus on obligations, rights, conditions, breach, remedies
- For corporate documents: focus on structure, governance, shareholder rights
- For banking/finance: focus on loan terms, security, default, enforcement
- If a section is not applicable to this document type, state "Not applicable to this document type" briefly and move on.

Respond in this exact JSON format:
{{
  "answer": "# Deep Research Report\\n\\n## 1. Executive Summary\\n[2-3 paragraph overview of the document — its TYPE (contract/petition/agreement), its PURPOSE, key parties, and ACTUAL current status based only on what is explicitly stated. Do NOT infer disputes or allegations. If this is a contract with no active dispute, describe it as a governing agreement.]\\n\\n## 2. Parties and Their Roles\\n[Every party identified — with their legal role and relationship to the document. For contracts: buyer/seller/licensor/licensee. For litigation: petitioner/respondent/court. For corporate: directors/shareholders/company.] \\n\\n## 3. Key Facts and Timeline\\n[Every significant event, obligation trigger, or dated entry in chronological order. For contracts: execution date, milestones, notice periods. For litigation: filings, orders, hearings. Each entry: Date/Event → Detail → Citation]\\n\\n## 4. Central Issues and Subject Matter\\n[The core subject matter — what is being agreed, governed, challenged, or resolved. For contracts: key obligations, conditions, dispute clauses. For litigation: legal questions, grounds of challenge, relevant statutes.]\\n\\n## 5. Obligations, Rights and Remedies\\n[What each party is required to do, entitled to receive, and what happens on breach or non-performance. For contracts: payment terms, delivery obligations, termination rights. For litigation: court directions, compliance obligations.]\\n\\n## 6. Financial and Commercial Terms\\n[Every financial figure — amounts, rates, penalties, payment schedules, consideration — with exact figures and source pages. If no financial terms exist, state so.]\\n\\n## 7. Key Relationships Between Entities\\n[The most important typed relationships from the knowledge graph that define the structure of this document]\\n\\n## 8. Risk Assessment and Open Issues\\n[What is clearly established vs ambiguous, gaps in the record, unresolved obligations, areas requiring clarification]\\n\\n## 9. Conclusion\\n[Summary of the document's legal effect, what obligations or rights are clearly established, what remains open or contested],
  "citations": [
    {{"text": "key fact", "source": "Graph|Document", "page": 0, "detail": "source detail"}}
  ],
  "confidence": 0.0-1.0,
  "answer_type": "DEEP_RESEARCH",
  "reasoning": "evidence quality assessment"
}}"""

    def __init__(self):
        self._client = OpenAI(api_key=settings.openai_api_key)

    # ── Normal query answer ────────────────────────────────────────────────────

    def answer(
        self,
        query: str,
        fused_context: FusedContext,
        graph_retriever=None,
        tree_retriever=None,
        fusion_engine=None,
        query_type: str = "FACT",
    ) -> FinalAnswer:
        """Standard query answering with confidence routing and multi-hop."""
        current_context = fused_context
        hops = 0

        # Fast path — only for simple FACT queries with high confidence
        # COMPLEX and RELATIONSHIP queries always go through sufficiency check
        if (
            query_type == "FACT"
            and current_context.high_confidence_count >= FAST_PATH_HIGH_CONFIDENCE_MIN
        ):
            logger.info(
                f"Fast path: FACT query with {current_context.high_confidence_count} "
                f"HIGH confidence results — skipping sufficiency check"
            )
            return self._generate_answer(query, current_context, hops, query_type)

        # Standard path — sufficiency check + multi-hop
        while hops < MAX_HOPS:
            check = self._check_sufficiency(query, current_context)
            logger.info(
                f"Hop {hops}: sufficient={check.is_sufficient}, "
                f"confidence={check.confidence:.2f} | {check.reason}"
            )

            if check.is_sufficient:
                break

            if (
                check.expand_entities
                and graph_retriever
                and tree_retriever
                and fusion_engine
            ):
                logger.info(f"Multi-hop {hops+1}: expanding {check.expand_entities}")
                current_context = self._expand_context(
                    query, current_context, check.expand_entities,
                    graph_retriever, tree_retriever, fusion_engine,
                    query_type=query_type,
                )
                hops += 1
            else:
                break

        return self._generate_answer(query, current_context, hops, query_type)

    # ── Deep Research mode ─────────────────────────────────────────────────────

    def deep_research(
        self,
        query: str,
        fused_context: FusedContext,
        graph_retriever=None,
        tree_retriever=None,
        fusion_engine=None,
    ) -> FinalAnswer:
        """
        Deep Research mode — exhaustive analysis using ALL available context.

        Differences from normal mode:
        - No sufficiency check — always uses everything available
        - Multi-hop always runs to maximum depth to gather all context
        - Full context window used (no truncation)
        - GPT-4o with 4000 token output for comprehensive report
        - Structured report format with 9 sections
        - No fast path — deep research is always thorough
        """
        logger.info("Deep Research mode activated — gathering all available context...")

        current_context = fused_context
        actual_hops = 0

        # Always do maximum hops to gather as much context as possible
        if graph_retriever and tree_retriever and fusion_engine:
            for hop in range(MAX_HOPS):
                # Use all high-degree entities as expansion targets
                expand_entities = [
                    r.entity_name for r in current_context.results
                    if r.confidence_level in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM)
                ][:6]

                if expand_entities:
                    logger.info(f"Deep Research hop {hop+1}: expanding {expand_entities}")
                    current_context = self._expand_context(
                        query, current_context, expand_entities,
                        graph_retriever, tree_retriever, fusion_engine,
                        query_type="DEEP_RESEARCH",
                    )
                    actual_hops += 1
                else:
                    break

        return self._generate_deep_research(query, current_context, actual_hops)

    # ── Sufficiency check ──────────────────────────────────────────────────────

    def _check_sufficiency(
        self, query: str, context: FusedContext
    ) -> SufficiencyCheck:
        total_triples = context.typed_triple_count + context.weak_triple_count

        # Stage 1 — graph quality gate
        if total_triples > 0:
            total_typed = context.typed_triple_count + len(context.gar_triples)
            typed_ratio = total_typed / total_triples
            if typed_ratio < MIN_TYPED_TRIPLE_RATIO:
                expand = [
                    r.entity_name for r in context.results
                    if r.confidence_level == ConfidenceLevel.HIGH
                ][:4]
                reason = (
                    f"Graph quality gate failed: "
                    f"{context.typed_triple_count} typed / "
                    f"{total_triples} total triples "
                    f"({typed_ratio:.0%} < {MIN_TYPED_TRIPLE_RATIO:.0%} threshold)"
                )
                return SufficiencyCheck(
                    is_sufficient=False,
                    missing_information=["Typed semantic relationships between key entities"],
                    expand_entities=expand,
                    confidence=0.0,
                    reason=reason,
                )

        # Stage 2 — LLM check
        try:
            context_str = context.to_prompt_context()
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": self.SUFFICIENCY_PROMPT.format(
                        query=query,
                        context=context_str[:8000],
                    )
                }],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=400,
            )

            data           = json.loads(response.choices[0].message.content)
            llm_sufficient = data.get("is_sufficient", True)
            llm_confidence = float(data.get("confidence", 0.7))
            is_sufficient  = (
                llm_sufficient and
                llm_confidence >= SUFFICIENCY_CONFIDENCE_THRESHOLD
            )
            reason = data.get("reasoning", "")
            if llm_sufficient and not is_sufficient:
                reason += (
                    f" [overridden: confidence {llm_confidence:.2f} < "
                    f"threshold {SUFFICIENCY_CONFIDENCE_THRESHOLD}]"
                )

            return SufficiencyCheck(
                is_sufficient=is_sufficient,
                missing_information=data.get("missing_information", []),
                expand_entities=data.get("expand_entities", []),
                confidence=llm_confidence,
                reason=reason,
            )

        except Exception as e:
            logger.error(f"Sufficiency check failed: {e}")
            return SufficiencyCheck(
                is_sufficient=True,
                missing_information=[],
                expand_entities=[],
                confidence=0.5,
                reason=f"Check failed ({e}), defaulting to sufficient",
            )

    # ── Context expansion ──────────────────────────────────────────────────────

    def _expand_context(
        self,
        query: str,
        current_context: FusedContext,
        expand_entities: list[str],
        graph_retriever,
        tree_retriever,
        fusion_engine,
        query_type: str = "COMPLEX",
    ) -> FusedContext:
        from retrieval.graph_retriever import TOP_K_BY_QUERY_TYPE, DEFAULT_TOP_K
        expanded_query = f"{query} {' '.join(expand_entities)}"
        graph_result   = graph_retriever.search(expanded_query, query_type=query_type)
        passages_k     = TOP_K_BY_QUERY_TYPE.get(query_type, DEFAULT_TOP_K)["passages"]
        tree_results   = tree_retriever.search(expanded_query, top_k=passages_k)
        new_context    = fusion_engine.fuse(query, graph_result, tree_results, graph_retriever)
        return fusion_engine.merge(current_context, new_context)

    # ── Normal answer generation ───────────────────────────────────────────────

    def _generate_answer(
        self,
        query: str,
        context: FusedContext,
        hops_taken: int,
        query_type: str = "FACT",
    ) -> FinalAnswer:
        try:
            context_str = context.to_prompt_context()

            # Adjust depth based on query type
            if query_type == "COMPLEX":
                max_tokens = 3000
                type_instruction = "\nThis is a COMPLEX analytical query. Produce a structured response with clear headings and sub-points. Go deep — the lawyer needs a thorough understanding, not a summary."
            elif query_type == "RELATIONSHIP":
                max_tokens = 2000
                type_instruction = "\nThis is a RELATIONSHIP query. List every relevant relationship as a bullet point. One relationship per line. Be exhaustive."
            else:
                max_tokens = 1500
                type_instruction = "\nThis is a FACT query. Answer directly and concisely in 2-3 sentences."

            response = self._client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": self.ANSWER_PROMPT.format(
                        query=query,
                        context=context_str[:16000],
                    ) + type_instruction
                }],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=max_tokens,
            )

            data = json.loads(response.choices[0].message.content)
            return FinalAnswer(
                query=query,
                answer=data.get("answer", "Unable to generate answer."),
                citations=data.get("citations", []),
                confidence=float(data.get("confidence", 0.7)),
                hops_taken=hops_taken,
                answer_type=data.get("answer_type", "DIRECT"),
                is_deep_research=False,
            )

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return FinalAnswer(
                query=query,
                answer=f"Answer generation failed: {e}",
                citations=[],
                confidence=0.0,
                hops_taken=hops_taken,
                answer_type="PARTIAL",
            )

    # ── Deep research answer generation ───────────────────────────────────────

    def _generate_deep_research(
        self,
        query: str,
        context: FusedContext,
        actual_hops: int = 0,
    ) -> FinalAnswer:
        """
        Generate comprehensive research report.
        Uses full context window and 4000 token output.
        """
        try:
            context_str = context.to_prompt_context()

            logger.info(
                f"Deep Research: feeding {len(context_str):,} chars to GPT-4o "
                f"({context.high_confidence_count} HIGH, "
                f"{context.medium_confidence_count} MEDIUM confidence results, "
                f"{context.typed_triple_count} typed triples, "
                f"{len(context.gar_triples)} GAR triples)"
            )

            response = self._client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": self.DEEP_RESEARCH_PROMPT.format(
                        query=query,
                        # Use full context — no truncation for deep research
                        context=context_str[:32000],
                    )
                }],
                response_format={"type": "json_object"},
                temperature=0.1,
                # 4x the normal token limit for comprehensive report
                max_tokens=4000,
            )

            data = json.loads(response.choices[0].message.content)
            return FinalAnswer(
                query=query,
                answer=data.get("answer", "Unable to generate research report."),
                citations=data.get("citations", []),
                confidence=float(data.get("confidence", 0.9)),
                hops_taken=actual_hops,
                answer_type="DEEP_RESEARCH",
                is_deep_research=True,
            )

        except Exception as e:
            logger.error(f"Deep research generation failed: {e}")
            return FinalAnswer(
                query=query,
                answer=f"Deep research failed: {e}",
                citations=[],
                confidence=0.0,
                hops_taken=0,
                answer_type="PARTIAL",
                is_deep_research=True,
            )