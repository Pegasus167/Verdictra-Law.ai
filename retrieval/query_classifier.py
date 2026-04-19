"""
retrieval/query_classifier.py
------------------------------
Classifies incoming queries to determine which retrieval path(s) to use.

Query types:
    RELATIONSHIP — asks purely about connections between named entities
                   → Graph path only
    FACT         — asks for a single self-contained value that exists
                   verbatim in the document (number, date, identifier)
                   → Tree path only
    COMPLEX      — everything else: legal events, orders, judgments,
                   sequences of events, justifications, analysis,
                   ownership chains, party roles + evidence
                   → Both paths fire

Key design decision:
    The old prompt listed "What did the Supreme Court order on X date?"
    as a FACT example. That was wrong. Court orders involve entities
    (Supreme Court, Bombay HC, petitioner, bank), relationships
    (set aside, declared, directed, ordered), AND documentary evidence
    (exact wording of the order). That is COMPLEX by definition.

    FACT is reserved for queries whose answer is a single atomic value
    that exists verbatim in the document — an amount, a date, a
    certificate number, a plot number. No entity traversal needed.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from config import settings

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    RELATIONSHIP = "RELATIONSHIP"   # Graph path only
    FACT = "FACT"                   # Tree path only
    COMPLEX = "COMPLEX"             # Both paths


@dataclass
class ClassificationResult:
    query: str
    query_type: QueryType
    confidence: float
    reasoning: str
    key_entities: list[str]
    key_terms: list[str]


CLASSIFICATION_PROMPT = """You are an expert legal document query analyst.

Classify the following query into ONE of three retrieval types.

─────────────────────────────────────────────────────────────────
FACT
─────────────────────────────────────────────────────────────────
Use ONLY when the answer is a single atomic value that exists
verbatim in the document — a number, amount, date, identifier,
certificate number, or plot number. No entity relationships needed.

FACT examples:
- "What is the total demand amount in the January 2024 notice?"
- "What are the plot numbers involved in the case?"
- "What is the CIN number of CELIR LLP?"
- "On what date was the Drainage Completion Certificate obtained?"

─────────────────────────────────────────────────────────────────
RELATIONSHIP
─────────────────────────────────────────────────────────────────
Use ONLY when the query is purely about how named entities connect
to each other. No documentary evidence needed — just the graph.

RELATIONSHIP examples:
- "Who are the directors of CELIR LLP?"
- "What is the legal relationship between Bafna Motors and Union Bank?"
- "Which court did CELIR approach first?"

─────────────────────────────────────────────────────────────────
COMPLEX  ← default for legal queries
─────────────────────────────────────────────────────────────────
Use for EVERYTHING ELSE. Legal queries almost always require both
relationship context (who did what to whom) AND documentary evidence
(what exactly was said, ordered, or decided). When in doubt use COMPLEX.

COMPLEX includes:
- Any court order, judgment, ruling, or direction
- Any question about what happened in a legal proceeding
- Questions about justification, legality, or rights
- Questions about sequences of events or chains of ownership
- Questions about what any party did or was required to do
- Questions involving "why", "how", "justify", "explain"
- Any question mentioning: court, order, ruled, held, directed,
  set aside, declared, petition, writ, appeal, judgment, scheme,
  undertaking, mortgage, auction, certificate, transfer, charge

COMPLEX examples:
- "What did the Supreme Court order on 21 September 2023?"
- "Is MIDC justified in demanding ULC charges from CELIR?"
- "What is the chain of ownership of the plots?"
- "What did Bafna Motors undertake and was it fulfilled?"
- "Explain the Amnesty Scheme and its provisions"
- "Who are all the parties and what are their roles?"
- "What was the sequence of events leading to the demand notice?"
- "What did the Bombay High Court hold on 17 August 2023?"

─────────────────────────────────────────────────────────────────

Respond in this exact JSON format:
{
  "query_type": "RELATIONSHIP" | "FACT" | "COMPLEX",
  "confidence": 0.0-1.0,
  "reasoning": "One sentence explanation",
  "key_entities": ["entity1", "entity2"],
  "key_terms": ["term1", "term2", "term3"]
}"""


class QueryClassifier:
    """
    Classifies queries to determine which retrieval path to use.

    Usage:
        classifier = QueryClassifier()
        result = classifier.classify("What did the Supreme Court order on 21 Sept 2023?")
        print(result.query_type)   # COMPLEX
    """

    def __init__(self):
        self._client = OpenAI(api_key=settings.openai_api_key)

    def classify(self, query: str) -> ClassificationResult:
        """Classify a query using GPT-4o-mini."""
        try:
            response = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": CLASSIFICATION_PROMPT},
                    {"role": "user", "content": f"Query: {query}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=300,
            )

            data = json.loads(response.choices[0].message.content)

            query_type = QueryType(data.get("query_type", "COMPLEX"))

            # Safety net: legal proceeding keywords always force COMPLEX
            # This catches any edge cases the LLM still misclassifies
            FORCE_COMPLEX = [
                "court", "order", "ordered", "ruled", "held", "directed",
                "judgment", "judgement", "petition", "writ", "appeal",
                "set aside", "declared", "upheld", "dismissed", "allowed",
                "scheme", "undertaking", "mortgage", "auction", "certificate",
                "transfer charge", "uLC", "amnesty", "allottee", "leaseholder",
            ]
            query_lower = query.lower()
            if (
                query_type == QueryType.FACT
                and any(kw in query_lower for kw in FORCE_COMPLEX)
            ):
                logger.info(
                    f"  Classifier override: FACT → COMPLEX "
                    f"(legal keyword detected in query)"
                )
                query_type = QueryType.COMPLEX

            return ClassificationResult(
                query=query,
                query_type=query_type,
                confidence=float(data.get("confidence", 0.7)),
                reasoning=data.get("reasoning", ""),
                key_entities=data.get("key_entities", []),
                key_terms=data.get("key_terms", []),
            )

        except Exception as e:
            logger.error(f"Classification failed: {e} — defaulting to COMPLEX")
            return ClassificationResult(
                query=query,
                query_type=QueryType.COMPLEX,
                confidence=0.5,
                reasoning=f"Classification failed, defaulting to COMPLEX: {e}",
                key_entities=[],
                key_terms=query.split()[:5],
            )