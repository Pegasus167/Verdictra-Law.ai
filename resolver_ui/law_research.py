"""
resolver_ui/law_research.py
----------------------------
Law Research mode — Indian law Q&A via GPT-4o-mini.
Sessions stored in SQLite alongside user accounts.

Endpoints:
  GET    /law-research/sessions              — list user's sessions
  POST   /law-research/sessions              — create new session
  GET    /law-research/sessions/{id}         — get session with messages
  PUT    /law-research/sessions/{id}/name    — rename session
  DELETE /law-research/sessions/{id}         — delete session
  POST   /law-research/sessions/{id}/ask     — SSE streaming answer
"""

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from resolver_ui.auth import require_auth, UserInfo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/law-research", tags=["law-research"])

DB_PATH = os.environ.get("USER_DB_PATH", "/app/data/users.db")

# ── Indian law system prompt ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert on Indian law with deep knowledge of:

**Statutes and Codes**
- Constitution of India (all Articles, Parts, Schedules, and Amendments)
- Indian Penal Code, 1860 (IPC) and Bharatiya Nyaya Sanhita, 2023 (BNS)
- Code of Criminal Procedure, 1973 (CrPC) and Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS)
- Code of Civil Procedure, 1908 (CPC)
- Indian Evidence Act, 1872 and Bharatiya Sakshya Adhiniyam, 2023 (BSA)
- Contract Act, 1872; Transfer of Property Act, 1882; Specific Relief Act, 1963
- Companies Act, 2013; Insolvency and Bankruptcy Code, 2016 (IBC)
- SARFAESI Act, 2002; Recovery of Debts and Bankruptcy Act, 1993
- Negotiable Instruments Act, 1881; Income Tax Act, 1961; GST legislation
- All major central and state legislation

**Landmark Judgments**
- Supreme Court of India judgments up to early 2024
- Important High Court judgments across jurisdictions
- Constitutional bench decisions and five-judge bench rulings

**How to answer:**
- Lead with the specific provision (Section X, Article Y) as a heading
- State the law precisely — what it says, what it requires, what it prohibits
- For landmark cases: cite full case name, year, court, and key holding
- Where a provision has been amended or replaced (e.g., IPC → BNS), note both
- Format clearly with ## headings for different aspects of a complex answer
- Use **bold** for key legal terms and provision numbers

**Important:**
- Knowledge cutoff: early 2024. For very recent judgments or amendments after this date, advise the user to verify independently
- You provide legal information, not legal advice. For specific strategy on a pending matter, the user should consult a qualified advocate
- If uncertain about a specific provision or judgment, say so clearly — never fabricate citations
- Keep answers precise and practical — lawyers need to work with this information"""


# ── SQLite helpers ──────────────────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_law_research_tables() -> None:
    """Create law_research_sessions table. Safe to call on every startup."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS law_research_sessions (
                id         TEXT PRIMARY KEY,
                user_id    INTEGER NOT NULL,
                name       TEXT NOT NULL DEFAULT 'New Research',
                messages   TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Auto-name generation ────────────────────────────────────────────────────────

async def _generate_session_name(question: str, answer: str) -> str:
    """Generate a 4-5 word session name from the first exchange."""
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a 4-5 word title for this law research session. "
                        "Be specific — mention the section, act, or legal concept. "
                        "No quotes, no punctuation. Just the title."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question[:200]}\nAnswer summary: {answer[:300]}",
                },
            ],
            max_tokens=20,
            temperature=0.3,
        )
        name = resp.choices[0].message.content.strip().strip('"\'')
        return name[:80] if name else "Law Research Session"
    except Exception as e:
        logger.warning(f"[LawResearch] Auto-name failed: {e}")
        return question[:60] + "..." if len(question) > 60 else question


# ── Models ──────────────────────────────────────────────────────────────────────

class RenameRequest(BaseModel):
    name: str

class AskRequest(BaseModel):
    question: str


# ── Endpoints ───────────────────────────────────────────────────────────────────

@router.get("/sessions")
async def list_sessions(user: UserInfo = Depends(require_auth)):
    init_law_research_tables()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, updated_at FROM law_research_sessions "
            "WHERE user_id = ? ORDER BY updated_at DESC",
            (user.user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("/sessions", status_code=201)
async def create_session(user: UserInfo = Depends(require_auth)):
    if not user.user_id:
        raise HTTPException(400, "User ID required for law research sessions.")
    init_law_research_tables()
    session_id = str(uuid.uuid4())
    now = _now()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO law_research_sessions (id, user_id, name, messages, created_at, updated_at) "
            "VALUES (?, ?, 'New Research', '[]', ?, ?)",
            (session_id, user.user_id, now, now),
        )
        conn.commit()
    return {"id": session_id, "name": "New Research", "updated_at": now}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, user: UserInfo = Depends(require_auth)):
    init_law_research_tables()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM law_research_sessions WHERE id = ? AND user_id = ?",
            (session_id, user.user_id),
        ).fetchone()
    if not row:
        raise HTTPException(404, "Session not found.")
    data = dict(row)
    data["messages"] = json.loads(data["messages"])
    return data


@router.put("/sessions/{session_id}/name")
async def rename_session(
    session_id: str,
    req: RenameRequest,
    user: UserInfo = Depends(require_auth),
):
    if not req.name.strip():
        raise HTTPException(400, "Name cannot be empty.")
    with _get_conn() as conn:
        result = conn.execute(
            "UPDATE law_research_sessions SET name = ?, updated_at = ? "
            "WHERE id = ? AND user_id = ?",
            (req.name.strip()[:80], _now(), session_id, user.user_id),
        )
        conn.commit()
        if result.rowcount == 0:
            raise HTTPException(404, "Session not found.")
    return {"ok": True}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user: UserInfo = Depends(require_auth)):
    with _get_conn() as conn:
        conn.execute(
            "DELETE FROM law_research_sessions WHERE id = ? AND user_id = ?",
            (session_id, user.user_id),
        )
        conn.commit()
    return {"ok": True}


@router.post("/sessions/{session_id}/ask")
async def ask(
    session_id: str,
    req: AskRequest,
    user: UserInfo = Depends(require_auth),
):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    # Load session
    init_law_research_tables()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM law_research_sessions WHERE id = ? AND user_id = ?",
            (session_id, user.user_id),
        ).fetchone()
    if not row:
        raise HTTPException(404, "Session not found.")

    session = dict(row)
    history = json.loads(session["messages"])
    is_first_message = len(history) == 0

    async def generate():
        import asyncio
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

            # Build messages for OpenAI
            openai_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            # Include last 10 exchanges for context (20 messages)
            for msg in history[-20:]:
                openai_messages.append({"role": msg["role"], "content": msg["content"]})
            openai_messages.append({"role": "user", "content": req.question.strip()})

            # Stream response
            stream = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=openai_messages,
                max_tokens=1500,
                temperature=0.2,
                stream=True,
            )

            full_answer = ""
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_answer += delta
                    # Stream word by word
                    yield f"data: {json.dumps({'type': 'word', 'content': delta})}\n\n"
                    await asyncio.sleep(0)

            # Save to session history
            new_history = history + [
                {"role": "user",      "content": req.question.strip(), "created_at": _now()},
                {"role": "assistant", "content": full_answer,           "created_at": _now()},
            ]

            # Auto-name on first message
            auto_name = None
            if is_first_message:
                auto_name = await _generate_session_name(req.question, full_answer)
                yield f"data: {json.dumps({'type': 'name', 'content': auto_name})}\n\n"

            # Persist to DB
            with _get_conn() as conn:
                update = {"messages": json.dumps(new_history, ensure_ascii=False), "updated_at": _now()}
                if auto_name:
                    conn.execute(
                        "UPDATE law_research_sessions SET messages = ?, updated_at = ?, name = ? WHERE id = ?",
                        (update["messages"], update["updated_at"], auto_name, session_id),
                    )
                else:
                    conn.execute(
                        "UPDATE law_research_sessions SET messages = ?, updated_at = ? WHERE id = ?",
                        (update["messages"], update["updated_at"], session_id),
                    )
                conn.commit()

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            logger.info(f"[LawResearch] Session {session_id}: {len(full_answer)} chars")

        except Exception as e:
            logger.error(f"[LawResearch] Error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )