"""
resolver_ui/app.py
-------------------
FastAPI web server for LAW.ai.

Routes:
    GET  /cases                    — List all cases (JSON)
    GET  /cases/{case_id}          — Single case metadata (JSON)
    GET  /conversation/{case_id}   — Conversation history (JSON)
    GET  /resolution-state/{case_id} — Resolution state (JSON)
    GET  /kge-status/{case_id}     — KGE training status (JSON)
    POST /upload                   — Upload PDF + start ingestion in background
    GET  /staged-status/{case_id}  — Get staged decisions (JSON)
    POST /stage/{case_id}          — Stage a decision
    POST /confirm-all/{case_id}    — Commit decisions + apply to Neo4j + start KGE
    POST /ask/{case_id}            — SSE streaming answer
    GET  /pdf/{case_id}/{filename} — Serve PDF file

Run with:
    poetry run uvicorn main:app --reload
"""

import json
import logging
import os
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request, Form, UploadFile, File
from typing import List
from fastapi.responses import (
    HTMLResponse, RedirectResponse, JSONResponse,
    FileResponse, StreamingResponse,
)
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.security import HTTPBearer

from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_ingestion_locks: dict[str, bool] = {}
_ingestion_locks_mutex = threading.Lock()
# Global semaphore — only one ingestion pipeline runs at a time
# GLiNER model is not thread-safe under concurrent load on CPU
_global_ingestion_semaphore = threading.Semaphore(1)

app = FastAPI(title="LAW.ai")
from resolver_ui.auth import require_auth, router as auth_router
from fastapi import Depends
app.include_router(auth_router)
templates = Jinja2Templates(directory="resolver_ui/templates")

# CORS — allow React frontend on port 3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    settings.ensure_dirs()
    for case_dir in settings.cases_dir.iterdir():
        if case_dir.is_dir():
            stale = case_dir / "staged_decisions.json"
            if stale.exists():
                stale.unlink()
                logger.info(f"Cleared stale staged_decisions for {case_dir.name}")

@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "service": "verdictra-law-ai"})
# ── Helpers ────────────────────────────────────────────────────────────────────

def load_metadata(case_id: str) -> dict:
    with open(settings.case_metadata(case_id), "r", encoding="utf-8") as f:
        return json.load(f)


def load_state(case_id: str) -> dict:
    with open(settings.case_resolution_state(case_id), "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(case_id: str, state: dict):
    with open(settings.case_resolution_state(case_id), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def staged_path(case_id: str) -> Path:
    return settings.case_path(case_id) / "staged_decisions.json"


def load_staged(case_id: str) -> dict:
    p = staged_path(case_id)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_staged(case_id: str, staged: dict):
    with open(staged_path(case_id), "w", encoding="utf-8") as f:
        json.dump(staged, f, indent=2, ensure_ascii=False)


def load_conversation(case_id: str) -> list:
    p = settings.case_conversation(case_id)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_conversation(case_id: str, history: list):
    with open(settings.case_conversation(case_id), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def all_cases() -> list:
    cases = []
    if not settings.cases_dir.exists():
        return cases
    for case_dir in settings.cases_dir.iterdir():
        if not case_dir.is_dir():
            continue
        meta_path = case_dir / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                cases.append(json.load(f))
        except Exception:
            continue
    cases.sort(key=lambda c: c.get("created_at", ""), reverse=True)
    return cases


# ── Background ingestion ───────────────────────────────────────────────────────

def _run_ingestion_background(pdf_path: str, case_id: str):
    """
    Full ingestion pipeline running in a background thread.
    PDF → Markdown → Tree → GLiNER → Neo4j → extraction.json → pages.json
    Updates metadata.json status throughout.
    """
    logger.info(f"[Ingestion] Starting for {case_id}...")
    try:
        from ingestion import run_pipeline
        run_pipeline(pdf_path, case_id=case_id)
        logger.info(f"[Ingestion] Complete for {case_id}")

        # After ingestion, run entity resolver to generate resolution_state.json
        logger.info(f"[Resolver] Running entity resolver for {case_id}...")
        try:
            from pipeline.entity_resolver import EntityResolver
            resolver = EntityResolver()
            resolver.resolve(
                str(settings.case_extraction(case_id)),
                str(settings.case_resolution_state(case_id)),
            )
            logger.info(f"[Resolver] Complete for {case_id}")

            # Update status to review
            meta = load_metadata(case_id)
            meta["status"] = "review"
            with open(settings.case_metadata(case_id), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            logger.info(f"[Ingestion] Status set to 'review' for {case_id}")

        except Exception as e:
            logger.error(f"[Resolver] Failed for {case_id}: {e}")
            # Still mark as review so user can proceed
            try:
                meta = load_metadata(case_id)
                meta["status"] = "review"
                with open(settings.case_metadata(case_id), "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
            except Exception:
                pass

    except Exception as e:
        logger.error(f"[Ingestion] Failed for {case_id}: {e}")
        import traceback
        traceback.print_exc()
        try:
            meta = load_metadata(case_id)
            meta["status"] = "failed"
            meta["error"] = str(e)
            with open(settings.case_metadata(case_id), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass


# ── JSON API endpoints ─────────────────────────────────────────────────────────

@app.get("/cases")
async def list_cases(user=Depends(require_auth)):
    cases = all_cases()
    # Admin sees all cases - lawyers only see their own
    if user.role != "admin":
        cases = [c for c in cases if c.get("created_by") == user.username]
    return JSONResponse(cases)


@app.get("/cases/{case_id}")
async def get_case(case_id: str):
    try:
        return JSONResponse(load_metadata(case_id))
    except Exception:
        return JSONResponse({"error": "Not found"}, status_code=404)


@app.get("/conversation/{case_id}")
async def get_conversation(case_id: str):
    return JSONResponse(load_conversation(case_id))


@app.get("/resolution-state/{case_id}")
async def resolution_state(case_id: str):
    try:
        return JSONResponse(load_state(case_id))
    except Exception:
        return JSONResponse({"error": "Not found"}, status_code=404)


@app.get("/kge-status/{case_id}")
async def kge_status(case_id: str):
    try:
        from pipeline.kge_trainer import get_kge_status
        status = get_kge_status(case_id)
        return JSONResponse({"case_id": case_id, "kge_status": status})
    except Exception:
        return JSONResponse({"case_id": case_id, "kge_status": "not_started"})

# ── 1. ADD /domains endpoint ───────────
 
@app.get("/domains")
async def list_domains():
    """Return available case domains for the upload form dropdown."""
    try:
        from pipeline.domains.registry import DomainRegistry
        registry = DomainRegistry()
        return JSONResponse(registry.list_domains())
    except Exception as e:
        logger.error(f"Failed to load domains: {e}")
        # Fallback hardcoded list if registry fails
        return JSONResponse([
            {"id": "constitutional", "name": "Constitutional & Writ",   "description": "Writ petitions, fundamental rights, Article 226/32, PIL"},
            {"id": "property",       "name": "Property & Real Estate",   "description": "Property disputes, lease, SARFAESI auctions, development authority"},
            {"id": "banking_finance","name": "Banking & Finance",        "description": "Loan recovery, NPA, SARFAESI, DRT, debt restructuring"},
            {"id": "corporate",      "name": "Corporate & Company Law",  "description": "Company disputes, shareholder rights, NCLT, mergers"},
            {"id": "criminal",       "name": "Criminal Law",             "description": "FIR, bail, charge sheet, trial, conviction, appeal"},
            {"id": "tax",            "name": "Tax Law",                  "description": "Income tax, GST, customs, assessment disputes"},
            {"id": "labour",         "name": "Labour & Employment",      "description": "Industrial disputes, termination, labour court, awards"},
            {"id": "ip_patent",      "name": "IP & Patents",             "description": "Patents, trademarks, copyright, licensing disputes"},
        ])

@app.get("/staged-status/{case_id}")
async def staged_status(case_id: str):
    return JSONResponse(load_staged(case_id))


# ── PDF serving ────────────────────────────────────────────────────────────────

@app.get("/pdf/{case_id}/{filename}")
async def serve_pdf(case_id: str, filename: str):
    pdf_path = settings.case_path(case_id) / filename
    if not pdf_path.exists():
        return JSONResponse({"error": "PDF not found"}, status_code=404)
    return FileResponse(str(pdf_path), media_type="application/pdf")


# ── Jinja2 HTML routes (legacy — keep until React fully replaces) ──────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "request": request,
        "cases": all_cases(),
    })


@app.get("/processing/{case_id}", response_class=HTMLResponse)
async def processing(request: Request, case_id: str):
    try:
        meta = load_metadata(case_id)
    except Exception:
        return RedirectResponse("/")
    return templates.TemplateResponse(request, "processing.html", {
        "request": request,
        "case": meta,
    })


@app.get("/case/{case_id}")
async def open_case(case_id: str):
    try:
        meta = load_metadata(case_id)
    except Exception:
        return RedirectResponse("/")
    status = meta.get("status", "processing")
    if status == "ready":
        return RedirectResponse(f"/query/{case_id}")
    elif status == "review":
        return RedirectResponse(f"/review/{case_id}")
    else:
        return RedirectResponse(f"/processing/{case_id}")


@app.get("/review/{case_id}", response_class=HTMLResponse)
async def review(request: Request, case_id: str, idx: int = 0):
    try:
        state  = load_state(case_id)
        staged = load_staged(case_id)
        meta   = load_metadata(case_id)
    except Exception:
        return RedirectResponse("/")

    needs_review = state["needs_review"]
    if not needs_review or idx >= len(needs_review):
        return RedirectResponse(f"/complete/{case_id}")

    group    = needs_review[idx]
    total    = len(needs_review)
    reviewed = len(staged)

    sidebar = []
    for i, g in enumerate(needs_review):
        stage  = staged.get(str(i))
        status = stage["decision"] if stage else "PENDING"
        sidebar.append({
            "idx":         i,
            "label":       g.get("canonical_name") or g.get("schema_type", f"Group {i+1}"),
            "schema_type": g.get("schema_type", ""),
            "status":      status,
        })

    return templates.TemplateResponse(request, "review.html", {
        "request":        request,
        "group":          group,
        "idx":            idx,
        "total":          total,
        "reviewed":       reviewed,
        "has_prev":       idx > 0,
        "has_next":       idx < total - 1,
        "next_idx":       min(idx + 1, total - 1),
        "prev_idx":       max(idx - 1, 0),
        "sidebar":        sidebar,
        "current_staged": staged.get(str(idx)),
        "case_id":        case_id,
        "case_name":      meta.get("case_name", case_id),
        "pdf_filename":   meta.get("pdf_filename", ""),
    })


@app.get("/complete/{case_id}", response_class=HTMLResponse)
async def complete(request: Request, case_id: str):
    try:
        state = load_state(case_id)
        meta  = load_metadata(case_id)
    except Exception:
        return RedirectResponse("/")

    needs_review = state["needs_review"]
    reviewed     = sum(1 for g in needs_review if g.get("human_decision"))
    skipped      = len(needs_review) - reviewed

    return templates.TemplateResponse(request, "complete.html", {
        "request":       request,
        "case_id":       case_id,
        "case_name":     meta.get("case_name", case_id),
        "reviewed":      reviewed,
        "skipped":       skipped,
        "auto_merge":    len(state["auto_merge"]),
        "disagreements": sum(1 for g in needs_review if g.get("disagreement")),
    })


@app.get("/query/{case_id}", response_class=HTMLResponse)
async def query_page(request: Request, case_id: str):
    try:
        meta    = load_metadata(case_id)
        history = load_conversation(case_id)
    except Exception:
        return RedirectResponse("/")
    return templates.TemplateResponse(request, "query.html", {
        "request":   request,
        "case_id":   case_id,
        "case_name": meta.get("case_name", case_id),
        "pdf_name":  meta.get("pdf_filename", ""),
        "history":   history,
    })


# ── Upload — fires background ingestion ───────────────────────────────────────

@app.post("/upload")
async def upload_case(
    case_name:  str = Form(...),
    pdf_files:  List[UploadFile] = File(...),
    domain:     str = Form(default="constitutional"),
    user=Depends(require_auth),
):
    from pipeline.document_registry import DocumentRegistry

    # Sanitize case_id
    case_id = re.sub(r"[^a-z0-9_]", "_", case_name.lower().strip())
    case_id = re.sub(r"_+", "_", case_id).strip("_")

    settings.ensure_case_dirs(case_id)

    # Initialise document registry
    registry = DocumentRegistry(settings.case_path(case_id))

    # Save all uploaded files and register each one
    saved_files = []
    for upload in pdf_files:
        content  = await upload.read()
        filename = upload.filename or f"{case_id}.pdf"
        # Save to documents/ subfolder
        file_path = registry.document_path(filename)
        with open(file_path, "wb") as f:
            f.write(content)
        # Register and get doc_id
        doc_id = registry.add_document(
            filename=filename,
            file_size_bytes=len(content),
            uploaded_by=user.username,
        )
        saved_files.append({"doc_id": doc_id, "filename": filename, "path": str(file_path)})
        logger.info(f"Saved {filename} as {doc_id} for case {case_id}")

    # For backward compatibility: set pdf_filename to first file
    first_filename = saved_files[0]["filename"] if saved_files else f"{case_id}.pdf"

    metadata = {
        "case_id":      case_id,
        "case_name":    case_name,
        "pdf_filename": first_filename,
        "domain":       domain,
        "status":       "processing",
        "created_at":   datetime.now().isoformat(),
        "pages":        None,
        "kge_status":   "not_started",
        "has_tree":     False,
        "created_by":   user.username,
        "doc_count":    len(saved_files),
    }
    with open(settings.case_metadata(case_id), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(settings.case_conversation(case_id), "w", encoding="utf-8") as f:
        json.dump([], f)

    # Prevent duplicate ingestion
    with _ingestion_locks_mutex:
        if _ingestion_locks.get(case_id):
            return JSONResponse(
                {"error": f"Ingestion already running for '{case_id}'. Please wait."},
                status_code=409,
            )
        _ingestion_locks[case_id] = True

    def _run_and_unlock(saved_files: list, case_id: str):
        with _global_ingestion_semaphore:
            try:
                _run_ingestion_background_multi(saved_files, case_id)
            finally:
                with _ingestion_locks_mutex:
                    _ingestion_locks[case_id] = False

    thread = threading.Thread(
        target=_run_and_unlock,
        args=(saved_files, case_id),
        daemon=True,
        name=f"ingestion-{case_id}",
    )
    thread.start()
    logger.info(f"Background ingestion started for {case_id} ({len(saved_files)} files, domain: {domain})")

    return JSONResponse({
        "case_id":   case_id,
        "status":    "processing",
        "doc_count": len(saved_files),
        "documents": [{"doc_id": f["doc_id"], "filename": f["filename"]} for f in saved_files],
    })

# 3. Add new _run_ingestion_background_multi function.
#    Place it RIGHT AFTER the existing _run_ingestion_background function.
#    Add this complete function:

def _run_ingestion_background_multi(saved_files: list, case_id: str):
    """
    Multi-file ingestion. Processes each file sequentially.
    Runs entity resolver ONCE after all files are processed.
    """
    from pipeline.document_registry import DocumentRegistry

    logger.info(f"[Ingestion] Starting multi-file ingestion for {case_id} ({len(saved_files)} files)...")

    registry = DocumentRegistry(settings.case_path(case_id))

    try:
        from ingestion import run_pipeline

        for file_info in saved_files:
            doc_id   = file_info["doc_id"]
            filename = file_info["filename"]
            path     = file_info["path"]

            logger.info(f"[Ingestion] Processing {filename} ({doc_id})...")
            registry.update_status(doc_id, "processing")

            try:
                run_pipeline(
                    path,
                    case_id=case_id,
                    doc_id=doc_id,
                    doc_filename=filename,
                )
                registry.update_status(doc_id, "processed")
                logger.info(f"[Ingestion] {filename} complete")
            except Exception as e:
                logger.error(f"[Ingestion] {filename} failed: {e}")
                registry.update_status(doc_id, "failed")
                # Update case metadata with error
                try:
                    meta = load_metadata(case_id)
                    meta["status"] = "failed"
                    meta["error"]  = f"{filename}: {str(e)}"
                    with open(settings.case_metadata(case_id), "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)
                except Exception:
                    pass
                return  # Stop processing remaining files on failure

        # All files processed — run entity resolver once
        logger.info(f"[Resolver] Running entity resolver for {case_id}...")
        try:
            from pipeline.entity_resolver import EntityResolver
            resolver = EntityResolver()
            resolver.resolve(
                str(settings.case_extraction(case_id)),
                str(settings.case_resolution_state(case_id)),
            )
            logger.info(f"[Resolver] Complete for {case_id}")

            meta = load_metadata(case_id)
            meta["status"] = "review"
            with open(settings.case_metadata(case_id), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            logger.info(f"[Ingestion] Status set to 'review' for {case_id}")

        except Exception as e:
            logger.error(f"[Resolver] Failed for {case_id}: {e}")
            try:
                meta = load_metadata(case_id)
                meta["status"] = "review"
                with open(settings.case_metadata(case_id), "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
            except Exception:
                pass

    except Exception as e:
        logger.error(f"[Ingestion] Multi-file ingestion failed for {case_id}: {e}")
        import traceback; traceback.print_exc()
        try:
            meta = load_metadata(case_id)
            meta["status"] = "failed"
            meta["error"]  = str(e)
            with open(settings.case_metadata(case_id), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass
# ── Stage decision ─────────────────────────────────────────────────────────────

@app.post("/stage/{case_id}")
async def stage_decision(case_id: str, request: Request):
    body    = await request.json()
    idx     = str(body.get("idx"))
    staged  = load_staged(case_id)
    staged[idx] = {
        "group_id":  body.get("group_id"),
        "decision":  body.get("decision"),
        "buckets":   body.get("buckets", {}),
        "staged_at": datetime.now().isoformat(),
    }
    save_staged(case_id, staged)
    return JSONResponse({"status": "staged", "idx": idx})


# ── Confirm all ────────────────────────────────────────────────────────────────

@app.post("/confirm-all/{case_id}")
async def confirm_all(case_id: str):
    state        = load_state(case_id)
    staged       = load_staged(case_id)
    needs_review = state["needs_review"]
    log_path     = settings.case_decisions_log(case_id)

    for idx_str, decision in staged.items():
        idx = int(idx_str)
        if idx >= len(needs_review):
            continue
        group = needs_review[idx]

        llm_votes    = [c.get("llm_vote") for c in group.get("candidates", [])]
        llm_merge    = sum(1 for v in llm_votes if v == "YES") > len(llm_votes) / 2
        disagreement = llm_merge != (decision["decision"] == "MERGE")

        group["human_decision"] = decision["decision"]
        group["human_buckets"]  = decision.get("buckets", {})
        group["disagreement"]   = disagreement
        group["decided_at"]     = decision["staged_at"]

        buckets = decision.get("buckets", {})
        if buckets:
            group["canonical_name"] = next(iter(buckets.keys()))

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"[{datetime.now().isoformat()}] "
                f"Group {decision['group_id']} | "
                f"Decision: {decision['decision']} | "
                f"Buckets: {json.dumps(buckets)} | "
                f"Disagreement: {disagreement}\n"
            )

    save_state(case_id, state)

    # Apply decisions to Neo4j
    try:
        from pipeline.entity_resolver import EntityResolver
        EntityResolver().apply_decisions(
            settings.case_resolution_state(case_id), log_path
        )
        logger.info(f"Neo4j decisions applied for {case_id}")
    except Exception as e:
        logger.error(f"Neo4j apply failed: {e}")

    # Clear staged decisions
    sp = staged_path(case_id)
    if sp.exists():
        sp.unlink()

    # Mark case ready
    try:
        meta = load_metadata(case_id)
        meta["status"]     = "ready"
        meta["kge_status"] = "not_started"
        with open(settings.case_metadata(case_id), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

    # Fire KGE training in background — non-blocking
    try:
        from pipeline.kge_trainer import start_kge_training
        started = start_kge_training(case_id)
        if started:
            logger.info(f"KGE background training started for {case_id}")
    except Exception as e:
        logger.warning(f"KGE training could not start: {e}")

    return JSONResponse({"status": "committed", "count": len(staged)})

# ── Add both endpoints to resolver_ui/app.py ──────────────────────────────────
# Place after the /confirm-all endpoint


@app.delete("/cases/{case_id}")
async def delete_case(case_id: str, user=Depends(require_auth)):
    """
    Delete a case completely:
    - Wipes all Neo4j nodes and relationships for this case_id
    - Deletes the case folder from disk
    """
    # Check ownership
    try: 
        meta = load_metadata(case_id)
        if user.role != "admin" and meta.get("created_by") != user.username:
            return JSONResponse(
                {"error": "Not authorized to delete this case"},
                status_code=403,
            )
    except Exception:
        pass
    import shutil

    errors = []

    # 1. Wipe Neo4j nodes for this case
    try:
        from pipeline.graph_builder import GraphBuilder
        with GraphBuilder() as builder:
            builder.clear_case(case_id)
        logger.info(f"Neo4j nodes cleared for case '{case_id}'")
    except Exception as e:
        logger.error(f"Neo4j clear failed for '{case_id}': {e}")
        errors.append(f"Graph clear failed: {e}")

    # 2. Delete case folder from disk
    try:
        case_dir = settings.case_path(case_id)
        if case_dir.exists():
            shutil.rmtree(case_dir)
            logger.info(f"Case folder deleted: {case_dir}")
        else:
            logger.warning(f"Case folder not found: {case_dir}")
    except Exception as e:
        logger.error(f"Folder delete failed for '{case_id}': {e}")
        errors.append(f"Folder delete failed: {e}")

    # 3. Clear pipeline cache so next upload doesn't use stale state
    cache_key = f"pipeline_{case_id}"
    if hasattr(app.state, cache_key):
        delattr(app.state, cache_key)

    if errors:
        return JSONResponse(
            {"status": "partial", "case_id": case_id, "errors": errors},
            status_code=207,
        )

    return JSONResponse({"status": "deleted", "case_id": case_id})


@app.post("/retry/{case_id}")
async def retry_ingestion(case_id: str):
    """
    Retry a failed ingestion.
    Resets metadata status to processing and re-fires the pipeline.
    """
    try:
        meta = load_metadata(case_id)
    except Exception:
        return JSONResponse({"error": "Case not found"}, status_code=404)

    pdf_filename = meta.get("pdf_filename", f"{case_id}.pdf")
    pdf_path     = settings.case_path(case_id) / pdf_filename

    if not pdf_path.exists():
        return JSONResponse(
            {"error": f"PDF not found at {pdf_path}"},
            status_code=404,
        )

    # Reset metadata
    meta["status"]        = "processing"
    meta["current_stage"] = 1
    meta.pop("error", None)
    with open(settings.case_metadata(case_id), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Clear any stale pipeline cache
    cache_key = f"pipeline_{case_id}"
    if hasattr(app.state, cache_key):
        delattr(app.state, cache_key)

    # Re-fire ingestion in background thread
    thread = threading.Thread(
        target=_run_ingestion_background,
        args=(str(pdf_path), case_id),
        daemon=True,
        name=f"ingestion-retry-{case_id}",
    )
    thread.start()
    logger.info(f"Retry ingestion started for {case_id}")

    return JSONResponse({"status": "retrying", "case_id": case_id})

# ── Ask — SSE streaming ────────────────────────────────────────────────────────

@app.post("/ask/{case_id}")
async def ask(case_id: str, request: Request, user=Depends(require_auth)):
    import asyncio

    body     = await request.json()
    question = body.get("question", "").strip()
    if not question:
        return JSONResponse({"error": "Empty question"}, status_code=400)

    async def generate():
        try:
            from query_pipeline import QueryPipeline

            cache_key = f"pipeline_{case_id}"
            if not hasattr(app.state, cache_key):
                pipeline = QueryPipeline()
                pipeline.load(
                    extraction_json_path=str(settings.case_extraction(case_id))
                )
                setattr(app.state, cache_key, pipeline)
            pipeline = getattr(app.state, cache_key)

            answer = pipeline.query(question)

            # Log graph relationships
            logger.info(f"\n=== [{case_id}] GRAPH RELATIONSHIPS ===")
            for c in (answer.citations or []):
                if c.get("source") == "Graph":
                    logger.info(f"  [Graph] {c.get('detail', '')} p.{c.get('page', '')}")

            # Stream words
            words = answer.answer.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"data: {json.dumps({'type': 'word', 'content': chunk})}\n\n"
                await asyncio.sleep(0.04)

            # Build document citations only
            pdf_filename = load_metadata(case_id).get("pdf_filename", "")
            citations = [
                {
                    "text": c.get("text", ""),
                    "page": c.get("page"),
                    "pdf":  pdf_filename,
                }
                for c in (answer.citations or [])
                if c.get("source") == "Document" and c.get("page")
            ]

            yield f"data: {json.dumps({'type': 'done', 'citations': citations, 'confidence': answer.confidence, 'answer_type': answer.answer_type, 'hops': answer.hops_taken})}\n\n"

            # Save to conversation history
            history = load_conversation(case_id)
            history.append({
                "question":    question,
                "answer":      answer.answer,
                "citations":   citations,
                "confidence":  answer.confidence,
                "answer_type": answer.answer_type,
                "hops":        answer.hops_taken,
                "asked_at":    datetime.now().isoformat(),
            })
            save_conversation(case_id, history)

        except Exception as e:
            logger.error(f"Query failed: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ── DEEP RESEARCH endpoint ──

@app.post("/deep-research/{case_id}")
async def deep_research(case_id: str, request: Request, user=Depends(require_auth)):
    """
    Deep Research mode — comprehensive analysis using all available context.
    Activates when lawyer clicks the Deep Research button in QueryPage.
    Returns a structured research report with 9 sections.
    """
    import asyncio

    body  = await request.json()
    query = body.get("query", "").strip()
    if not query:
        query = "Provide a comprehensive analysis of this case"

    async def generate():
        try:
            from query_pipeline import QueryPipeline

            cache_key = f"pipeline_{case_id}"
            if not hasattr(app.state, cache_key):
                pipeline = QueryPipeline()
                pipeline.load(
                    extraction_json_path=str(settings.case_extraction(case_id))
                )
                setattr(app.state, cache_key, pipeline)
            pipeline = getattr(app.state, cache_key)

            # Run COMPLEX classification to get both graph + tree paths
            from retrieval.query_classifier import QueryType
            graph_result = pipeline.graph_retriever.search(query, top_k=50)
            tree_results = pipeline.tree_retriever.search(query, top_k=20)
            fused = pipeline.fusion.fuse(
                query, graph_result, tree_results, pipeline.graph_retriever
            )

            # Deep research mode on the agent
            answer = pipeline.agent.deep_research(
                query,
                fused,
                graph_retriever=pipeline.graph_retriever,
                tree_retriever=pipeline.tree_retriever,
                fusion_engine=pipeline.fusion,
            )

            logger.info(
                f"[Deep Research] {case_id}: "
                f"{len(answer.answer)} chars, "
                f"confidence={answer.confidence:.2f}"
            )

            # Stream words
            words = answer.answer.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield f"data: {json.dumps({'type': 'word', 'content': chunk})}\n\n"
                await asyncio.sleep(0.02)

            # Build citations
            pdf_filename = load_metadata(case_id).get("pdf_filename", "")
            citations = [
                {
                    "text": c.get("text", ""),
                    "page": c.get("page"),
                    "pdf":  pdf_filename,
                }
                for c in (answer.citations or [])
                if c.get("page")
            ]

            yield f"data: {json.dumps({'type': 'done', 'citations': citations, 'confidence': answer.confidence, 'answer_type': 'DEEP_RESEARCH', 'hops': answer.hops_taken})}\n\n"

            # Save to conversation history
            history = load_conversation(case_id)
            history.append({
                "question":    f"[DEEP RESEARCH] {query}",
                "answer":      answer.answer,
                "citations":   citations,
                "confidence":  answer.confidence,
                "answer_type": "DEEP_RESEARCH",
                "hops":        answer.hops_taken,
                "asked_at":    datetime.now().isoformat(),
            })
            save_conversation(case_id, history)

        except Exception as e:
            logger.error(f"Deep research failed: {e}")
            import traceback; traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
# ── Legacy redirects ───────────────────────────────────────────────────────────

@app.get("/review", response_class=HTMLResponse)
async def review_legacy(request: Request, idx: int = 0):
    return RedirectResponse(f"/review/celir_case?idx={idx}")


@app.get("/query", response_class=HTMLResponse)
async def query_legacy(request: Request):
    return RedirectResponse("/query/celir_case")


@app.get("/complete", response_class=HTMLResponse)
async def complete_legacy(request: Request):
    return RedirectResponse("/complete/celir_case")


@app.get("/pdf/{filename}")
async def serve_pdf_legacy(filename: str):
    return RedirectResponse(f"/pdf/celir_case/{filename}")

@app.get("/annotations/{case_id}")
async def get_annotations(case_id: str):
    """Load all post-it annotations for a case."""
    path = settings.case_path(case_id) / "annotations.json"
    if not path.exists():
        return JSONResponse([])
    with open(path, "r", encoding="utf-8") as f:
        return JSONResponse(json.load(f))
 
 
@app.post("/annotations/{case_id}")
async def create_annotation(case_id: str, request: Request):
    """Save a new post-it annotation."""
    body = await request.json()
 
    path = settings.case_path(case_id) / "annotations.json"
    existing = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
 
    annotation = {
        "id": f"ann_{uuid.uuid4().hex[:8]}",
        "page": body.get("page"),
        "pdf": body.get("pdf"),
        "note": body.get("note", ""),
        "position": body.get("position", {"x": 0.1, "y": 0.1}),
        "anchor_text": body.get("anchor_text", ""),
        "color": body.get("color", "#fef08a"),
        "created_at": datetime.now().isoformat(),
    }
 
    existing.append(annotation)
 
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
 
    return JSONResponse(annotation)
 
 
@app.delete("/annotations/{case_id}/{annotation_id}")
async def delete_annotation(case_id: str, annotation_id: str):
    """Delete a post-it annotation by id."""
    path = settings.case_path(case_id) / "annotations.json"
    if not path.exists():
        return JSONResponse({"status": "not_found"})
 
    with open(path, "r", encoding="utf-8") as f:
        existing = json.load(f)
 
    existing = [a for a in existing if a["id"] != annotation_id]
 
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
 
    return JSONResponse({"status": "deleted"})


if __name__ == "__main__":
    uvicorn.run("resolver_ui.app:app", host="0.0.0.0", port=8000, reload=True)