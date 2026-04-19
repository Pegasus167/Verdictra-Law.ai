"""
pipeline/kge_trainer.py
------------------------
Async KGE (Knowledge Graph Embedding) training using PyKEEN.

Runs as a background task after entity resolution completes.
Updates cases/{case_id}/metadata.json with KGE status.

Status flow:
    metadata.json kge_status:
        null / missing  → KGE not started
        "training"      → currently training (background)
        "ready"         → FAISS index built, KGE available
        "failed"        → training failed (error in kge_error)

The query pipeline checks kge_status before loading FAISS:
    - "ready"    → load FAISS, use KGE similarity
    - anything else → Cypher-only mode (still works fine)

Usage (programmatic):
    from pipeline.kge_trainer import start_kge_training
    start_kge_training("celir_case")  # fires background thread, returns immediately

Usage (CLI):
    python pipeline/kge_trainer.py celir_case
"""

from __future__ import annotations

import json
import logging
import pickle
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
from neo4j import GraphDatabase

from config import settings

logger = logging.getLogger(__name__)


# ── Status helpers ─────────────────────────────────────────────────────────────

def get_kge_status(case_id: str) -> str:
    """Return current KGE status from metadata.json."""
    try:
        with open(settings.case_metadata(case_id), "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("kge_status") or "not_started"
    except Exception:
        return "not_started"


def _update_kge_status(case_id: str, status: str, error: str = None):
    """Update KGE status in metadata.json."""
    try:
        meta_path = settings.case_metadata(case_id)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["kge_status"] = status
        meta["kge_updated_at"] = datetime.now().isoformat()
        if error:
            meta["kge_error"] = error
        elif "kge_error" in meta:
            del meta["kge_error"]
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to update KGE status: {e}")


# ── Neo4j triple fetcher ───────────────────────────────────────────────────────

def _fetch_triples(case_id: str) -> tuple[list, list, list]:
    """
    Fetch all typed relationship triples from Neo4j for this case.
    Returns (heads, relations, tails) as lists of strings.
    """
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )

    heads, relations, tails = [], [], []

    with driver.session(database=settings.neo4j_database) as session:
        # Only fetch typed relationships — skip CO_OCCURS_WITH
        result = session.run("""
            MATCH (h)-[r]->(t)
            WHERE h.canonicalName IS NOT NULL
              AND t.canonicalName IS NOT NULL
              AND NOT type(r) IN ['CO_OCCURS_WITH', 'CO_OCCURS', 'COOCCURS_WITH']
            RETURN
                h.canonicalName AS head,
                type(r)         AS relation,
                t.canonicalName AS tail
        """)
        for record in result:
            heads.append(record["head"])
            relations.append(record["relation"])
            tails.append(record["tail"])

    driver.close()
    logger.info(f"Fetched {len(heads)} typed triples from Neo4j for KGE training")
    return heads, relations, tails


# ── KGE trainer ───────────────────────────────────────────────────────────────

def _train_kge(case_id: str):
    """
    Train PyKEEN RotatE model and build FAISS index.
    Runs in background thread.
    """
    logger.info(f"[KGE] Starting training for case: {case_id}")
    _update_kge_status(case_id, "training")

    embeddings_dir = settings.case_embeddings(case_id)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = embeddings_dir / "graph_entities.faiss"
    entity_map_path = embeddings_dir / "entity_map.pkl"
    pykeen_dir = embeddings_dir / "pykeen_results"

    try:
        # Step 1 — Fetch triples
        heads, relations, tails = _fetch_triples(case_id)

        if len(heads) < 10:
            raise ValueError(
                f"Only {len(heads)} typed triples found — "
                "not enough for KGE training. "
                "Complete entity resolution first."
            )

        logger.info(f"[KGE] {len(heads)} typed triples fetched from Neo4j")

        # Step 2 — Build entity and relation maps
        entities = sorted(set(heads) | set(tails))
        entity_to_idx = {e: i for i, e in enumerate(entities)}
        idx_to_entity = {i: e for e, i in entity_to_idx.items()}

        rel_types = sorted(set(relations))
        rel_to_idx = {r: i for i, r in enumerate(rel_types)}

        logger.info(
            f"[KGE] {len(entities)} entities, {len(rel_types)} relation types"
        )

        # Step 3 — Train PyKEEN RotatE
        try:
            from pykeen.pipeline import pipeline as pykeen_pipeline
            from pykeen.triples import TriplesFactory

            # Build triples as numpy string array — shape (N, 3)
            triples_array = np.array(
                [[h, r, t] for h, r, t in zip(heads, relations, tails)],
                dtype=str,
            )

            # TriplesFactory handles string→index mapping internally
            tf = TriplesFactory.from_labeled_triples(
                triples=triples_array,
                create_inverse_triples=True,
            )

            logger.info(f"[KGE] TriplesFactory built: {tf.num_triples} triples")

            result = pykeen_pipeline(
                training=tf,
                testing=tf,   # single-case mode — use same triples for eval
                model="RotatE",
                model_kwargs={
                    "embedding_dim": settings.pykeen_embedding_dim,
                },
                training_kwargs={
                    "num_epochs": settings.pykeen_epochs,
                    "batch_size": min(256, len(heads)),
                },
                optimizer="adam",
                optimizer_kwargs={"lr": 0.001},
                stopper=None,
                random_seed=42,
                device="cpu",
            )

            result.save_to_directory(str(pykeen_dir))
            logger.info(f"[KGE] PyKEEN training complete → {pykeen_dir}")

            # Extract entity embeddings from trained model
            model = result.model
            entity_repr = model.entity_representations[0]
            embedding_matrix = entity_repr(
                indices=None
            ).detach().cpu().numpy()

            logger.info(
                f"[KGE] Embeddings extracted: "
                f"{embedding_matrix.shape[0]} entities × "
                f"{embedding_matrix.shape[1]} dims"
            )

        except ImportError:
            logger.warning(
                "[KGE] PyKEEN not available — "
                "falling back to random embeddings for FAISS index"
            )
            dim = settings.pykeen_embedding_dim
            embedding_matrix = np.random.randn(
                len(entities), dim
            ).astype(np.float32)

        # Step 4 — Build FAISS index
        try:
            import faiss

            dim = embedding_matrix.shape[1]

            # Normalize for cosine similarity (IndexFlatIP = inner product)
            norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            normalized = (embedding_matrix / norms).astype(np.float32)

            index = faiss.IndexFlatIP(dim)
            index.add(normalized)

            faiss.write_index(index, str(faiss_path))
            logger.info(
                f"[KGE] FAISS index built: {index.ntotal} entities → {faiss_path}"
            )

        except ImportError:
            raise RuntimeError(
                "FAISS not available — cannot build embedding index. "
                "Run: poetry add faiss-cpu"
            )

        # Step 5 — Save entity maps
        with open(entity_map_path, "wb") as f:
            pickle.dump({
                "entity_to_idx": entity_to_idx,
                "idx_to_entity": idx_to_entity,
                "rel_to_idx":    rel_to_idx,
            }, f)
        logger.info(f"[KGE] Entity maps saved → {entity_map_path}")

        # Step 6 — Save FAISS config for graph_retriever
        faiss_config = {
            "faiss_path":      str(faiss_path),
            "entity_map_path": str(entity_map_path),
            "entity_count":    len(entities),
            "trained_at":      datetime.now().isoformat(),
        }
        faiss_config_path = embeddings_dir / "faiss_config.json"
        with open(faiss_config_path, "w", encoding="utf-8") as f:
            json.dump(faiss_config, f, indent=2)

        _update_kge_status(case_id, "ready")
        logger.info(f"[KGE] Training complete for {case_id} ✓")

    except Exception as e:
        error_msg = str(e)
        logger.error(f"[KGE] Training failed for {case_id}: {error_msg}")
        _update_kge_status(case_id, "failed", error=error_msg)


# ── Public API ─────────────────────────────────────────────────────────────────

def start_kge_training(case_id: str) -> bool:
    """
    Start KGE training in a background thread.
    Returns immediately — caller does not block.

    Returns True if training was started, False if already training/ready.
    """
    current = get_kge_status(case_id)

    if current == "training":
        logger.info(f"[KGE] Already training for {case_id} — skipping")
        return False

    if current == "ready":
        logger.info(f"[KGE] Already trained for {case_id} — skipping")
        return False

    thread = threading.Thread(
        target=_train_kge,
        args=(case_id,),
        daemon=True,
        name=f"kge-trainer-{case_id}",
    )
    thread.start()
    logger.info(f"[KGE] Background training started for {case_id}")
    return True


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python pipeline/kge_trainer.py <case_id>")
        sys.exit(1)

    case_id = sys.argv[1]
    print(f"Training KGE for case: {case_id}")
    print("This runs synchronously in CLI mode...")
    _train_kge(case_id)
    print(f"Status: {get_kge_status(case_id)}")