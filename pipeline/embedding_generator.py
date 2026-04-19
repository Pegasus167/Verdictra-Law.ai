"""
pipeline/embedding_generator.py
---------------------------------
Stage 4 of the ingestion pipeline — runs AFTER entity resolution.

Responsibilities:
    1. Pull all triples (head, relation, tail) from Neo4j
    2. Train a RotatE KGE model using PyKEEN
    3. Save node embeddings to:
       - FAISS index (fast similarity search at query time)
       - Neo4j node properties (persistent storage + inspection)
    4. Save entity-to-index mapping for lookup

Why RotatE over TransE:
    RotatE handles symmetric and antisymmetric relations well.
    Legal graphs have both: "RELATED_TO" is symmetric,
    "FILED_BY" is antisymmetric. RotatE captures both.

Run after ingestion + entity resolution:
    poetry run python pipeline/embedding_generator.py
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import faiss
from neo4j import GraphDatabase
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

from config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
EMBEDDINGS_DIR = Path("data/embeddings")
FAISS_INDEX_PATH = EMBEDDINGS_DIR / "graph_entities.faiss"
ENTITY_MAP_PATH = EMBEDDINGS_DIR / "entity_map.pkl"
PYKEEN_RESULTS_PATH = EMBEDDINGS_DIR / "pykeen_results"


# ── Step 1: Pull triples from Neo4j ───────────────────────────────────────────

def fetch_triples_from_neo4j() -> list[tuple[str, str, str]]:
    """
    Pull all (head, relation, tail) triples from Neo4j.
    Returns list of string triples for PyKEEN.
    """
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )

    triples = []
    with driver.session(database=settings.neo4j_database) as session:
        result = session.run("""
            MATCH (h)-[r]->(t)
            WHERE h.canonicalName IS NOT NULL
              AND t.canonicalName IS NOT NULL
            RETURN
                h.canonicalName AS head,
                type(r)         AS relation,
                t.canonicalName AS tail
        """)
        for record in result:
            head = record["head"]
            rel = record["relation"]
            tail = record["tail"]
            if head and rel and tail:
                triples.append((head, rel, tail))

    driver.close()
    logger.info(f"Fetched {len(triples)} triples from Neo4j")
    return triples


# ── Step 2: Train PyKEEN KGE model ────────────────────────────────────────────

def train_kge(triples: list[tuple[str, str, str]]):
    """
    Train RotatE KGE model on the graph triples.
    Returns the trained pipeline result.
    """
    if len(triples) < 10:
        raise ValueError(
            f"Too few triples ({len(triples)}) to train KGE. "
            "Ensure entity resolution is complete and graph has relationships."
        )

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Training {settings.pykeen_model} on {len(triples)} triples "
        f"({settings.pykeen_epochs} epochs, dim={settings.pykeen_embedding_dim})..."
    )

    # Build triples factory
    tf = TriplesFactory.from_labeled_triples(
        triples=np.array(triples),
        create_inverse_triples=True,  # Improves quality for sparse graphs
    )

    # Train/validation split
    training, testing = tf.split([0.8, 0.2], random_state=42)

    # Run PyKEEN pipeline
    result = pipeline(
        training=training,
        testing=testing,
        model=settings.pykeen_model,
        model_kwargs={
            "embedding_dim": settings.pykeen_embedding_dim,
        },
        training_kwargs={
            "num_epochs": settings.pykeen_epochs,
            "batch_size": 256,
        },
        optimizer="Adam",
        optimizer_kwargs={"lr": 0.001},
        random_seed=42,
        device="cpu",
    )

    # Save PyKEEN results
    result.save_to_directory(str(PYKEEN_RESULTS_PATH))
    logger.info(f"PyKEEN results saved to {PYKEEN_RESULTS_PATH}")

    return result, tf


# ── Step 3: Build FAISS index ──────────────────────────────────────────────────

def build_faiss_index(
    result,
    tf: TriplesFactory,
) -> tuple[faiss.IndexFlatIP, dict[str, int], dict[int, str]]:
    """
    Extract entity embeddings from trained model and build FAISS index.

    Returns:
        index: FAISS index for similarity search
        entity_to_idx: canonical_name → FAISS index position
        idx_to_entity: FAISS index position → canonical_name
    """
    # Extract entity embeddings
    model = result.model
    entity_embeddings = (
        model.entity_representations[0](indices=None)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # Normalize for cosine similarity (inner product on normalized = cosine)
    norms = np.linalg.norm(entity_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    entity_embeddings_normalized = entity_embeddings / norms

    # Build entity maps
    entity_to_idx = {}
    idx_to_entity = {}

    for entity_label, idx in tf.entity_to_id.items():
        entity_to_idx[entity_label] = idx
        idx_to_entity[idx] = entity_label

    # Build FAISS index (inner product for cosine similarity)
    dim = entity_embeddings_normalized.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(entity_embeddings_normalized)

    logger.info(
        f"FAISS index built: {index.ntotal} entities, "
        f"embedding dim={dim}"
    )

    return index, entity_to_idx, idx_to_entity


# ── Step 4: Save everything ────────────────────────────────────────────────────

def save_embeddings(
    index: faiss.IndexFlatIP,
    entity_to_idx: dict[str, int],
    idx_to_entity: dict[int, str],
    result,
    tf: TriplesFactory,
):
    """Save FAISS index and entity maps to disk."""
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    logger.info(f"FAISS index saved → {FAISS_INDEX_PATH}")

    # Save entity maps
    entity_map = {
        "entity_to_idx": entity_to_idx,
        "idx_to_entity": idx_to_entity,
    }
    with open(ENTITY_MAP_PATH, "wb") as f:
        pickle.dump(entity_map, f)
    logger.info(f"Entity map saved → {ENTITY_MAP_PATH}")


# ── Step 5: Store embeddings back in Neo4j ────────────────────────────────────

def store_embeddings_in_neo4j(
    result,
    tf: TriplesFactory,
    idx_to_entity: dict[int, str],
):
    """
    Store embedding vectors as node properties in Neo4j.
    Allows inspection and backup — FAISS is still used for actual search.
    """
    model = result.model
    entity_embeddings = (
        model.entity_representations[0](indices=None)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )

    updated = 0
    with driver.session(database=settings.neo4j_database) as session:
        for idx, entity_name in idx_to_entity.items():
            embedding = entity_embeddings[idx].tolist()
            result_neo = session.run("""
                MATCH (n {canonicalName: $name})
                SET n.embedding = $embedding,
                    n.embeddingDim = $dim,
                    n.embeddingModel = $model
                RETURN count(n) AS updated
            """, {
                "name": entity_name,
                "embedding": embedding,
                "dim": len(embedding),
                "model": settings.pykeen_model,
            })
            updated += result_neo.single()["updated"]

    driver.close()
    logger.info(f"Embeddings stored in Neo4j for {updated} nodes")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_embedding_generator():
    """Full embedding generation pipeline."""

    logger.info("Step 1: Fetching triples from Neo4j...")
    triples = fetch_triples_from_neo4j()

    if not triples:
        logger.error("No triples found. Run ingestion + entity resolution first.")
        return

    logger.info("Step 2: Training KGE model...")
    result, tf = train_kge(triples)

    logger.info("Step 3: Building FAISS index...")
    index, entity_to_idx, idx_to_entity = build_faiss_index(result, tf)

    logger.info("Step 4: Saving FAISS index + entity maps...")
    save_embeddings(index, entity_to_idx, idx_to_entity, result, tf)

    logger.info("Step 5: Storing embeddings in Neo4j...")
    store_embeddings_in_neo4j(result, tf, idx_to_entity)

    logger.info("\n✓ Embedding generation complete.")
    logger.info(f"  FAISS index:  {FAISS_INDEX_PATH}")
    logger.info(f"  Entity map:   {ENTITY_MAP_PATH}")
    logger.info(f"  PyKEEN model: {PYKEEN_RESULTS_PATH}")
    logger.info("\nNext: poetry run python retrieval/query_classifier.py")


if __name__ == "__main__":
    run_embedding_generator()