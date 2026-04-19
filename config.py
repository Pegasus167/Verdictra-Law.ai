"""
config.py
---------
Central configuration for the GraphRAG Legal system.
All credentials and settings are loaded from .env — never hardcoded.

Usage:
    from config import settings
    print(settings.neo4j_uri)
    print(settings.case_path("celir_case"))
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Settings(BaseSettings):
    """
    All settings loaded from environment variables or .env file.
    Pydantic validates types automatically — if a required var is
    missing, the app fails loudly on startup rather than silently later.
    """

    # --- Neo4j ---
    neo4j_uri: str = Field(..., env="NEO4J_URI")
    neo4j_username: str = Field(..., env="NEO4J_USERNAME")
    neo4j_password: str = Field(..., env="NEO4J_PASSWORD")
    neo4j_database: str = Field("neo4j", env="NEO4J_DATABASE")

    # --- OpenAI ---
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")

    # --- Cases root ---
    # All case files live under cases/{case_id}/
    cases_dir: Path = Field(Path("cases"), env="CASES_DIR")

    # --- Legacy paths (kept for pipeline scripts that haven't migrated yet) ---
    pdf_split_dir: Path = Field(Path("data/split_pdfs"), env="PDF_SPLIT_DIR")
    jsonld_output_dir: Path = Field(Path("data/jsonld_output"), env="JSONLD_OUTPUT_DIR")
    sqlite_db_path: Path = Field(Path("data/case_docs.db"), env="SQLITE_DB_PATH")
    ontology_path: Path = Field(Path("ontology.ttl"), env="ONTOLOGY_PATH")
    shapes_path: Path = Field(Path("shapes.ttl"), env="SHAPES_PATH")

    # --- GLiNER ---
    gliner_model: str = Field("urchade/gliner_mediumv2.1", env="GLINER_MODEL")

    # --- PyKEEN ---
    pykeen_model: str = Field("RotatE", env="PYKEEN_MODEL")
    pykeen_epochs: int = Field(100, env="PYKEEN_EPOCHS")
    pykeen_embedding_dim: int = Field(128, env="PYKEEN_EMBEDDING_DIM")

    # --- Embeddings ---
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL"
    )

    # --- Retrieval ---
    top_k_graph: int = Field(20, env="TOP_K_GRAPH")
    top_k_tree: int = Field(10, env="TOP_K_TREE")

    # Fusion weights
    fusion_graph_weight: float = Field(0.6, env="FUSION_GRAPH_WEIGHT")
    fusion_tree_weight: float = Field(0.4, env="FUSION_TREE_WEIGHT")

    # Node scoring weights
    node_score_similarity: float = Field(0.5, env="NODE_SCORE_SIMILARITY")
    node_score_importance: float = Field(0.3, env="NODE_SCORE_IMPORTANCE")
    node_score_confidence: float = Field(0.2, env="NODE_SCORE_CONFIDENCE")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

    # ── Case path helpers ──────────────────────────────────────────────────────

    def case_path(self, case_id: str) -> Path:
        """Root folder for a case. e.g. cases/celir_case/"""
        return self.cases_dir / case_id

    def case_pdf(self, case_id: str, pdf_filename: str) -> Path:
        """Path to the PDF file for a case."""
        return self.case_path(case_id) / pdf_filename

    def case_extraction(self, case_id: str) -> Path:
        """Path to extraction.json for a case."""
        return self.case_path(case_id) / "extraction.json"

    def case_tree_index(self, case_id: str) -> Path:
        """Path to tree_index/ folder for a case."""
        return self.case_path(case_id) / "tree_index"

    def case_embeddings(self, case_id: str) -> Path:
        """Path to embeddings/ folder for a case."""
        return self.case_path(case_id) / "embeddings"

    def case_resolution_state(self, case_id: str) -> Path:
        """Path to resolution_state.json for a case."""
        return self.case_path(case_id) / "resolution_state.json"

    def case_decisions_log(self, case_id: str) -> Path:
        """Path to decisions.log for a case."""
        return self.case_path(case_id) / "decisions.log"

    def case_metadata(self, case_id: str) -> Path:
        """Path to metadata.json for a case."""
        return self.case_path(case_id) / "metadata.json"

    def case_conversation(self, case_id: str) -> Path:
        """Path to conversation_history.json for a case."""
        return self.case_path(case_id) / "conversation_history.json"

    def ensure_case_dirs(self, case_id: str):
        """Create all required directories for a new case."""
        dirs = [
            self.case_path(case_id),
            self.case_tree_index(case_id),
            self.case_embeddings(case_id),
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

    def ensure_dirs(self):
        """Create top-level directories."""
        self.cases_dir.mkdir(parents=True, exist_ok=True)


# Singleton — import this everywhere
settings = Settings()