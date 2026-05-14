<div align="center">

<img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
<img src="https://img.shields.io/badge/Neo4j-AuraDB-4581C3?style=for-the-badge&logo=neo4j&logoColor=white" />
<img src="https://img.shields.io/badge/Docker-Deployed-2496ED?style=for-the-badge&logo=docker&logoColor=white" />

<br />
<br />

```
 ██╗   ██╗███████╗██████╗ ██████╗ ██╗ ██████╗████████╗██████╗  █████╗
 ██║   ██║██╔════╝██╔══██╗██╔══██╗██║██╔════╝╚══██╔══╝██╔══██╗██╔══██╗
 ██║   ██║█████╗  ██████╔╝██║  ██║██║██║        ██║   ██████╔╝███████║
 ╚██╗ ██╔╝██╔══╝  ██╔══██╗██║  ██║██║██║        ██║   ██╔══██╗██╔══██║
  ╚████╔╝ ███████╗██║  ██║██████╔╝██║╚██████╗   ██║   ██║  ██║██║  ██║
   ╚═══╝  ╚══════╝╚═╝  ╚═╝╚═════╝ ╚═╝ ╚═════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝
```

### Legal Intelligence Platform for Indian Law Firms

*Upload a case bundle. Ask any question. Get a cited answer in under 30 seconds.*

<br />

</div>

---

## What This Is

Verdictra is a **hybrid GraphRAG legal intelligence system** built specifically for Indian litigation. It transforms a 200-page case bundle into a queryable knowledge graph in 15 minutes — allowing lawyers and their teams to ask natural language questions about any matter and receive answers grounded in the actual document, cited to the exact page.

This is not a chatbot. It is a structured extraction and reasoning pipeline with human-in-the-loop validation at every critical stage.

---

## Architecture

```
PDF Upload
├── pdf_to_markdown.py       pdfplumber (digital) + Tesseract OCR parallel (scanned)
├── tree_builder.py          Markdown → TreeNode hierarchy → tree.json + pages.json
├── entity_extractor.py      GLiNER (gliner_mediumv2.1) · domain-specific labels · threshold 0.5
├── validator.py             SHACL validation via pyshacl + shapes.ttl + ontology.ttl
├── relationship_extractor.py  Async GPT-4o-mini · typed relationships only · confidence ≥ 0.70
├── graph_builder.py         Neo4j MERGE · (canonicalName, case_id) IS NODE KEY
├── document_registry.py     Multi-file upload tracking · doc_id · document_date
├── extractors/              File type router · PDF · DOCX · email · image · TXT
├── entity_management.py     Confirmed entity registry · incremental resolution
├── entity_resolver.py       sentence-transformers clustering · LLM scoring · AUTO/REVIEW/KEEP
└── kge_trainer.py           PyKEEN RotatE · 100 epochs · 128-dim FAISS index
```

```
Query
├── query_classifier.py      GPT-4o · FACT / RELATIONSHIP / COMPLEX routing
├── graph_retriever.py       Cypher traversal (3-hop) → KGE/FAISS escalation
├── tree_retriever.py        BM25 over pages.json passages
├── fusion + agent           NodeScore formula · fast path · sufficiency check · SSE streaming
└── deep_research()          9-section report · top_k=50 entities · 4000 tokens
```

---

## Pipeline Stages

| Stage | Component | What Happens |
|-------|-----------|-------------|
| **1** | PDF Extraction | pdfplumber for digital pages · 4-worker parallel Tesseract for scanned |
| **2** | Tree Construction | Heading-aware Markdown → hierarchical tree preserving section boundaries |
| **3** | Entity Extraction | GLiNER on 300-word chunks · domain labels merged with universal labels |
| **4** | SHACL Validation | pyshacl rejects noise before Neo4j — length, stopwords, confidence, shape conformance |
| **5** | Relationship Extraction | Async LLM on entity pairs · typed edges (FILED_BY, GOVERNED_BY, MORTGAGED_WITH…) |
| **6** | Graph Construction | Neo4j with case_id on every node and relationship — full case isolation |
| **7** | Entity Resolution | Clustering → LLM scoring → human review → MERGE/KEEP/SKIP |
| **8** | KGE Training | PyKEEN RotatE → FAISS index from 128-dim entity embeddings |

---

## Retrieval

Verdictra runs **four paths simultaneously** and fuses results using a confidence-weighted formula:

```
Query
  ├── Cypher Path       Direct graph traversal · exact typed relationships
  ├── KGE/FAISS Path    Structural similarity via RotatE embeddings · indirect connections
  ├── BM25 Tree Path    Document passage retrieval · verbatim text with source page
  └── GAR Fusion        NodeScore = (1/√(N+1)) × Σ(chunk_scores) · confidence routing
```

**Fast path:** 5+ HIGH confidence nodes, FACT queries only → direct answer
**Multi-hop:** COMPLEX and RELATIONSHIP queries always run sufficiency check + up to 3 hops. FACT queries skip to answer when confidence is high.
**Deep Research:** full graph retrieval (top_k=50) + 9-section structured report

---

## Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI · Python 3.13 · Uvicorn |
| **Entity Extraction** | GLiNER (urchade/gliner_mediumv2.1) |
| **Knowledge Graph** | Neo4j AuraDB · Cypher |
| **KGE** | PyKEEN RotatE · FAISS (128-dim) |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **LLM** | GPT-4o (complex) · GPT-4o-mini (fast path, resolver) |
| **Ontology** | RDF/OWL (ontology.ttl) · SHACL (shapes.ttl) · pyshacl |
| **Frontend** | React 18 · TypeScript · Vite · Tailwind CSS |
| **Auth** | JWT (python-jose) · role-based case ownership |
| **Infrastructure** | Docker · Nginx · Digital Ocean (Ubuntu 24.04 · 4vCPU · 8GB · BLR1) |

---

## Domain Coverage

Eight domain configurations with specialised GLiNER label sets:

`constitutional` `property` `banking_finance` `corporate` `criminal` `ip_patent` `tax` `labour`

Each domain extends the universal label set with domain-specific entity types, relationship vocabulary, and heading patterns. The universal ontology (`ontology.ttl`) and SHACL shapes (`shapes.ttl`) apply to all domains.

---

## Entity Schema

Universal entity classes defined in `ontology.ttl`:

```
Person · Organization · Government · Court · StatutoryBody
FinancialInstitution · Asset · Agreement · Demand · LegalProceeding
Transaction · Regulation · Event · Location · Identifier · Role
```

Typed relationship vocabulary:

```
FILED_BY · ISSUED_BY · ISSUED_TO · TRANSFERRED_TO · LEASED_TO
MORTGAGED_WITH · HOLDS_CHARGE_OVER · RULED_FOR · RULED_AGAINST
DIRECTED · OVERSEES · INVOLVES · AFFILIATED_WITH · DIRECTOR_OF
ADVOCATES_FOR · SUBLET_TO · OWNS · HAS_OFFICER · RELATED_TO
```

---

## Human-in-the-Loop Trust (HILT)

Every entity extracted by GLiNER is presented for human review before being committed to the knowledge graph. The review interface shows:

- Every entity grouped by merge candidates
- The exact page in the source PDF where each entity was found
- Clickable page citations — opens PDF at the source location
- LLM vote, confidence score, and reasoning for each merge decision
- MERGE / KEEP / SKIP decisions staged before final commit

This is a deliberate architectural choice. Indian legal practice runs on evidence and verified reasoning. The review page is the trust layer — not a UX feature.

---

## Repository Structure

```
LAW.ai/
├── pipeline/
│   ├── pdf_to_markdown.py       PDF → Markdown (pdfplumber + Tesseract)
│   ├── tree_builder.py          Markdown → TreeNode hierarchy
│   ├── entity_extractor.py      GLiNER entity extraction
│   ├── validator.py             SHACL validation layer
│   ├── relationship_extractor.py  Async LLM relationship extraction
│   ├── graph_builder.py         Neo4j write layer (case_id partitioned)
│   ├── entity_resolver.py       Clustering + LLM scoring + resolution state
│   ├── kge_trainer.py           PyKEEN RotatE + FAISS index
│   ├── document_registry.py
│   ├── entity_management.py
│   ├── domains/
│   │   ├── registry.py          Domain config loader
│   │   ├── universal.json
│   │   ├── constitutional.json
│   │   ├── banking_finance.json
│   │   └── ...
│   └── extractors/
│       ├── router.py
│       ├── pdf_extractor.py
│       ├── docx_extractor.py
│       ├── email_extractor.py
│       ├── image_extractor.py
│       └── txt_extractor.py
├── retrieval/
│   ├── graph_retriever.py       Cypher + KGE/FAISS retrieval
│   ├── tree_retriever.py        BM25 passage retrieval
│   ├── query_classifier.py      FACT / RELATIONSHIP / COMPLEX routing
│   ├── agent.py                 Answer generation + Deep Research
│   └── query_pipeline.py        Pipeline orchestrator
├── resolver_ui/
│   ├── app.py                   FastAPI application (all endpoints)
│   └── auth.py                  JWT authentication + case ownership
├── app/frontend/
│   └── src/
│       ├── pages/
│       │   ├── CasesPage.tsx
│       │   ├── ProcessingPage.tsx
│       │   ├── SummaryPage.tsx
│       │   ├── ReviewPage.tsx
│       │   ├── QueryPage.tsx
│       │   ├── LoginPage.tsx
│       │   ├── SignupPage.tsx
│       │   ├── VerifyEmailPage.tsx
│       │   ├── PasswordPages.tsx
│       │   ├── ProfilePage.tsx
│       │   └── WelcomePage.tsx
│       ├── lib/api.ts            Auth-wrapped API client
│       └── App.tsx               Router + auth guard
├── ontology.ttl                  RDF/OWL entity class definitions
├── shapes.ttl                    SHACL validation shapes
├── ingestion.py                  Pipeline entry point
├── query_pipeline.py             Query entry point
├── config.py                     Settings and path resolution
├── Dockerfile                    Backend container
├── Dockerfile.frontend           Frontend container
└── docker-compose.yml            Service orchestration
```

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | — | Health check |
| `POST` | `/auth/login` | — | Returns JWT token |
| `GET` | `/auth/me` | ✓ | Current user info |
| `GET` | `/cases` | ✓ | List cases (admin: all · lawyer: own) |
| `GET` | `/cases/{case_id}` | — | Single case metadata |
| `POST` | `/upload` | ✓ | Upload PDF + start ingestion |
| `DELETE` | `/cases/{case_id}` | ✓ | Delete case (Neo4j + disk) |
| `POST` | `/retry/{case_id}` | — | Retry failed ingestion |
| `GET` | `/resolution-state/{case_id}` | — | Entity resolution state |
| `GET` | `/staged-status/{case_id}` | — | Staged review decisions |
| `POST` | `/stage/{case_id}` | — | Stage a review decision |
| `POST` | `/confirm-all/{case_id}` | — | Commit decisions + trigger KGE |
| `POST` | `/ask/{case_id}` | ✓ | SSE streaming query |
| `POST` | `/deep-research/{case_id}` | ✓ | SSE deep research report |
| `GET` | `/pdf/{case_id}/{filename}` | — | Serve PDF file |
| `GET` | `/domains` | — | Available domain configurations |
| `GET/POST/DELETE` | `/annotations/{case_id}` | — | Post-it annotations |
| `POST` | `/auth/signup` | — | Self-service registration |
| `POST` | `/auth/verify-email` | — | Email verification |
| `POST` | `/auth/forgot-password` | — | Password reset request |
| `POST` | `/auth/reset-password` | — | Password reset confirm |
| `GET/PUT` | `/auth/profile` | ✓ | User profile |
| `POST` | `/cases/{case_id}/documents` | ✓ | Add documents to existing case |
| `GET` | `/kge-status/{case_id}` | ✓ | KGE training status |
| `GET` | `/conversation/{case_id}` | ✓ | Conversation history |

---

## Production Status

| Component | Status |
|-----------|--------|
| Ingestion pipeline | ✅ End-to-end working |
| SHACL validation | ✅ Wired into pipeline |
| Knowledge graph (case_id partitioned) | ✅ Working |
| Four-path retrieval + GAR fusion | ✅ Working |
| Entity resolution + human review | ✅ Working |
| Deep Research (9-section report) | ✅ Working |
| SSE streaming | ✅ Working |
| PDF citation buttons + inline viewer | ✅ Working |
| Post-it annotation system | ✅ Working |
| JWT auth + case ownership | ✅ Working |
| Delete case + retry ingestion | ✅ Working |
| Docker deployment (DO BLR1) | ✅ Live at 168.144.86.77 |
| Multi-file upload (PDF/DOCX/email/image/TXT) | ✅ Working |
| Document date extraction + provenance | ✅ Working |
| Phase 3 incremental entity resolution | ✅ Working |
| Light theme (Noto Serif + Manrope + warm cream) | ✅ Working |
| Timeline reconstruction endpoint | ✅ Working |
| HTTPS / Let's Encrypt | ⏳ Pending domain purchase |
| Rate limiting | ⏳ Pending |
| PDF upload size limit (50MB) | ⏳ Pending |
| Neo4j upgrade (Free → Professional) | ⏳ Pending |
| Proper password hashing (argon2) | ⏳ Pending |
| MongoDB migration (JSON → DB) | ⏳ Pending |
| Cross-case intelligence | ⏳ Pending |
| GLiNER fine-tune on Indian legal data | 🔜 Planned |
| Self-service signup + email verification | 🔜 In progress |
| Law Research mode (Indian legal LLM) | 🔜 Planned |

---

## Known Issues

**passlib bcrypt crash on Python 3.13** — passlib's bcrypt runs a self-test at import that fails with `ValueError: password cannot be longer than 72 bytes`. auth.py uses plain string comparison as a workaround. Replace with argon2-cffi before production scale.

**Neo4j AuraDB Free tier** — 200k node / 400k relationship limit. Sleeps after inactivity. Upgrade to Professional before onboarding more than 3–4 firms.

**FAISS dimension mismatch** — If you see `assert d == self.d` in logs, the FAISS index was built with a different model. Delete `cases/{case_id}/embeddings/` and run KGE training again. The index must be built and queried in the same 128-dim KGE space — not sentence-transformer space (384-dim).

**Neo4j property warnings on pre-Phase-4 cases** — Cases ingested before Phase 4 deployment will generate `UnknownPropertyKeyWarning` for `documentDate` and `docUploadOrder`. These are warnings not errors — queries still return correct results. Re-ingest the case to stamp all relationships with provenance data.

---

## Deployment

```bash
# Server: Digital Ocean · Ubuntu 24.04 · 4vCPU · 8GB · Bangalore BLR1

# First deploy
git clone https://github.com/Pegasus167/Verdictra-Law.ai.git /opt/verdictra
cd /opt/verdictra && nano .env
docker compose build && docker compose up -d

# Update after push
git pull && docker compose build backend && docker compose up -d

# Logs
docker compose logs backend -f
docker compose ps
curl http://localhost:8000/health

# Once DNS is live
# curl https://verdictra.ai/health

```

---

<div align="center">

**Verdictra** · Legal Intelligence Platform · Built for Indian Law

*Grounded in evidence. Verified by you.*

</div>