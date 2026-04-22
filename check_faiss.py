import sys
sys.stdout.reconfigure(encoding='utf-8')
import faiss
import pickle
from pathlib import Path

faiss_path = Path('cases/celir_llp_vs_midc/embeddings/graph_entities.faiss')
entity_path = Path('cases/celir_llp_vs_midc/embeddings/entity_map.pkl')

if faiss_path.exists():
    index = faiss.read_index(str(faiss_path))
    print(f'FAISS index dimension: {index.d}')
    print(f'FAISS total vectors: {index.ntotal}')
else:
    print('No FAISS index found')

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
test_vec = model.encode(['test'], normalize_embeddings=True)
print(f'Sentence transformer dimension: {test_vec.shape[1]}')
