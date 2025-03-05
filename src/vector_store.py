import faiss
import numpy as np
from sklearn.cluster import KMeans
import config

class VectorStore:
    def __init__(self):
        self.dimension = config.EMBEDDING_DIM
        self.index = None
        self.kmeans = None
        self.cluster_centers = None

    def build_index(self, embeddings: np.ndarray):
        """Build FAISS index from embeddings."""
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        # Perform clustering
        self.kmeans = KMeans(n_clusters=config.N_CLUSTERS, random_state=42)
        cluster_assignments = self.kmeans.fit_predict(embeddings)
        self.cluster_centers = self.kmeans.cluster_centers_
        
        return cluster_assignments

    def search(self, query_embedding: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors."""
        if self.index is None:
            raise ValueError("Index not built yet")

        # Normalize query vector
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, k)
        
        return distances, indices

    def save_index(self, path: str):
        """Save FAISS index to disk."""
        if self.index is not None:
            faiss.write_index(self.index, str(path))

    def load_index(self, path: str):
        """Load FAISS index from disk."""
        self.index = faiss.read_index(str(path)) 