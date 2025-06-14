import os
import pickle

import faiss
import numpy as np




class VectorStore:
    def __init__(self, index_path='faiss_index.bin'):
        self.index_path = index_path
        self.map_path   = index_path + '.map'
        self.index      = None
        self.id_map     = []
                # Build or load the FAISS index within Flask’s app context
        from app import app
        with app.app_context():
            self._load_or_build()

    def _load_or_build(self):
        if os.path.exists(self.index_path) and os.path.exists(self.map_path):
            # Load existing index and ID mapping
            self.index  = faiss.read_index(self.index_path)
            with open(self.map_path, 'rb') as f:
                self.id_map = pickle.load(f)
        else:
            self._build_index()

    def _build_index(self):
                # Lazy‐load Report to avoid circular imports
        from app import Report

        # 1. Fetch all embeddings from your Report model
        rows = Report.query.with_entities(Report.id, Report.img_emb).all()
        embs, self.id_map = [], []
        for rid, emb_blob in rows:
            vec = np.frombuffer(emb_blob, dtype=np.float32)
            embs.append(vec)
            self.id_map.append(rid)

        if embs:
            mat = np.stack(embs)
            # 2. Normalize if you used cosine similarity before
            faiss.normalize_L2(mat)
            dim = mat.shape[1]
            # 3. Create an inner-product index (cosine on normalized vectors)
                    # 3. Create an HNSW index for inner-product (cosine on normalized vectors)
            hnsw = faiss.IndexHNSWFlat(dim, 32)       # 32 is the number of neighbors in the graph (M)
            hnsw.hnsw.efConstruction = 40            # control index build speed vs accuracy
            hnsw.hnsw.efSearch       = 64            # control query speed vs recall
            self.index = hnsw
            self.index.add(mat)

        else:
                # Empty HNSW index placeholder (will rebuild dimension on first add)
            hnsw_empty = faiss.IndexHNSWFlat(1, 32)
            hnsw_empty.hnsw.efConstruction = 40
            hnsw_empty.hnsw.efSearch       = 64
            self.index = hnsw_empty

            

        # 4. Persist index + mapping
        faiss.write_index(self.index, self.index_path)
        with open(self.map_path, 'wb') as f:
            pickle.dump(self.id_map, f)

    def search(self, query_emb, k=5):
        q = np.array(query_emb, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        distances, indices = self.index.search(q, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                # no more real vectors
                break
            results.append((self.id_map[idx], float(dist)))
        return results

