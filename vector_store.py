import os
import pickle

import faiss
import numpy as np
import threading

from flask import current_app

class VectorStore:
    def __init__(self, index_path: str, emb_dim: int):
        self.index_path = index_path
        self.emb_dim = emb_dim
        self.map_path   = self.index_path + '.map'
        self.index      = None
        self.id_map     = []
        self._lock = threading.Lock() # For thread-safety
        # now load or build, knowing we're already in an app context
        self._load_or_build()

    def _load_or_build(self):
        with self._lock:
            if os.path.exists(self.index_path) and os.path.exists(self.map_path):
                # Load existing index and ID mapping
                self.index  = faiss.read_index(self.index_path)
                with open(self.map_path, 'rb') as f:
                    self.id_map = pickle.load(f)
                current_app.logger.info(f"Loaded FAISS index from {self.index_path}")
            else:
                current_app.logger.info(f"FAISS index not found, building new index.")
                self._build_index()

            current_app.logger.info(f"FAISS store: index={self.index_path}  map={self.map_path}")

    def _build_index(self):
        # delayed import to avoid circular
        from models import Report
        # 1. Fetch all embeddings from your Report model
        # 1. Load all saved embeddings
        rows = Report.query.with_entities(Report.id, Report.img_emb).all()
        embs, self.id_map = [], []
        for rid, emb_blob in rows:
             # already an array, just need to move and type
            if emb_blob is not None:
                arr = np.array(emb_blob, dtype='float32')
                embs.append(arr)
                self.id_map.append(rid)

        if embs:
            mat = np.stack(embs)
            dim = mat.shape[1]
            # 2. Normalize if you used cosine similarity before
            faiss.normalize_L2(mat)
            
            # 💡 HNSW for efficient inner‐product search
            hnsw = faiss.IndexHNSWFlat(dim, 32)       # 32 is the number of neighbors in the graph (M)
            hnsw.hnsw.efConstruction = 40            # control index build speed vs accuracy
            hnsw.hnsw.efSearch       = 64            # control query speed vs recall
            self.index = hnsw
            self.index.add(mat)
        
        else:
            # no embeddings yet: build an empty index with known dimension from constructor
            dim = self.emb_dim
            hnsw_empty = faiss.IndexHNSWFlat(dim, 32)
            hnsw_empty.hnsw.efConstruction = 40
            hnsw_empty.hnsw.efSearch       = 64
            self.index = hnsw_empty

       
            

        # 3. Persist index + mapping
        faiss.write_index(self.index, self.index_path)
        with open(self.map_path, 'wb') as f:
            pickle.dump(self.id_map, f)

    def add(self, emb: np.ndarray, obj_id: int, duplicate_threshold: float = 0.90) -> bool:
        """
        Adds an embedding to the FAISS index and persists the changes.
        Handles lazy index rebuild if the store was initially empty.
        Performs duplicate checking and returns True if the item is new, False if it's a duplicate.
        """
        with self._lock:
            return self._do_add(emb, obj_id, duplicate_threshold)

    def _do_add(self, emb: np.ndarray, obj_id: int, duplicate_threshold: float) -> bool:
        """Internal method for adding an embedding, assumes lock is held."""
        # emb is expected to be a 1D float32 array
        if not self.id_map:
            # rebuild the index at the right dimension if it was empty
            dim = emb.shape[0]
            # Use the same HNSW parameters as in _build_index
            idx = faiss.IndexHNSWFlat(dim, 32)       # M
            idx.hnsw.efConstruction = 40            # efConstruction
            idx.hnsw.efSearch       = 64            # efSearch
            self.index = idx
            current_app.logger.info(f"FAISS index rebuilt with dimension {dim} on first add.")

        # Duplicate check
        dups = self.search(emb, k=3) # Search with the raw embedding, normalization is handled in search
        current_app.logger.info(f"FAISS duplicate check results: {dups}")
        if dups:
            top_id, top_score = dups[0]
            current_app.logger.info(f"FAISS-dup-check: against #{top_id} score={top_score:.3f}")
            if top_score >= duplicate_threshold:
                return False # It's a duplicate

        vec = emb.reshape(1, -1) # Reshape to (1, dim) for FAISS add
        faiss.normalize_L2(vec)
        self.index.add(vec)
        self.id_map.append(obj_id)
        # Persist both
        faiss.write_index(self.index, self.index_path)
        with open(self.map_path, 'wb') as f:
            pickle.dump(self.id_map, f)
        current_app.logger.info(f"Added embedding for object ID {obj_id} to FAISS index.")
        return True # Item was new and added

    def search(self, query_emb, k=5):
        with self._lock:
            return self._do_search(query_emb, k)

    def _do_search(self, query_emb, k=5):
        """Internal method for searching, assumes lock is held."""
        q = np.array(query_emb, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q) # Normalize query embedding
        distances, indices = self.index.search(q, k)
        results = [(self.id_map[idx], float(dist)) for dist, idx in zip(distances[0], indices[0]) if idx >= 0]
        return results
