import faiss
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class Memory:
    def __init__(self, index_folder="memory_index"):
        self.index_folder = index_folder
        self.index_path = os.path.join(index_folder, "faiss.index")
        self.store_path = os.path.join(index_folder, "store.pkl")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        if os.path.exists(self.index_path) and os.path.exists(self.store_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.store_path, "rb") as f:
                self.text_store = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(384)  # 384 for MiniLM
            self.text_store = []

    def add_memory(self, text):
        embedding = self.model.encode([text])
        self.index.add(np.array(embedding).astype("float32"))
        self.text_store.append(text)
        self.save_index()

    def fetch_relevant(self, query, top_k=3):
        if not self.text_store:
            return []
        query_embedding = self.model.encode([query]).astype("float32")
        D, I = self.index.search(query_embedding, top_k)
        return [self.text_store[i] for i in I[0] if i < len(self.text_store)]

    def save_index(self):
        os.makedirs(self.index_folder, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.store_path, "wb") as f:
            pickle.dump(self.text_store, f)
