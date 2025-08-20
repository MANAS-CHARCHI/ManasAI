import faiss
import numpy as np
import os, pickle

INDEX_FILE = "style.index"
META_FILE = "meta.pkl"

def load_index(dim=1536):
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(META_FILE, "rb") as f:
            meta = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(dim)
        meta = {}
    return index, meta

def save_index(index, meta):
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump(meta, f)
