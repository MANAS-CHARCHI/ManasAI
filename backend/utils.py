import openai, os
import numpy as np
from .db import load_index, save_index

openai.api_key = os.getenv("OPENAI_API_KEY")
EMB_MODEL = "text-embedding-3-small"

def get_embedding(text: str):
    resp = openai.Embedding.create(model=EMB_MODEL, input=text)
    return np.array(resp["data"][0]["embedding"], dtype="float32")

def upsert_text(text_id, text, meta):
    index, meta_store = load_index()
    emb = get_embedding(text).reshape(1, -1)
    index.add(emb)
    meta_store[len(meta_store)] = {"id": text_id, "text": text, "meta": meta}
    save_index(index, meta_store)

def query_similar(query, k=5):
    index, meta_store = load_index()
    q_emb = get_embedding(query).reshape(1, -1)
    D, I = index.search(q_emb, k)
    results = [meta_store[i] for i in I[0] if i in meta_store]
    return results

def generate_text(prompt, context):
    context_texts = "\n".join([c["text"] for c in context])
    messages = [
        {"role":"system", "content":"You are an assistant that writes in my personal tone."},
        {"role":"user", "content": f"Context from my past:\n{context_texts}\n\nNow: {prompt}"}
    ]
    resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=messages)
    return resp.choices[0].message["content"]
