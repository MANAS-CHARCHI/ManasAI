from fastapi import FastAPI
from pydantic import BaseModel
from .utils import upsert_text, query_similar, generate_text

app = FastAPI()

class Sample(BaseModel):
    text_id: str
    text: str
    meta: dict = {}

class Query(BaseModel):
    prompt: str

@app.post("/add_sample")
def add_sample(sample: Sample):
    upsert_text(sample.text_id, sample.text, sample.meta)
    return {"status": "ok"}

@app.post("/generate")
def generate(query: Query):
    context = query_similar(query.prompt)
    answer = generate_text(query.prompt, context)
    return {"answer": answer, "context": context}
