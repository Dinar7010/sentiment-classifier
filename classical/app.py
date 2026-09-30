from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
import joblib
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel

BASE_DIR = Path(__file__).parent
BERT_DIR = BASE_DIR.parent / "bert_finetuning" / "saved_model"
labels={0:"нейтральный",1:"позитивный",2:"негативный"}

models={}
@ asynccontextmanager
async def lifespan(app: FastAPI):
    models["vectorizer"]=joblib.load(BASE_DIR / "vectorizer.pkl")
    models["tfidf"]=joblib.load(BASE_DIR / "model.pkl")
    models["bert_tokenizer"]=AutoTokenizer.from_pretrained(str(BERT_DIR))
    bert=AutoModelForSequenceClassification.from_pretrained(str(BERT_DIR))
    bert.eval()
    models["bert"] = bert
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)

class TextRequest(BaseModel):
    text: str
    model: Literal["tfidf", "bert"] = "tfidf"

def predict_tfidf(text: str) -> int:
    X = models["vectorizer"].transform([text])
    return int(models["tfidf"].predict(X)[0])

def predict_bert(text: str) -> int:
    inputs=models["bert_tokenizer"](text,return_tensors="pt",truncation=True,max_length=128)
    with torch.no_grad():
        logits = models["bert"](**inputs).logits
    return int(logits.argmax(dim=-1))

@app.get("/")
def read_root():
    return {"message": "Sentiment classifier API работает"}

@app.get("/health")
def health():
    return {"status": "ok",
            "models": ["tfidf", "bert"],
            "loaded":list(models.keys()),
            "len":len(models["vectorizer"].vocabulary_)}

@app.post("/predict")
def predict(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")
    if request.model == "tfidf":
        label=predict_tfidf(request.text)
    else:
        label=predict_bert(request.text)
    return{"text": request.text,"model":request.model,"label": label,"sentiment": labels[label],}