import joblib
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
app = FastAPI()
vectorizer = joblib.load('vectorizer.pkl')
model = joblib.load('model.pkl')
labels={0:"нейтральный",1:"позитивный",2:"негативный"}
class TextRequest(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Sentiment classifier API работает"}

@app.get("/health")
def health():
    return {"status": "ok",
            "model": type(model).__name__,
            "classes":[int(c)for c in model.classes_],
            "len":len(vectorizer.vocabulary_)}

@app.get("/LABELS")
def read_labels():
    return {"LABELS": labels}

@app.post("/predict")
def predict(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")
    X = vectorizer.transform([request.text])
    prediction = model.predict(X)
    label = int(prediction[0])
    return{"text": request.text,"label": label,"sentiment": labels[label],}