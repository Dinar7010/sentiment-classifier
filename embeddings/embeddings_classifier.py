import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import joblib

df = pd.read_csv('C:\\Users\\User\PyCharmMiscProject\sentiment-project\sentiment_dataset.csv')  # 3 столбца, 290458 строк, label - 3 знач (0-нейтральный, 1-позитивный, 2-негативный)
dfs = df.sample(n=5000, random_state=42)

ENCODER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
X = model.encode(dfs['text'].tolist(), show_progress_bar=True)
y = dfs['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = LinearSVC()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

print(classification_report(y_test, pred))
print(accuracy_score(y_test, pred))

joblib.dump(clf, Path(__file__).parent / "embeddings_clf.pkl")
print("Классификатор сохранён")