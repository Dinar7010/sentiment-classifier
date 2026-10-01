import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
df=pd.read_csv('C:\\Users\\User\PyCharmMiscProject\sentiment-project\sentiment_dataset.csv')#3 столбца,290458 строк, label - 3 знач(0-нейтральный отзыв,1-поз,2-нег), пропусков нет
X=df['text']
y=df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
vectorizer = TfidfVectorizer(max_features=20000,ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model=LinearSVC()
model.fit(X_train_vec, y_train)
prediction=model.predict(X_test_vec)
print(classification_report(y_test,prediction))
print(accuracy_score(y_test,prediction))
cm=confusion_matrix(y_test,prediction,labels=[0,1,2])
sns.heatmap(cm,annot=True,fmt='d',xticklabels=['нейтральный','позитивный','негативный'], yticklabels=['нейтральный','позитивный','негативный'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
joblib.dump(vectorizer,'vectorizer.pkl')
joblib.dump(model,'model.pkl')