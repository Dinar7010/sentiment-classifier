import requests

texts=["фильм крутой","фильм плохой","не понял суть фильма","",123]
url = "http://127.0.0.1:8000"
models = ["tfidf", "bert"]
response = requests.get(f"{url}/health",timeout=5)
print(response.status_code,response.json())
print()
for text in texts:
    for model in models:
        response = requests.post(f"{url}/predict",json={"text":text,"model":model},timeout=30)
        print(response.status_code,response.json())
        print()