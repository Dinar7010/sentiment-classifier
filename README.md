# Sentiment Classifier — классификация тональности отзывов

Сравнение трёх подходов к классификации тональности текстовых отзывов на русском языке
(нейтральный / положительный / отрицательный): от классического ML до fine-tuning трансформера + REST API.

## Задача

На вход подаётся датасет с текстовыми отзывами. Для каждого отзыва определяется класс тональности:

- `0` — нейтральный
- `1` — позитивный
- `2` — негативный

## Данные

[Russian Sentiment Dataset](https://www.kaggle.com/datasets/mar1mba/russian-sentiment-dataset) с Kaggle —
290 458 отзывов из 11 источников (отзывы на одежду, фильмы, аниме, банки, новости и др.).

**Оговорка:** официальное описание значений меток на странице датасета отсутствует.
Соответствие классов 0/1/2 определено эмпирически.

## Подходы и результаты

| Подход | Файл | Данные | Accuracy                           |
|---|---|---|------------------------------------|
| TF-IDF + LinearSVC | `classical/sentiment.py` | 290 458 строк | **0.70**                           |
| Sentence embeddings + LinearSVC | `embeddings/embeddings_classifier.py` | 5 000 строк | 0.637                              |
| Fine-tuned RuBERT | `bert_finetuning/finetune_bert.py` | 1 000 строк | 0.6-0.7 (в запусках 0.615 и 0.675) |

### Выводы

- Классический TF-IDF-подход оказался сильнее готовых sentence-embeddings на этой конкретной задаче —
  вероятно, потому что тональность здесь часто определяется конкретными словами-маркерами
  ("ужасно", "прекрасно"), которые TF-IDF отлично улавливает, а общая семантическая близость,
  которую ищут embeddings, не всегда с этим коррелирует.
- Fine-tuning RuBERT показал результат, близкий к embeddings-подходу, при этом обучался
  всего на **1000** примерах — в 5 раз меньше, чем embeddings-эксперимент. Это иллюстрирует
  главное преимущество fine-tuning: эффективное использование малых объёмов данных за счёт
  переноса знаний из предобучения.
- На большем объёме данных fine-tuning, скорее всего, обошёл бы оба других подхода — это
  не проверено в рамках эксперимента из-за ограничений по времени обучения на CPU.

## Технологии

- Python, pandas
- scikit-learn (`TfidfVectorizer`, `LinearSVC`, метрики)
- sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`)
- PyTorch, HuggingFace `transformers` (`DeepPavlov/rubert-base-cased`)
- matplotlib, seaborn — визуализация
- FastAPI, uvicorn, pydantic — REST API, joblib — сохранение моделей

## Как запустить

1. Установить зависимости:

```bash
pip install fastapi uvicorn joblib scikit-learn pandas torch transformers sentence-transformers requests matplotlib seaborn
```

2. Скачать датасет с [Kaggle](https://www.kaggle.com/datasets/mar1mba/russian-sentiment-dataset)
   и положить `sentiment_dataset.csv` в корень проекта.
3. Обучить BERT (один раз, на CPU занимает несколько минут):

```bash
python bert_finetuning/finetune_bert.py
```

4. Запустить API:

```bash
python classical/run_server.py
```

Сервер стартует на `http://127.0.0.1:8000`, документация на `/docs`.


## API
### Эндпоинты: GET / (приветствие), GET /health (состояние и список моделей), POST /predict (тело: text и  model, одна из tfidf, bert, embeddings, по умолчанию стоит tfidf).

### Пример запроса
```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"text": "фильм крутой", "model": "bert"},
)
print(response.json())
```

Ответ:

```json
{"text": "фильм крутой", "model": "bert", "label": 1, "sentiment": "позитивный"}
```


## Воспроизведение
Готовые `.pkl` для TF-IDF и embeddings лежат в репозитории (они маленькие),
поэтому переобучать их не обязательно, но кодировщик embeddings (paraphrase-multilingual-MiniLM-L12-v2) скачивается автоматически при первом старте сервера, поэтому нужен интернет, и первый запуск будет дольше.

Модель BERT в репозиторий не входит: папка `bert_finetuning/saved_model/`
весит сотни мегабайт и добавлена в `.gitignore`. Сервер ищет её при старте,
поэтому перед первым запуском API нужно один раз выполнить:

```bash
python bert_finetuning/finetune_bert.py
```

Остальные скрипты нужны только для переобучения моделей.


## Ограничения

Датасет составлен из 11 разнородных источников, каждый со своей стилистикой текста.
Часть отзывов содержит смешанную тональность (например, положительную оценку сервиса
при негативной оценке товара в рамках одного текста) — это ограничивает потолок точности
для всех трёх подходов. Embeddings- и BERT-эксперименты проводились на существенно меньшей
подвыборке данных из-за ограничений по времени обучения — прямое сравнение с TF-IDF (290k строк)
не полностью честное по этой причине.