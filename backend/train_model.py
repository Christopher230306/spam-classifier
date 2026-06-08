import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "SMSSpamCollection"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"])
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    (
        "vectorizer",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000
        )
    ),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n")
print(report)
print("Confusion Matrix:\n", matrix)

joblib.dump(model, MODEL_DIR / "spam_classifier.pkl")
print("\nModel saved to backend/models/spam_classifier.pkl")

sample_messages = [
    "Congratulations! You have won a free iPhone. Call now.",
    "Hey, are you coming to class today?",
    "Urgent! Claim your prize money now.",
    "Can you send me the notes after lunch?",
    "You won a car"
]

print("\nSample Predictions:")
for msg in sample_messages:
    prediction = model.predict([msg])[0]
    probabilities = model.predict_proba([msg])[0]
    confidence = max(probabilities)
    print(f"{msg} --> {prediction} ({confidence:.3f})")