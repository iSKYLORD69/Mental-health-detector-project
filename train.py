import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ── Load data ──────────────────────────────────────────────
df = pd.read_csv("data/mental_health.csv")

# Drop rows where text or label is missing
df = df.dropna(subset=["statement", "status"])

print(f"Total samples: {len(df)}")
print(f"Label distribution:\n{df['status'].value_counts()}\n")

# ── Prepare features and labels ────────────────────────────
X = df["statement"]
y = df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Build and train the model pipeline ─────────────────────
# Pipeline chains TF-IDF → Logistic Regression automatically
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=15000,   # only keep top 15k words
        ngram_range=(1, 2),   # use single words AND word pairs
        stop_words="english"  # ignore "the", "is", "and" etc.
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        C=1.0,                # regularization — lower = simpler model
        class_weight="balanced"  # handles unequal label counts
    ))
])

print("Training model...")
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ── Save the model ─────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
print("Model saved to models/model.pkl")