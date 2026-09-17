import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


FILE = "data/full_pipeline_results.csv"

df = pd.read_csv(FILE)

print("=" * 70)
print("SIMPLE BASELINE - TF-IDF + LOGISTIC REGRESSION")
print("=" * 70)

# --------------------------------------------------
# INTENT CLASSIFICATION
# --------------------------------------------------

X = df["message"].fillna("")
y = df["true_intent"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
    
)

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

intent_accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nINTENT BASELINE")
print("-" * 70)
print(f"Test examples: {len(X_test)}")
print(
    f"Accuracy: {intent_accuracy:.4f} "
    f"({intent_accuracy * 100:.2f}%)"
)


# --------------------------------------------------
# DECISION BASELINE
# --------------------------------------------------
# For a simple decision baseline, use a rule-based
# safety/escalation detector.

def simple_decision(message):

    message = str(message).lower()

    escalation_keywords = [
        "safety",
        "unsafe",
        "fraud",
        "unauthorized",
        "charged twice",
        "stolen",
        "lost phone",
        "disabled account",
        "can't login",
        "cannot login",
        "security",
        "threat",
        "harassment",
        "accident",
        "emergency",
        "urgent"
    ]

    for keyword in escalation_keywords:
        if keyword in message:
            return "escalate"

    return "auto_handle"


decision_predictions = df["message"].apply(
    simple_decision
)

decision_accuracy = accuracy_score(
    df["true_decision"],
    decision_predictions
)

print("\nDECISION BASELINE")
print("-" * 70)
print(
    f"Accuracy: {decision_accuracy:.4f} "
    f"({decision_accuracy * 100:.2f}%)"
)

print("\n" + "=" * 70)
print("SIMPLE BASELINE SUMMARY")
print("=" * 70)

print(
    f"Intent accuracy   : "
    f"{intent_accuracy * 100:.2f}%"
)

print(
    f"Decision accuracy : "
    f"{decision_accuracy * 100:.2f}%"
)