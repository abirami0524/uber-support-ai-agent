from dotenv import load_dotenv
load_dotenv()
import time
import pandas as pd
from classify_intent import classify, INTENTS

df = pd.read_csv("data/golden_set_TO_LABEL.csv", encoding="latin1")
labeled = df[df['intent'].notna() & (df['intent'].str.strip() != "")].copy()
print(f"Total labeled rows: {len(labeled)}")

def classify_with_retry(msg, max_retries=5):
    for attempt in range(max_retries):
        try:
            return classify(str(msg))
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                wait = 15  # back off and wait for quota to refresh
                print(f"  Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                print(f"  Non-rate-limit error: {e}")
                return "ERROR"
    return "ERROR"

predictions = []
for i, row in labeled.iterrows():
    msg = row['customer_message']
    if pd.isna(msg) or str(msg).strip() == "":
        predictions.append("spam_irrelevant")
    else:
        pred = classify_with_retry(msg)
        predictions.append(pred)

    time.sleep(4)  # 20 req/min limit = max ~15/min to be safe -> 4s between calls

    if len(predictions) % 10 == 0:
        print(f"Processed {len(predictions)}/{len(labeled)}...")

labeled['predicted_intent'] = predictions
labeled['correct'] = labeled['intent'].str.strip().str.lower() == labeled['predicted_intent'].str.strip().str.lower()
accuracy = labeled['correct'].mean()

print(f"\n=== RESULTS ===")
print(f"Accuracy: {accuracy:.2%}")
print(f"\n--- Per-intent accuracy ---")
for intent in INTENTS:
    subset = labeled[labeled['intent'].str.strip().str.lower() == intent]
    if len(subset) > 0:
        acc = subset['correct'].mean()
        print(f"{intent}: {acc:.2%} ({len(subset)} examples)")

labeled.to_csv("data/intent_eval_results.csv", index=False)
print(f"\nSaved to data/intent_eval_results.csv")

errors = labeled[~labeled['correct']]
print(f"\n--- Sample misclassifications ({len(errors)} total) ---")
for i, row in errors.head(10).iterrows():
    print(f"MSG: {row['customer_message'][:80]}")
    print(f"  TRUE: {row['intent']}  |  PREDICTED: {row['predicted_intent']}")
    print()