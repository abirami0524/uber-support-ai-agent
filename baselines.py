from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/golden_set_TO_LABEL.csv", encoding="latin1")
labeled = df[df['intent'].notna() & (df['intent'].str.strip() != "")].copy()
labeled['intent'] = labeled['intent'].str.strip().str.lower()
labeled['customer_message'] = labeled['customer_message'].fillna("")

print(f"Total labeled rows: {len(labeled)}\n")

# ============ BASELINE 1: TRIVIAL (always predict majority class) ============
majority_class = labeled['intent'].value_counts().idxmax()
labeled['trivial_pred'] = majority_class
trivial_acc = (labeled['intent'] == labeled['trivial_pred']).mean()

print(f"=== BASELINE 1: TRIVIAL (always predict '{majority_class}') ===")
print(f"Accuracy: {trivial_acc:.2%}\n")

# ============ BASELINE 2: SIMPLE (keyword rules) ============
def keyword_classify(text):
    text = text.lower()
    if any(w in text for w in ['charge', 'refund', 'overcharg', 'fee', 'fare', 'price', 'bill']):
        return 'fare_billing_dispute'
    if any(w in text for w in ['unsafe', 'threat', 'assault', 'hit me', 'abus', 'rude', 'inappropriate']):
        return 'driver_safety_incident'
    if any(w in text for w in ['wait', 'cancel', 'no show', "didn't show", 'late', 'eta']):
        return 'wait_time_cancellation'
    if any(w in text for w in ['login', 'password', 'account', 'verif', 'hacked', 'deactivat']):
        return 'account_access'
    if any(w in text for w in ['app', 'bug', 'error', 'glitch', 'crash']):
        return 'app_technical_issue'
    if any(w in text for w in ['left my', 'lost my', 'forgot my', 'lost item']):
        return 'lost_item'
    if any(w in text for w in ['thank', 'great', 'awesome', 'love']):
        return 'compliment_other'
    if any(w in text for w in ['http', 'promo', 'free ride', 'coupon']):
        return 'spam_irrelevant'
    return 'general_complaint'  # default fallback

labeled['simple_pred'] = labeled['customer_message'].apply(keyword_classify)
simple_acc = (labeled['intent'] == labeled['simple_pred']).mean()

print(f"=== BASELINE 2: SIMPLE (keyword rules) ===")
print(f"Accuracy: {simple_acc:.2%}\n")

# ============ SUMMARY TABLE ============
llm_acc = 0.745  # from your evaluate_intent.py run — update if it changes

print("=== SUMMARY: Intent Classification Accuracy ===")
print(f"{'Method':<30} {'Accuracy':>10}")
print(f"{'-'*42}")
print(f"{'Trivial (majority class)':<30} {trivial_acc:>9.2%}")
print(f"{'Simple (keyword rules)':<30} {simple_acc:>9.2%}")
print(f"{'LLM classifier (yours)':<30} {llm_acc:>9.2%}")

labeled.to_csv("data/baseline_results.csv", index=False)
print(f"\nSaved to data/baseline_results.csv")