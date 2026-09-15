from dotenv import load_dotenv
load_dotenv()
import time
import pandas as pd
from pipeline import process_message

df = pd.read_csv("data/golden_set_TO_LABEL.csv", encoding="latin1")
labeled = df[df['intent'].notna() & (df['intent'].str.strip() != "")].copy()
print(f"Running full pipeline on {len(labeled)} golden examples...\n")

results = []
for i, row in labeled.iterrows():
    msg = row['customer_message']
    if pd.isna(msg) or str(msg).strip() == "":
        continue
    try:
        output = process_message(str(msg))
        output["true_intent"] = row['intent'].strip().lower()
        output["true_decision"] = row['decision'].strip().lower()
        output["true_reason"] = row.get('reason', '')
        results.append(output)
    except Exception as e:
        print(f"Error on row {i}: {e}")

    if len(results) % 20 == 0:
        print(f"Processed {len(results)}/{len(labeled)}...")
    time.sleep(1)  # be gentle on Groq rate limits

results_df = pd.DataFrame(results)

# ---- Metrics ----
results_df['intent_correct'] = results_df['intent'].str.lower() == results_df['true_intent']
results_df['decision_correct'] = results_df['decision'].str.lower() == results_df['true_decision']

intent_acc = results_df['intent_correct'].mean()
decision_acc = results_df['decision_correct'].mean()

print(f"\n=== FULL PIPELINE RESULTS ===")
print(f"Intent accuracy: {intent_acc:.2%}")
print(f"Escalation decision accuracy: {decision_acc:.2%}")

# Escalation precision/recall (escalate = positive class)
tp = ((results_df['decision'] == 'escalate') & (results_df['true_decision'] == 'escalate')).sum()
fp = ((results_df['decision'] == 'escalate') & (results_df['true_decision'] == 'auto_handle')).sum()
fn = ((results_df['decision'] == 'auto_handle') & (results_df['true_decision'] == 'escalate')).sum()
tn = ((results_df['decision'] == 'auto_handle') & (results_df['true_decision'] == 'auto_handle')).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\nEscalation Precision: {precision:.2%} (of predicted escalations, how many were truly escalate-worthy)")
print(f"Escalation Recall: {recall:.2%} (of true escalations, how many did we catch)")
print(f"Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

results_df.to_csv("data/full_pipeline_results.csv", index=False)
print(f"\nSaved full results to data/full_pipeline_results.csv")