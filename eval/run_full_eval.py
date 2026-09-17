from dotenv import load_dotenv
load_dotenv()

import time
import pandas as pd
from pipeline import process_message

# --------------------------------------------------
# Load golden dataset
# --------------------------------------------------
df = pd.read_csv(
    "data/golden_set_TO_LABEL.csv",
    encoding="latin1"
)

labeled = df[
    df['intent'].notna() &
    (df['intent'].str.strip() != "")
].copy()

# --------------------------------------------------
# Load already completed results
# --------------------------------------------------
results_file = "data/full_pipeline_results.csv"

try:
    existing_results_df = pd.read_csv(results_file)

    completed = len(existing_results_df)

    print(f"Found {completed} already completed results.")
    print(f"Resuming from row {completed + 1}/{len(labeled)}...\n")

    # Convert existing results into list of dictionaries
    results = existing_results_df.to_dict("records")

except FileNotFoundError:
    print("No previous results file found. Starting from row 1.\n")
    results = []
    completed = 0


# --------------------------------------------------
# Process only remaining rows
# --------------------------------------------------
for position, (i, row) in enumerate(
    labeled.iloc[completed:].iterrows(),
    start=completed + 1
):

    msg = row['customer_message']

    if pd.isna(msg) or str(msg).strip() == "":
        continue

    try:
        output = process_message(str(msg))

        output["true_intent"] = str(row['intent']).strip().lower()
        output["true_decision"] = str(row['decision']).strip().lower()
        output["true_reason"] = row.get('reason', '')

        results.append(output)

        print(f"Processed {position}/{len(labeled)}...")

        # --------------------------------------------------
        # SAVE AFTER EVERY SUCCESSFUL ROW
        # This prevents losing progress if Groq stops again.
        # --------------------------------------------------
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_file, index=False)

    except Exception as e:
        print(f"Error on row {i}: {e}")

        # Save whatever has already been completed
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_file, index=False)

    # Be gentle on Groq rate limits
    time.sleep(1)


# --------------------------------------------------
# Final results
# --------------------------------------------------
results_df = pd.DataFrame(results)

# --------------------------------------------------
# Metrics
# --------------------------------------------------
results_df['intent_correct'] = (
    results_df['intent'].str.lower() ==
    results_df['true_intent']
)

results_df['decision_correct'] = (
    results_df['decision'].str.lower() ==
    results_df['true_decision']
)

intent_acc = results_df['intent_correct'].mean()
decision_acc = results_df['decision_correct'].mean()

print("\n=== FULL PIPELINE RESULTS ===")
print(f"Intent accuracy: {intent_acc:.2%}")
print(f"Escalation decision accuracy: {decision_acc:.2%}")


# --------------------------------------------------
# Escalation precision / recall
# --------------------------------------------------
tp = (
    (results_df['decision'] == 'escalate') &
    (results_df['true_decision'] == 'escalate')
).sum()

fp = (
    (results_df['decision'] == 'escalate') &
    (results_df['true_decision'] == 'auto_handle')
).sum()

fn = (
    (results_df['decision'] == 'auto_handle') &
    (results_df['true_decision'] == 'escalate')
).sum()

tn = (
    (results_df['decision'] == 'auto_handle') &
    (results_df['true_decision'] == 'auto_handle')
).sum()

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0
)

recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0
)

print(
    f"\nEscalation Precision: {precision:.2%} "
    "(of predicted escalations, how many were truly escalate-worthy)"
)

print(
    f"Escalation Recall: {recall:.2%} "
    "(of true escalations, how many did we catch)"
)

print(f"Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

# --------------------------------------------------
# Final save
# --------------------------------------------------
results_df.to_csv(results_file, index=False)

print(f"\nSaved full results to {results_file}")
print(f"Total completed: {len(results_df)}/{len(labeled)}")

