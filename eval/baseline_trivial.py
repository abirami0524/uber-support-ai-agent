import pandas as pd

FILE = "data/full_pipeline_results.csv"

df = pd.read_csv(FILE)

print("=" * 70)
print("TRIVIAL BASELINE")
print("=" * 70)

# --------------------------------------------------
# INTENT BASELINE
# Always predict the most common intent
# --------------------------------------------------

majority_intent = df["true_intent"].mode()[0]

df["baseline_intent"] = majority_intent

intent_accuracy = (
    df["baseline_intent"] == df["true_intent"]
).mean()

print("\nINTENT BASELINE")
print("-" * 70)
print("Majority intent:", majority_intent)
print(f"Accuracy: {intent_accuracy:.4f} ({intent_accuracy * 100:.2f}%)")


# --------------------------------------------------
# DECISION BASELINE
# Always predict the most common decision
# --------------------------------------------------

majority_decision = df["true_decision"].mode()[0]

df["baseline_decision"] = majority_decision

decision_accuracy = (
    df["baseline_decision"] == df["true_decision"]
).mean()

print("\nDECISION BASELINE")
print("-" * 70)
print("Majority decision:", majority_decision)
print(
    f"Accuracy: {decision_accuracy:.4f} "
    f"({decision_accuracy * 100:.2f}%)"
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 70)
print("TRIVIAL BASELINE SUMMARY")
print("=" * 70)

print(f"Intent accuracy   : {intent_accuracy * 100:.2f}%")
print(f"Decision accuracy : {decision_accuracy * 100:.2f}%")