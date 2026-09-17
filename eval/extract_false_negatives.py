import pandas as pd

# Load pipeline results
results = pd.read_csv(
    "data/full_pipeline_results.csv"
)

# Find false negatives:
# Ground truth = escalate
# Model prediction = auto_handle
false_negatives = results[
    (results["true_decision"].str.lower() == "escalate") &
    (results["decision"].str.lower() == "auto_handle")
].copy()

print(f"False negatives found: {len(false_negatives)}")

# Keep the most useful columns
columns = [
    "message",
    "true_intent",
    "intent",
    "true_decision",
    "decision",
    "true_reason",
    "reason",
    "drafted_reply",
    "grounding_examples"
]

false_negatives = false_negatives[columns]

# Save
output_file = "data/false_negatives.csv"

false_negatives.to_csv(
    output_file,
    index=False
)

print(f"Saved to: {output_file}")

# Display them
print("\n=== FALSE NEGATIVE EXAMPLES ===\n")

for i, row in false_negatives.iterrows():
    print("=" * 100)
    print(f"Example {i + 1}")
    print(f"\nCustomer message:\n{row['message']}")
    print(f"\nTrue intent:       {row['true_intent']}")
    print(f"Predicted intent:  {row['intent']}")
    print(f"True decision:     {row['true_decision']}")
    print(f"Predicted decision:{row['decision']}")
    print(f"True reason:       {row['true_reason']}")
    print(f"Model reason:      {row['reason']}")
    print(f"\nDrafted reply:\n{row['drafted_reply']}")

