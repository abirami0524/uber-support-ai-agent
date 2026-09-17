import pandas as pd

FILE = "data/full_pipeline_results.csv"

df = pd.read_csv(FILE)

# False negatives:
# Golden Set says escalate, pipeline says auto_handle
fn = df[
    (df["true_decision"] == "escalate") &
    (df["decision"] == "auto_handle")
].copy()

print("=" * 70)
print("FALSE NEGATIVE ANALYSIS")
print("=" * 70)

print(f"\nTotal Golden Set examples : {len(df)}")
print(f"False negatives           : {len(fn)}")

# --------------------------------------------------
# Intent distribution
# --------------------------------------------------

print("\nTRUE INTENT DISTRIBUTION")
print("-" * 70)

print(
    fn["true_intent"]
    .value_counts()
    .to_string()
)

# --------------------------------------------------
# Predicted intent distribution
# --------------------------------------------------

print("\nPREDICTED INTENT DISTRIBUTION")
print("-" * 70)

print(
    fn["intent"]
    .value_counts()
    .to_string()
)

# --------------------------------------------------
# Correct vs incorrect intent
# --------------------------------------------------

fn["intent_correct"] = (
    fn["true_intent"] == fn["intent"]
)

print("\nINTENT CLASSIFICATION")
print("-" * 70)

print(
    fn["intent_correct"]
    .value_counts()
    .rename({
        True: "Intent correct",
        False: "Intent incorrect"
    })
    .to_string()
)

# --------------------------------------------------
# Parser failures
# --------------------------------------------------

parser_failures = fn[
    fn["reason"]
    .astype(str)
    .str.contains(
        "Could not parse model output",
        case=False,
        na=False
    )
]

print("\nPARSER / FALLBACK FAILURES")
print("-" * 70)

print(
    f"Parser fallback cases: {len(parser_failures)}"
)

# --------------------------------------------------
# Intent + decision combinations
# --------------------------------------------------

print("\nTRUE INTENT → PREDICTED INTENT")
print("-" * 70)

intent_pairs = (
    fn.groupby(
        ["true_intent", "intent"]
    )
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print(intent_pairs.to_string(index=False))

# --------------------------------------------------
# Save false negatives
# --------------------------------------------------

output_columns = [
    "message",
    "true_intent",
    "intent",
    "true_decision",
    "decision",
    "reason"
]

fn[output_columns].to_csv(
    "data/false_negative_cases.csv",
    index=False
)

print("\n" + "=" * 70)
print("Saved:")
print("data/false_negative_cases.csv")
print("=" * 70)