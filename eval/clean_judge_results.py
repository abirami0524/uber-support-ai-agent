import pandas as pd

INPUT_FILE = "data/llm_judge_results.csv"
OUTPUT_FILE = "data/clean_judge_results.csv"

df = pd.read_csv(INPUT_FILE)

print("Original rows:", len(df))
print("Original columns:", list(df.columns))

# Keep only rows with valid LLM judge scores
score_columns = [
    "relevance",
    "grounding",
    "helpfulness",
    "safety"
]

for col in score_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

clean = df[
    df["relevance"].between(1, 5) &
    df["grounding"].between(1, 5) &
    df["helpfulness"].between(1, 5) &
    df["safety"].between(1, 5) &
    df["overall"].isin([
        "GOOD",
        "NEEDS_IMPROVEMENT",
        "BAD"
    ])
].copy()

# Remove duplicate rows if any
clean = clean.drop_duplicates()

clean.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print("CLEAN JUDGE RESULTS")
print("=" * 70)

print("Valid judged examples:", len(clean))
print("\nOverall distribution:")
print(clean["overall"].value_counts())

print("\nAverage scores:")
print(clean[score_columns].mean())

print(f"\nSaved to: {OUTPUT_FILE}")