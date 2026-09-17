import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

FILE = "data/human_agreement_sample_completed.csv"

df = pd.read_csv(FILE)

# --------------------------------------------------
# Compare Human vs Gemini
# --------------------------------------------------

print("=" * 70)
print("HUMAN vs GEMINI AGREEMENT")
print("=" * 70)

print(f"Rows evaluated: {len(df)}")


# --------------------------------------------------
# Overall agreement
# --------------------------------------------------

overall_match = (
    df["human_overall"].str.upper()
    == df["overall"].str.upper()
)

overall_agreement = overall_match.mean() * 100

print("\nOVERALL")
print("-" * 70)

print(f"Agreement: {overall_agreement:.2f}%")

# Cohen's Kappa
kappa = cohen_kappa_score(
    df["human_overall"].str.upper(),
    df["overall"].str.upper()
)

print(f"Cohen's Kappa: {kappa:.3f}")


# --------------------------------------------------
# Individual dimensions
# --------------------------------------------------

dimensions = [
    ("Relevance", "human_relevance", "relevance"),
    ("Grounding", "human_grounding", "grounding"),
    ("Helpfulness", "human_helpfulness", "helpfulness"),
    ("Safety", "human_safety", "safety"),
]

print("\nDIMENSION AGREEMENT")
print("-" * 70)

for name, human_col, gemini_col in dimensions:

    matches = (
        df[human_col] == df[gemini_col]
    )

    agreement = matches.mean() * 100

    kappa = cohen_kappa_score(
        df[human_col],
        df[gemini_col]
    )

    print(
        f"{name:<15} "
        f"Agreement: {agreement:6.2f}%   "
        f"Kappa: {kappa:.3f}"
    )


# --------------------------------------------------
# Overall confusion matrix
# --------------------------------------------------

labels = [
    "GOOD",
    "NEEDS_IMPROVEMENT",
    "BAD"
]

matrix = confusion_matrix(
    df["human_overall"].str.upper(),
    df["overall"].str.upper(),
    labels=labels
)

print("\nOVERALL CONFUSION MATRIX")
print("-" * 70)

print(
    pd.DataFrame(
        matrix,
        index=[f"Human: {x}" for x in labels],
        columns=[f"Gemini: {x}" for x in labels]
    )
)


# --------------------------------------------------
# Rows where Human and Gemini disagree
# --------------------------------------------------

disagreements = df[
    df["human_overall"].str.upper()
    != df["overall"].str.upper()
]

print("\nDISAGREEMENTS")
print("-" * 70)

print(f"Number of disagreements: {len(disagreements)}")

if len(disagreements) > 0:

    print(
        disagreements[
            [
                "row_index",
                "message",
                "overall",
                "human_overall",
                "judge_reason",
                "human_notes"
            ]
        ].to_string(index=False)
    )


# --------------------------------------------------
# Save agreement results
# --------------------------------------------------

summary = {
    "metric": [
        "Overall Agreement",
        "Overall Cohen Kappa",
        "Relevance Agreement",
        "Relevance Cohen Kappa",
        "Grounding Agreement",
        "Grounding Cohen Kappa",
        "Helpfulness Agreement",
        "Helpfulness Cohen Kappa",
        "Safety Agreement",
        "Safety Cohen Kappa",
    ],
    "value": [
        overall_agreement,
        kappa,

        (
            (df["human_relevance"] == df["relevance"]).mean()
            * 100
        ),

        cohen_kappa_score(
            df["human_relevance"],
            df["relevance"]
        ),

        (
            (df["human_grounding"] == df["grounding"]).mean()
            * 100
        ),

        cohen_kappa_score(
            df["human_grounding"],
            df["grounding"]
        ),

        (
            (df["human_helpfulness"] == df["helpfulness"]).mean()
            * 100
        ),

        cohen_kappa_score(
            df["human_helpfulness"],
            df["helpfulness"]
        ),

        (
            (df["human_safety"] == df["safety"]).mean()
            * 100
        ),

        cohen_kappa_score(
            df["human_safety"],
            df["safety"]
        ),
    ]
}

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    "data/human_gemini_agreement_summary.csv",
    index=False
)

print("\n" + "=" * 70)
print("AGREEMENT ANALYSIS COMPLETE")
print("=" * 70)

print(
    "\nSaved summary to:"
    "\ndata/human_gemini_agreement_summary.csv"
)