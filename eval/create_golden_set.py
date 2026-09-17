import pandas as pd

df = pd.read_csv("data/uber_riders_final.csv")

# ---- Sample 200 rows for labeling ----
# random_state fixed for reproducibility (mention this in your decision log)
golden = df.sample(200, random_state=42).copy()

# ---- Set up empty columns for you to hand-label ----
golden['intent'] = ""            # fill with one of the 8 categories below
golden['decision'] = ""          # fill with: auto_handle  OR  escalate
golden['reason'] = ""            # 1 short sentence: why this decision
golden['ideal_reply'] = ""       # optional but valuable: write what YOU think a good reply would say
golden['notes'] = ""             # anything ambiguous/weird about this example

# ---- Keep only what's needed for labeling (drop clutter columns) ----
golden_labeling = golden[[
    'customer_tweet_id', 'customer_text_clean', 'uber_reply_clean',
    'intent', 'decision', 'reason', 'ideal_reply', 'notes'
]].rename(columns={
    'customer_text_clean': 'customer_message',
    'uber_reply_clean': 'actual_uber_reply'
})

golden_labeling.to_csv("data/golden_set_TO_LABEL.csv", index=False)
print(f"Saved {len(golden_labeling)} rows to data/golden_set_TO_LABEL.csv")
print("\nOpen this file in Excel or VS Code's CSV viewer and fill in: intent, decision, reason, ideal_reply, notes")

print("""
--- INTENT REFERENCE (use exactly these labels) ---
1. fare_billing_dispute
2. driver_safety_incident
3. wait_time_cancellation
4. account_access
5. app_technical_issue
6. general_complaint
7. spam_irrelevant
8. compliment_other

--- DECISION REFERENCE ---
auto_handle   -> low-risk, policy-driven, safe for AI to handle directly
escalate      -> needs human judgment (safety, high-value disputes, legal/property damage, angry/vague cases needing more context)
""")