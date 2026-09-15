import pandas as pd
import re

df = pd.read_csv("data/uber_pairs.csv")

# Keep only initial complaints (not mid-thread follow-ups) for taxonomy building
initial = df[df['is_initial_complaint'] == True].copy()

def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'@\w+', '', text)          # remove @handles
    text = re.sub(r'https?://\S+', '', text)  # remove URLs
    text = re.sub(r'\s+', ' ', text).strip()  # collapse whitespace
    return text

initial['customer_text_clean'] = initial['customer_text'].apply(clean_text)
initial['uber_reply_clean'] = initial['uber_reply_text'].apply(clean_text)

# Rough filter: flag likely UberEats-related messages
eats_keywords = r'\b(eats|delivery|order|restaurant|food)\b'
initial['likely_eats'] = initial['customer_text_clean'].str.contains(eats_keywords, case=False, regex=True)

print(f"Total initial complaints: {len(initial)}")
print(f"Likely UberEats-related: {initial['likely_eats'].sum()}")
print(f"Likely rides-related: {(~initial['likely_eats']).sum()}")

# Save rides-only version
rides_only = initial[~initial['likely_eats']].copy()
rides_only.to_csv("data/uber_rides_initial.csv", index=False)
print(f"\nSaved {len(rides_only)} rides-related initial complaints to data/uber_rides_initial.csv")

# Print a random sample of 20 cleaned complaints for you to read and validate intent taxonomy
sample = rides_only.sample(20, random_state=42)
print("\n--- Random sample of 20 cleaned complaints ---\n")
for i, row in sample.iterrows():
    print(f"[{i}] {row['customer_text_clean']}")
    print(f"    -> UBER: {row['uber_reply_clean']}")
    print()