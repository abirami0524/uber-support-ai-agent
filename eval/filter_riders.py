import pandas as pd
import re

df = pd.read_csv("data/uber_rides_initial.csv")

driver_partner_keywords = r'\b(driver signup|drivers?\s*license|background check|vehicle inspection|partners\.uber|policy document|driver account|become a driver|drive for uber)\b'

df['likely_driver_partner'] = df['customer_text_clean'].str.contains(
    driver_partner_keywords, case=False, regex=True, na=False
)

print(f"Total: {len(df)}")
print(f"Likely driver/partner: {df['likely_driver_partner'].sum()}")

riders_only = df[~df['likely_driver_partner']].copy()
riders_only.to_csv("data/uber_riders_final.csv", index=False)
print(f"Saved {len(riders_only)} rider-only complaints to data/uber_riders_final.csv")

sample = riders_only.sample(20, random_state=7)
print("\n--- New sample of 20 ---\n")
for i, row in sample.iterrows():
    print(f"[{i}] {row['customer_text_clean']}")
    print(f"    -> UBER: {row['uber_reply_clean']}")
    print()