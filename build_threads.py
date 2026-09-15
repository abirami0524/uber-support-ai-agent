import os
import pandas as pd

# ---- Load data ----
csv_path = r"C:\Users\ADMIN\.cache\kagglehub\datasets\thoughtvector\customer-support-on-twitter\versions\10\twcs\twcs.csv"
df = pd.read_csv(csv_path)

print(f"Total rows: {len(df)}")

# ---- Step 1: Separate customer (inbound) messages and Uber's replies ----
# inbound == True means it's from a customer TO a brand
customer_msgs = df[df['inbound'] == True].copy()
uber_replies = df[df['author_id'] == 'Uber_Support'].copy()

print(f"Customer inbound messages (all brands): {len(customer_msgs)}")
print(f"Uber_Support outbound replies: {len(uber_replies)}")

# ---- Step 2: Pair each Uber reply with the customer tweet it responded to ----
# uber_replies['in_response_to_tweet_id'] points to the customer's tweet_id
pairs = uber_replies.merge(
    customer_msgs,
    left_on='in_response_to_tweet_id',
    right_on='tweet_id',
    suffixes=('_uber', '_customer')
)

print(f"\nMatched customer->Uber reply pairs: {len(pairs)}")

# ---- Step 3: Keep only relevant columns, rename for clarity ----
pairs = pairs[[
    'tweet_id_customer', 'text_customer', 'created_at_customer',
    'tweet_id_uber', 'text_uber', 'created_at_uber',
    'in_response_to_tweet_id_customer'  # tells us if the customer msg was itself a reply (mid-thread) or a fresh complaint
]].rename(columns={
    'tweet_id_customer': 'customer_tweet_id',
    'text_customer': 'customer_text',
    'created_at_customer': 'customer_created_at',
    'tweet_id_uber': 'uber_reply_id',
    'text_uber': 'uber_reply_text',
    'created_at_uber': 'uber_reply_created_at',
    'in_response_to_tweet_id_customer': 'customer_msg_is_reply_to'
})

# ---- Step 4: Flag "fresh" initial complaints vs mid-thread follow-ups ----
# If customer_msg_is_reply_to is NaN, this was the customer's FIRST message (a fresh complaint), not a reply
pairs['is_initial_complaint'] = pairs['customer_msg_is_reply_to'].isna()

print(f"\nFresh initial complaints (customer's first message): {pairs['is_initial_complaint'].sum()}")
print(f"Mid-thread follow-up messages: {(~pairs['is_initial_complaint']).sum()}")

# ---- Step 5: Save the reconstructed pairs for labeling / building the pipeline ----
os.makedirs("data", exist_ok=True)
pairs.to_csv("data/uber_pairs.csv", index=False)
print(f"\nSaved {len(pairs)} pairs to data/uber_pairs.csv")

# ---- Step 6: Show a few examples of INITIAL complaints only (this is what we'll mostly work with) ----
initial = pairs[pairs['is_initial_complaint']]
print(f"\n--- Sample initial complaints + Uber's reply ---\n")
for _, row in initial.head(5).iterrows():
    print(f"CUSTOMER: {row['customer_text']}")
    print(f"UBER:     {row['uber_reply_text']}")
    print("-" * 80)