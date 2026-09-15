import os
import pandas as pd

# ---- Step 1: Locate the dataset files ----
path = r"C:\Users\ADMIN\.cache\kagglehub\datasets\thoughtvector\customer-support-on-twitter\versions\10"

print("Contents of dataset folder:")
for item in os.listdir(path):
    full = os.path.join(path, item)
    if os.path.isdir(full):
        print(f"  {item}  -> DIR")
        for sub in os.listdir(full):
            print(f"      {sub}")
    else:
        size_mb = os.path.getsize(full) / (1024 * 1024)
        print(f"  {item}  -> FILE ({size_mb:.1f} MB)")

# ---- Step 2: Figure out the correct path to twcs.csv ----
# Handles both cases: 'twcs' as a file without extension, or as a folder containing the csv
twcs_path = os.path.join(path, "twcs")
if os.path.isdir(twcs_path):
    # It's a folder, find the csv inside
    csv_files = [f for f in os.listdir(twcs_path) if f.endswith(".csv")]
    csv_path = os.path.join(twcs_path, csv_files[0])
elif os.path.isfile(twcs_path):
    csv_path = twcs_path
elif os.path.isfile(twcs_path + ".csv"):
    csv_path = twcs_path + ".csv"
else:
    raise FileNotFoundError(f"Could not locate twcs csv near {twcs_path}")

print(f"\nUsing CSV file: {csv_path}")

# ---- Step 3: Load into pandas ----
df = pd.read_csv(csv_path)

print(f"\nShape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 5 rows:")
print(df.head())

# ---- Step 4: Uber_Support specific stats ----
uber_as_author = df[df['author_id'] == 'Uber_Support']
print(f"\nUber_Support messages (as author, i.e. brand replies): {len(uber_as_author)}")

# Inbound customer messages that got a response FROM Uber_Support
# (response_tweet_id on the customer's row would point forward, easier to check reverse:
#  Uber_Support's rows have in_response_to_tweet_id pointing to the customer tweet)
print(f"\nSample Uber_Support reply:")
print(uber_as_author[['tweet_id', 'text', 'in_response_to_tweet_id']].head(3))