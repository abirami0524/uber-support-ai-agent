from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

def build_index():
    df = pd.read_csv("data/uber_riders_final.csv")
    df = df[df['customer_text_clean'].notna() & (df['customer_text_clean'].str.strip() != "")].copy()
    SAMPLE_SIZE = 5000
    if len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    print(f"Building index from {len(df)} examples...")

    try:
        chroma_client.delete_collection("uber_complaints")
    except Exception:
        pass
    collection = chroma_client.create_collection("uber_complaints")

    batch_size = 200
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        texts = batch['customer_text_clean'].tolist()
        embeddings = embedder.encode(texts).tolist()
        ids = [str(idx) for idx in batch.index]
        metadatas = [
            {"customer_text": row['customer_text_clean'],
             "uber_reply": row['uber_reply_clean'] if pd.notna(row['uber_reply_clean']) else ""}
            for _, row in batch.iterrows()
        ]
        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)
        print(f"  Indexed {min(i+batch_size, len(df))}/{len(df)}")
    return collection

# ---- Load embedder once at module level ----
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# ---- Load existing collection if it exists, else build it ----
try:
    collection = chroma_client.get_collection("uber_complaints")
    print(f"Loaded existing index with {collection.count()} entries.")
except Exception:
    print("No existing index found, building it now...")
    collection = build_index()

def retrieve_similar(query: str, k: int = 3):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    return results

if __name__ == "__main__":
    test_query = "Why was I charged a cancellation fee when the driver cancelled on me?"
    print(f"\n--- Test retrieval for: '{test_query}' ---")
    results = retrieve_similar(test_query, k=3)
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"\n[{i+1}] Similar complaint: {meta['customer_text']}")
        print(f"    Historical Uber reply: {meta['uber_reply']}")