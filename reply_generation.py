from dotenv import load_dotenv
load_dotenv()
import os
from groq import Groq
from src_retrieval import retrieve_similar

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

REPLY_PROMPT_TEMPLATE = """You are a customer support agent for Uber, responding to a rider's message on Twitter.

Here are examples of how Uber has historically responded to similar rider complaints:
{examples}

Now, write a helpful, empathetic, and specific reply to this NEW rider message. Keep it concise (1-3 sentences), professional, and in Uber's typical tone. Do not just copy a generic template — try to be genuinely responsive to the specifics of this message where possible.

New rider message: "{message}"

Reply:"""

def generate_reply(message: str, k: int = 3) -> dict:
    retrieval_results = retrieve_similar(message, k=k)

    examples_text = ""
    for i, (doc, meta) in enumerate(zip(retrieval_results['documents'][0], retrieval_results['metadatas'][0])):
        examples_text += f"\nExample {i+1}:\nRider said: \"{meta['customer_text']}\"\nUber replied: \"{meta['uber_reply']}\"\n"

    prompt = REPLY_PROMPT_TEMPLATE.format(examples=examples_text, message=message)

    resp = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}]
    )

    reply = resp.choices[0].message.content.strip()

    return {
        "reply": reply,
        "grounding_examples": [
            {"similar_complaint": meta['customer_text'], "historical_reply": meta['uber_reply']}
            for meta in retrieval_results['metadatas'][0]
        ]
    }

if __name__ == "__main__":
    test_messages = [
        "Why should I pay a cancellation fee when the driver cancelled on me?!",
        "My driver was driving recklessly and almost hit another car, I'm scared",
        "I left my phone in the Uber, please help me get it back"
    ]

    print("--- Quick test ---\n")
    for msg in test_messages:
        result = generate_reply(msg)
        print(f"MESSAGE: {msg}")
        print(f"GENERATED REPLY: {result['reply']}")
        print(f"(grounded in {len(result['grounding_examples'])} similar historical cases)")
        print("-" * 80)