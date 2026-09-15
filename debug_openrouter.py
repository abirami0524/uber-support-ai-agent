from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

resp = client.chat.completions.create(
    model="z-ai/glm-5.2:free",
    messages=[{"role": "user", "content": "Classify this: 'why was I charged twice'. Respond with ONLY one word: fare_billing_dispute"}],
    max_tokens=50
)
print("FULL RESPONSE:", resp)
print("CHOICES:", resp.choices)