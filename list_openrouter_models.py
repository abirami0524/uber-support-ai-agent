from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

resp = client.models.list()
free_models = [m.id for m in resp.data if ':free' in m.id]
for m in free_models:
    print(m)