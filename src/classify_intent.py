from dotenv import load_dotenv
load_dotenv()
import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

INTENTS = [
    "fare_billing_dispute",
    "driver_safety_incident",
    "wait_time_cancellation",
    "account_access",
    "app_technical_issue",
    "lost_item",
    "general_complaint",
    "spam_irrelevant",
    "compliment_other"
]

PROMPT_TEMPLATE = """You are classifying customer support messages sent to Uber's support team on Twitter.

Classify the message below into EXACTLY ONE of these intents:
- fare_billing_dispute: wrong charge, overcharge, refund request, pricing dispute
- driver_safety_incident: unsafe driving, abuse, threats, accidents, discrimination, assault
- wait_time_cancellation: long wait, no-show, driver cancelled, ETA issues
- account_access: login, password, verification, account locked/hacked/deactivated
- app_technical_issue: app bugs, payment method errors, booking failures
- lost_item: left an item in the vehicle, needs driver contact to retrieve it
- general_complaint: vague dissatisfaction, feature requests, non-specific venting
- spam_irrelevant: unrelated ads, promotions, off-topic, not a real support issue
- compliment_other: praise, thanks, or anything positive

Message: "{message}"

Respond with ONLY the intent label, nothing else. No explanation, no punctuation."""


def classify(message: str) -> str:
    prompt = PROMPT_TEMPLATE.format(message=message)
    resp = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20
    )
    content = resp.choices[0].message.content
    if not content:
        return "general_complaint"
    label = content.strip().lower()
    if label not in INTENTS:
        for intent in INTENTS:
            if intent in label:
                return intent
        return "general_complaint"
    return label


if __name__ == "__main__":
    test_messages = [
        "Why should I pay a cancellation fee when the driver cancelled on me?!",
        "My driver was driving recklessly and almost hit another car, I'm scared",
        "I left my phone in the Uber, please help me get it back"
    ]
    print("--- Quick test ---")
    for msg in test_messages:
        label = classify(msg)
        print(f"{msg[:50]}... -> {label}")