from dotenv import load_dotenv
load_dotenv()
import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---- Rule-based signals (fast, deterministic, catches clear-cut cases) ----
SAFETY_KEYWORDS = [
    'assault', 'attack', 'hit me', 'hit my', 'threat', 'threatened', 'abuse', 'abused',
    'harassed', 'harassing', 'inappropriate', 'unsafe', 'reckless', 'accident', 'crash',
    'police', 'weapon', 'scared', 'afraid', 'hurt me'
]
SECURITY_KEYWORDS = [
    'hacked', 'unauthorized', 'stolen account', "isn't me", 'someone else charged',
    'fraud', 'scam'
]
DISCRIMINATION_KEYWORDS = [
    'refused to pick', 'refused me', 'because i', 'discriminat', 'wheelchair', 'disability',
    'crutches', 'blind', 'service dog'
]

def rule_based_signals(message: str) -> list:
    msg = message.lower()
    signals = []
    if any(kw in msg for kw in SAFETY_KEYWORDS):
        signals.append("safety_language_detected")
    if any(kw in msg for kw in SECURITY_KEYWORDS):
        signals.append("security_fraud_language_detected")
    if any(kw in msg for kw in DISCRIMINATION_KEYWORDS):
        signals.append("possible_discrimination_pattern")
    return signals

ESCALATION_PROMPT = """You are deciding whether a customer support message to Uber should be:
- "auto_handle": routine issue, low-risk, can be resolved with a standard templated response (e.g. routine fare disputes, wait time complaints, app bugs, general account help)
- "escalate": needs a human agent because it involves safety/physical harm, security/fraud (hacked accounts, unauthorized charges), discrimination, a pattern of unresolved prior complaints, or an unusually large financial discrepancy

Message: "{message}"

Detected rule-based signals: {signals}

Respond in EXACTLY this format, nothing else:
DECISION: <auto_handle or escalate>
REASON: <one short sentence explaining why>"""

def decide_escalation(message: str) -> dict:
    signals = rule_based_signals(message)
    signals_text = ", ".join(signals) if signals else "none"

    prompt = ESCALATION_PROMPT.format(message=message, signals=signals_text)

    resp = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}]
    )
    output = resp.choices[0].message.content.strip()

    # Parse the structured output
    decision = "auto_handle"
    reason = "Could not parse model output, defaulted to auto_handle for review."
    for line in output.split("\n"):
        if line.upper().startswith("DECISION:"):
            val = line.split(":", 1)[1].strip().lower()
            decision = "escalate" if "escalate" in val else "auto_handle"
        if line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    # Safety net: if rule-based signals fired strongly, force escalate regardless of LLM output
    if signals and decision == "auto_handle":
        decision = "escalate"
        reason = f"Overridden to escalate due to detected signal(s): {signals_text}. " + reason

    return {"decision": decision, "reason": reason, "rule_signals": signals}

if __name__ == "__main__":
    test_messages = [
        "Why should I pay a cancellation fee when the driver cancelled on me?!",
        "My driver was driving recklessly and almost hit another car, I'm scared",
        "I left my phone in the Uber, please help me get it back",
        "My account was hacked and someone is using my card for rides",
        "app won't let me add my payment method"
    ]

    print("--- Quick test ---\n")
    for msg in test_messages:
        result = decide_escalation(msg)
        print(f"MESSAGE: {msg}")
        print(f"DECISION: {result['decision']}")
        print(f"REASON: {result['reason']}")
        print(f"RULE SIGNALS: {result['rule_signals']}")
        print("-" * 80)