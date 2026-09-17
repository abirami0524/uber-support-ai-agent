from dotenv import load_dotenv
load_dotenv()

import os
import json
import re
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ============================================================
# RULE-BASED SIGNALS
# ============================================================

SAFETY_KEYWORDS = [
    'assault',
    'attack',
    'hit me',
    'hit my',
    'threat',
    'threatened',
    'abuse',
    'abused',
    'harassed',
    'harassing',
    'inappropriate',
    'unsafe',
    'reckless',
    'accident',
    'crash',
    'police',
    'weapon',
    'scared',
    'afraid',
    'hurt me'
]

SECURITY_KEYWORDS = [
    'hacked',
    'unauthorized',
    'stolen account',
    "isn't me",
    'someone else charged',
    'fraud',
    'scam'
]

DISCRIMINATION_KEYWORDS = [
    'refused to pick',
    'refused me',
    'discriminat',
    'wheelchair',
    'disability',
    'crutches',
    'blind',
    'service dog'
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


# ============================================================
# ESCALATION PROMPT
# ============================================================

ESCALATION_PROMPT = """You are deciding whether a customer support message to Uber should be:
- "auto_handle": routine issue, low-risk, can be resolved with a standard templated response
- "escalate": needs a human agent because it involves safety/physical harm, security/fraud, discrimination, a pattern of unresolved prior complaints, or an unusually large financial discrepancy

Message: "{message}"

Detected rule-based signals: {signals}

Respond in EXACTLY this format, nothing else:
DECISION: <auto_handle or escalate>
REASON: <one short sentence explaining why>"""


# ============================================================
# NORMALIZE MODEL DECISION
# ============================================================

def normalize_decision(value: str):

    value = value.strip().lower()

    # Auto-handle variations
    if value in [
        "auto_handle",
        "auto-handle",
        "auto handle",
        "autohandle"
    ]:
        return "auto_handle"

    # Escalation variations
    if value in [
        "escalate",
        "elevate",
        "escalation",
        "human",
        "human_agent",
        "human-agent",
        "manual_review",
        "manual-review"
    ]:
        return "escalate"

    return None


# ============================================================
# PARSE MODEL OUTPUT
# ============================================================

def parse_model_output(output: str):

    decision = None
    reason = None

    if not output:
        return None, None

    # --------------------------------------------------------
    # Clean markdown/code fences
    # --------------------------------------------------------

    cleaned = output.strip()

    cleaned = re.sub(
        r"```(?:json|text)?",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = cleaned.replace("```", "").strip()

    # --------------------------------------------------------
    # 1. Try JSON
    # --------------------------------------------------------

    try:

        parsed = json.loads(cleaned)

        if isinstance(parsed, dict):

            raw_decision = str(
                parsed.get("decision", "")
            ).strip().lower()

            raw_reason = str(
                parsed.get("reason", "")
            ).strip()

            decision = normalize_decision(raw_decision)

            if raw_reason:
                reason = raw_reason

            if decision:
                return decision, reason

    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # --------------------------------------------------------
    # 2. Try labeled text
    # --------------------------------------------------------

    for line in cleaned.splitlines():

        line = line.strip()

        # Remove markdown formatting
        line = line.replace("**", "").replace("__", "")

        upper_line = line.upper()

        # ----------------------------------------------------
        # DECISION
        # ----------------------------------------------------

        if upper_line.startswith("DECISION"):

            if ":" in line:
                value = line.split(":", 1)[1].strip()

            elif "-" in line:
                value = line.split("-", 1)[1].strip()

            else:
                value = line

            decision = normalize_decision(value.lower())

        # ----------------------------------------------------
        # REASON
        # ----------------------------------------------------

        elif upper_line.startswith("REASON"):

            if ":" in line:
                reason = line.split(":", 1)[1].strip()

            elif "-" in line:
                reason = line.split("-", 1)[1].strip()

    # --------------------------------------------------------
    # 3. Regex fallback
    # --------------------------------------------------------

    if decision is None:

        match = re.search(
            r"decision\s*[:=-]\s*([a-zA-Z_-]+)",
            cleaned,
            flags=re.IGNORECASE
        )

        if match:

            decision = normalize_decision(
                match.group(1).lower()
            )

    if reason is None:

        match = re.search(
            r"reason\s*[:=-]\s*(.+)",
            cleaned,
            flags=re.IGNORECASE
        )

        if match:
            reason = match.group(1).strip()

    return decision, reason


# ============================================================
# MAIN ESCALATION DECISION
# ============================================================

def decide_escalation(message: str):

    signals = rule_based_signals(message)

    signals_text = ", ".join(signals) if signals else "none"

    prompt = ESCALATION_PROMPT.format(
        message=message,
        signals=signals_text
    )

    output = ""

    # --------------------------------------------------------
    # MODEL CALL
    # Retry once if Groq returns an empty response
    # --------------------------------------------------------

    for attempt in range(2):

        try:

            resp = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=100
            )

            content = resp.choices[0].message.content
            output = content.strip() if content else ""

            # If we got a response, stop retrying
            if output:
                break

        except Exception as e:

            print(
                f"\nMODEL ERROR (attempt {attempt + 1}): {e}"
            )

    # --------------------------------------------------------
    # DEBUG
    # --------------------------------------------------------

    print("\nRAW MODEL OUTPUT:")
    print(repr(output))

    # --------------------------------------------------------
    # PARSE
    # --------------------------------------------------------

    decision, reason = parse_model_output(output)

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if decision is None:

        decision = "auto_handle"

        if output:

            reason = (
                "Model returned an unrecognized decision format; "
                "defaulted to auto_handle for review."
            )

        else:

            reason = (
                "Model returned an empty response after retry; "
                "defaulted to auto_handle for review."
            )

    if reason is None or not reason.strip():

        reason = "No reason provided by the model."

    # --------------------------------------------------------
    # RULE-BASED OVERRIDE
    # --------------------------------------------------------

    if signals and decision == "auto_handle":

        decision = "escalate"

        reason = (
            f"Overridden to escalate due to detected signal(s): "
            f"{signals_text}. {reason}"
        )

    return {
        "decision": decision,
        "reason": reason,
        "rule_signals": signals
    }


# ============================================================
# QUICK TEST
# ============================================================

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