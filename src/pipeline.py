from dotenv import load_dotenv
load_dotenv()
from src.classify_intent import classify
from src.reply_generation import generate_reply
from src.escalation import decide_escalation

def process_message(message: str) -> dict:
    """
    Core pipeline: given a raw customer message, returns intent, a grounded
    drafted reply, and an auto_handle/escalate decision with a reason.
    """
    intent = classify(message)
    reply_result = generate_reply(message, k=3)
    escalation_result = decide_escalation(message)

    return {
        "message": message,
        "intent": intent,
        "drafted_reply": reply_result["reply"],
        "grounding_examples": reply_result["grounding_examples"],
        "decision": escalation_result["decision"],
        "reason": escalation_result["reason"],
        "rule_signals": escalation_result["rule_signals"]
    }

if __name__ == "__main__":
    test_messages = [
        "Why should I pay a cancellation fee when the driver cancelled on me?!",
        "My driver was driving recklessly and almost hit another car, I'm scared",
        "I left my phone in the Uber, please help me get it back"
    ]

    print("--- Full pipeline test ---\n")
    for msg in test_messages:
        result = process_message(msg)
        print(f"MESSAGE: {result['message']}")
        print(f"INTENT: {result['intent']}")
        print(f"DECISION: {result['decision']}  |  REASON: {result['reason']}")
        print(f"DRAFTED REPLY: {result['drafted_reply']}")
        print("-" * 80)