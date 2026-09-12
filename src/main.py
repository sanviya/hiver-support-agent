import os
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Stage 1: Structured Classification Schema
class TriageClassification(BaseModel):
    intent: str = Field(description="One of: BATTERY_POWER, OS_SOFTWARE, ACCOUNT_SECURITY, HARDWARE_PHYSICAL, BILLING_STORE")
    customer_sentiment: str = Field(description="NEUTRAL, FRUSTRATED, or ENRAGED")
    urgency: str = Field(description="LOW, MEDIUM, or HIGH")
    reasoning: str = Field(description="1-sentence explanation of classification.")

# Stage 2: Deterministic Guardrail Check
def evaluate_escalation_guardrail(text: str, sentiment: str) -> tuple[bool, str]:
    t = text.lower()
    critical_triggers = [
        "refund", "double charge", "charged twice", "unauthorized purchase",
        "charged me", "cancel subscription", "billing error", "stole my money",
        "delivery address", "order placed", "shipping address", "cancel order",
        "hacked", "stolen", "locked out", "apple id locked", "security breach",
        "cracked screen", "shattered", "water damage", "dropped in water", "broken glass",
        "lawyer", "sue", "worst service", "unacceptable", "garbage", "tf @", "buggy af",
        "still waiting", "already sent dm", "no response", "thanks so much for nothing",
        "called you", "called support", "not working still"
    ]
    for trigger in critical_triggers:
        if trigger in t:
            return True, f"Deterministic guardrail keyword matched: '{trigger}'"
    if sentiment in ["FRUSTRATED", "ENRAGED"]:
        return True, f"Elevated sentiment detected: {sentiment}"
    return False, "Standard troubleshooting path"

# Stage 3: Response Generation
def generate_support_response(tweet: str, intent: str, escalate: bool) -> str:
    system_instruction = """You are an official @AppleSupport triage agent on Twitter/X.
Follow these strict tone guidelines:
- Empathetic, concise, direct, and under 240 characters.
- Never make false promises or offer direct refunds.
- If escalating, invite them to continue over Direct Message (DM) for privacy.
- If auto-replying, provide 1 actionable initial diagnostic step."""

    routing_prompt = f"""Customer Tweet: "{tweet}"
Classified Intent: {intent}
Escalate to Human Specialist: {escalate}

Draft the exact tweet response:"""

    res = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=routing_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
            max_output_tokens=100
        )
    )
    return res.text.strip()

def process_ticket(tweet: str):
    print("\n" + "=" * 55)
    print(f"Incoming Tweet: {tweet}")
    print("-" * 55)

    # 1. Classification
    classify_prompt = f"""Classify this tweet:
"{tweet}"

Categories: BATTERY_POWER, OS_SOFTWARE, ACCOUNT_SECURITY, HARDWARE_PHYSICAL, BILLING_STORE.
Sentiment: NEUTRAL, FRUSTRATED, ENRAGED.
Urgency: LOW, MEDIUM, HIGH."""

    res = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=classify_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TriageClassification,
            temperature=0.0
        )
    )
    analysis = TriageClassification.model_validate_json(res.text)
    print(f"Stage 1 [Classification]: {analysis.intent} | Sentiment: {analysis.customer_sentiment} | Urgency: {analysis.urgency}")
    print(f"Reasoning: {analysis.reasoning}")

    # 2. Guardrail Gate
    must_escalate, trigger_reason = evaluate_escalation_guardrail(tweet, analysis.customer_sentiment)
    action = "ESCALATE (Route to Human)" if must_escalate else "AUTO_REPLY (Automated Fix)"
    print(f"Stage 2 [Guardrail Gate]: {action}")
    print(f"Policy Note: {trigger_reason}")

    # 3. Grounded Generation
    reply = generate_support_response(tweet, analysis.intent, must_escalate)
    print(f"Stage 3 [Generated Reply]:\n\"{reply}\"")
    print("=" * 55)

if __name__ == "__main__":
    print("Apple Support AI Triage Pipeline Active. (Type 'exit' to quit)\n")
    while True:
        user_input = input("Enter a test customer tweet: ").strip()
        if user_input.lower() in ["exit", "quit", "q"]:
            break
        if not user_input:
            continue
        process_ticket(user_input)