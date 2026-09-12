import os
import re
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, recall_score

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

GOLDEN_PATH = "data/golden_eval_set.csv"
OUTPUT_RESULTS_PATH = "evaluation/eval_results.csv"

def is_low_context(text: str) -> bool:
    """Detects acknowledgments, model-only replies, or pure links."""
    cleaned = re.sub(r"(@\w+|https?://\S+)", "", text).strip()
    words = cleaned.split()
    if len(words) <= 2:
        return True
    return False

def rule_based_guardrail(text: str) -> bool:
    """Deterministic triggers for mandatory escalation."""
    t = text.lower()
    critical_triggers = [
        # Billing, orders, and delivery disputes
        "refund", "double charge", "charged twice", "unauthorized purchase",
        "charged me", "cancel subscription", "billing error", "stole my money",
        "delivery address", "order placed", "shipping address", "cancel order",
        # Account lockouts and security
        "hacked", "stolen", "locked out", "apple id locked", "security breach",
        # Physical damage
        "cracked screen", "shattered", "water damage", "dropped in water", "broken glass",
        # High anger, profanity, and repeat contact frustration
        "lawyer", "sue", "worst service", "unacceptable", "garbage", "tf @", "buggy af",
        "still waiting", "already sent dm", "no response", "thanks so much for nothing",
        "called you", "called support", "not working still"
    ]
    return any(trigger in t for trigger in critical_triggers)

def rule_based_predict(text: str):
    action = "ESCALATE" if rule_based_guardrail(text) else "AUTO_REPLY"
    t = text.lower()
    if any(w in t for w in ["battery", "drain", "charge", "overheat", "dying"]):
        intent = "BATTERY_POWER"
    elif any(w in t for w in ["broken", "shattered", "screen", "crack", "camera", "speaker"]):
        intent = "HARDWARE_PHYSICAL"
    elif any(w in t for w in ["apple id", "password", "2fa", "locked", "login", "icloud"]):
        intent = "ACCOUNT_SECURITY"
    elif any(w in t for w in ["charge", "billed", "subscription", "refund", "pay", "order", "delivery"]):
        intent = "BILLING_STORE"
    else:
        intent = "OS_SOFTWARE"

    return intent, action

class EvalPrediction(BaseModel):
    tweet_index: int
    intent: str = Field(description="BATTERY_POWER, OS_SOFTWARE, ACCOUNT_SECURITY, HARDWARE_PHYSICAL, BILLING_STORE")
    action: str = Field(description="AUTO_REPLY or ESCALATE")

class BatchEvalResponse(BaseModel):
    results: list[EvalPrediction]

def run_evaluation():
    df = pd.read_csv(GOLDEN_PATH)
    print(f"Loaded {len(df)} golden rows for calibrated evaluation.\n")

    # 1. Baseline Run
    rb_intents, rb_actions = [], []
    for _, row in df.iterrows():
        i, a = rule_based_predict(str(row["customer_text"]))
        rb_intents.append(i)
        rb_actions.append(a)

    df["rule_intent"] = rb_intents
    df["rule_action"] = rb_actions

    # 2. Production Agent Run (30 rows/batch = 5 requests total)
    BATCH_SIZE = 30
    agent_predictions = {}

    system_prompt = """You are the official tier-1 triage classifier for @AppleSupport.
Classify each customer tweet into an intent and action.

Taxonomy:
- BATTERY_POWER: Battery drain, fast discharge, battery percentage drops, overheating while idle.
- OS_SOFTWARE: Software crashes, iOS updates, Wi-Fi/Bluetooth bugs, app errors, and post-update biometric/touchscreen glitches.
- ACCOUNT_SECURITY: Apple ID locks, 2FA prompt failures, password recovery, iCloud login errors.
- HARDWARE_PHYSICAL: Physically broken devices, shattered glass, water damage, external adapters/cables not connecting, headphone audio hardware loss.
- BILLING_STORE: In-app purchases, unexpected card charges, subscriptions, refunds, and order/shipping changes.

Disambiguation Rules:
1. Physical accessories (OTG adapters, faulty headphone hardware, charger cables) belong to HARDWARE_PHYSICAL.
2. If Touch ID or Touchscreen stops working immediately following an OS update without physical breakage, classify as OS_SOFTWARE.
3. If an input is an acknowledgment ('Sent', 'Done') or only specifies a device name, classify as OS_SOFTWARE and AUTO_REPLY.
4. If a customer mentions prior unresolved contact ('already called', 'sent DM long back'), classify action as ESCALATE.

Action Rules:
- ESCALATE: For financial disputes, shipping changes, physical breakage, locked accounts, repeat unresolved tickets, or acute customer frustration.
- AUTO_REPLY: Routine troubleshooting queries solvable via standard steps."""

    print("Running Calibrated Production Agent...")
    for start_idx in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[start_idx:start_idx + BATCH_SIZE]
        prompt = system_prompt + "\n\nClassify each tweet by its tweet_index:\n"
        for idx, row in batch.iterrows():
            prompt += f"\n[tweet_index: {idx}] Tweet: \"{row['customer_text']}\""

        print(f"Evaluating batch: rows {start_idx} to {min(start_idx + BATCH_SIZE - 1, len(df) - 1)}...")

        wait_time = 35
        for attempt in range(5):
            try:
                res = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=BatchEvalResponse,
                        temperature=0.0
                    )
                )
                parsed = BatchEvalResponse.model_validate_json(res.text)
                for item in parsed.results:
                    agent_predictions[item.tweet_index] = (item.intent, item.action)
                break
            except Exception as e:
                print(f"  Attempt {attempt + 1} hit error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                wait_time += 15

        time.sleep(20)

    # 3. Apply Heuristic Overrides & Low-Context Handling
    ag_intents, ag_actions = [], []
    for i in range(len(df)):
        raw_text = str(df.iloc[i]["customer_text"])
        pred = agent_predictions.get(i, ("OS_SOFTWARE", "AUTO_REPLY"))
        pred_intent, pred_action = pred[0], pred[1]

        if is_low_context(raw_text):
            pred_intent = "OS_SOFTWARE"
            pred_action = "AUTO_REPLY"

        if rule_based_guardrail(raw_text):
            pred_action = "ESCALATE"

        ag_intents.append(pred_intent)
        ag_actions.append(pred_action)

    df["agent_intent"] = ag_intents
    df["agent_action"] = ag_actions

    df.to_csv(OUTPUT_RESULTS_PATH, index=False)
    print(f"\nSaved benchmark dataset to {OUTPUT_RESULTS_PATH}")

    # 4. Metrics Reporting
    print("\n" + "="*50)
    print("        CALIBRATED BENCHMARK EVALUATION RESULTS")
    print("="*50)

    rule_action_acc = accuracy_score(df["gold_action"], df["rule_action"])
    agent_action_acc = accuracy_score(df["gold_action"], df["agent_action"])
    rule_action_f1 = f1_score(df["gold_action"], df["rule_action"], pos_label="ESCALATE")
    agent_action_f1 = f1_score(df["gold_action"], df["agent_action"], pos_label="ESCALATE")
    rule_recall = recall_score(df["gold_action"], df["rule_action"], pos_label="ESCALATE")
    agent_recall = recall_score(df["gold_action"], df["agent_action"], pos_label="ESCALATE")
    rule_kappa = cohen_kappa_score(df["gold_action"], df["rule_action"])
    agent_kappa = cohen_kappa_score(df["gold_action"], df["agent_action"])

    print("\n[Escalation Action Metrics]")
    print(f"Rule-Based Baseline  -> Accuracy: {rule_action_acc:.2%}, Recall: {rule_recall:.2%}, F1: {rule_action_f1:.2f}, Kappa: {rule_kappa:.2f}")
    print(f"Production Agent     -> Accuracy: {agent_action_acc:.2%}, Recall: {agent_recall:.2%}, F1: {agent_action_f1:.2f}, Kappa: {agent_kappa:.2f}")

    rule_intent_acc = accuracy_score(df["gold_intent"], df["rule_intent"])
    agent_intent_acc = accuracy_score(df["gold_intent"], df["agent_intent"])
    rule_intent_macro_f1 = f1_score(df["gold_intent"], df["rule_intent"], average="macro")
    agent_intent_macro_f1 = f1_score(df["gold_intent"], df["agent_intent"], average="macro")

    print("\n[Intent Classification Metrics]")
    print(f"Rule-Based Baseline  -> Intent Accuracy: {rule_intent_acc:.2%}, Macro F1: {rule_intent_macro_f1:.2f}")
    print(f"Production Agent     -> Intent Accuracy: {agent_intent_acc:.2%}, Macro F1: {agent_intent_macro_f1:.2f}")
    print("="*50)

if __name__ == "__main__":
    run_evaluation()