import os
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

client = genai.Client(api_key=api_key)

INPUT_FILE = "data/golden_eval_set.csv"
OUTPUT_FILE = "data/golden_eval_set.csv"
BATCH_SIZE = 15  # Process 15 rows per API request

class LabeledTweet(BaseModel):
    tweet_index: int
    intent: str = Field(description="One of: BATTERY_POWER, OS_SOFTWARE, ACCOUNT_SECURITY, HARDWARE_PHYSICAL, BILLING_STORE")
    action: str = Field(description="AUTO_REPLY or ESCALATE")
    reason: str = Field(description="Short reason if escalated, otherwise NONE")

class BatchLabelResponse(BaseModel):
    results: list[LabeledTweet]

def prelabel_in_batches():
    df = pd.read_csv(INPUT_FILE)
    total_rows = len(df)
    print(f"[*] Pre-labeling {total_rows} rows in batches of {BATCH_SIZE}...")

    results_dict = {}

    system_rules = """You are auditing customer support tweets for AppleSupport.
Categories for intent:
- BATTERY_POWER: Battery drain, device overheating, charging cable/port issues.
- OS_SOFTWARE: App crashes, iOS/macOS update issues, frozen screen, lag, Bluetooth/Wi-Fi bugs.
- ACCOUNT_SECURITY: Locked Apple ID, 2FA codes, forgotten passwords, iCloud sync/login issues.
- HARDWARE_PHYSICAL: Shattered glass, broken speaker, cracked camera, damaged buttons.
- BILLING_STORE: Subscriptions, unexpected charges, refund requests, App Store purchasing.

Action rules:
- ESCALATE: If it involves refunds/charges, account lockout, physical damage, severe anger, or asks to DM.
- AUTO_REPLY: For standard troubleshooting (restarts, settings verification, standard updates).
"""

    for start_idx in range(0, total_rows, BATCH_SIZE):
        batch = df.iloc[start_idx:start_idx + BATCH_SIZE]
        batch_prompt = system_rules + "\n\nLabel each of the following tweets by its tweet_index:\n"
        
        for idx, row in batch.iterrows():
            batch_prompt += f"\n[tweet_index: {idx}] Tweet: \"{row['customer_text']}\""

        print(f"[*] Processing rows {start_idx} to {min(start_idx + BATCH_SIZE - 1, total_rows - 1)}...")

        success = False
        attempts = 0
        while not success and attempts < 3:
            try:
                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=batch_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=BatchLabelResponse,
                        temperature=0.0
                    )
                )
                parsed = BatchLabelResponse.model_validate_json(response.text)
                for item in parsed.results:
                    results_dict[item.tweet_index] = (item.intent, item.action, item.reason)
                success = True
            except Exception as e:
                attempts += 1
                print(f"    [!] Hit error: {e}. Retrying in 20s (Attempt {attempts}/3)...")
                time.sleep(20)

        # Pause to remain safely below the 5 RPM rate limit
        time.sleep(13)

    # Map parsed responses back into the dataframe
    intents, actions, reasons = [], [], []
    for i in range(total_rows):
        res = results_dict.get(i, ("OS_SOFTWARE", "AUTO_REPLY", "NONE"))
        intents.append(res[0])
        actions.append(res[1])
        reasons.append(res[2])

    df["gold_intent"] = intents
    df["gold_action"] = actions
    df["gold_reason"] = reasons

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[✓] Finished! All rows labeled and saved to {OUTPUT_FILE}.")
    print("[*] Next step: Open data/golden_eval_set.csv and review the rows.")

if __name__ == "__main__":
    prelabel_in_batches()