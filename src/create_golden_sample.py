import pandas as pd

SOURCE_CSV = "data/apple_support_pairs.csv"
OUTPUT_GOLDEN_CSV = "data/golden_eval_set.csv"
SAMPLE_SIZE = 150

def generate_golden_sample():
    df = pd.read_csv(SOURCE_CSV)
    
    # Take a random sample of 150 rows using a fixed seed so it is reproducible
    sample = df.sample(n=SAMPLE_SIZE, random_state=42).copy()
    
    # Add empty columns for your human judgment
    sample["gold_intent"] = ""
    sample["gold_action"] = ""       # AUTO_REPLY or ESCALATE
    sample["gold_reason"] = ""       # Why did you choose that action?
    
    # Organize columns for easy reading
    ordered_cols = [
        "customer_tweet_id",
        "customer_text",
        "brand_reply_text",
        "gold_intent",
        "gold_action",
        "gold_reason"
    ]
    
    sample[ordered_cols].to_csv(OUTPUT_GOLDEN_CSV, index=False)
    print(f"Generated {SAMPLE_SIZE} rows to label in {OUTPUT_GOLDEN_CSV}")

if __name__ == "__main__":
    generate_golden_sample()