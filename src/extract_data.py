import os
import pandas as pd

DATASET_PATH = "data/twcs.csv"
TARGET_BRAND = "AppleSupport"
CHUNK_SIZE = 100_000
MAX_PAIRS_NEEDED = 3000
OUTPUT_CSV = "data/apple_support_pairs.csv"

def extract_pairs():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Could not find {DATASET_PATH}. Put twcs.csv inside the data folder first.")
        return

    print("Step 1: Finding brand replies...")
    brand_replies = []
    parent_ids = set()

    for chunk in pd.read_csv(DATASET_PATH, chunksize=CHUNK_SIZE, low_memory=False):
        mask = (chunk["author_id"] == TARGET_BRAND) & (~chunk["inbound"]) & (chunk["in_response_to_tweet_id"].notna())
        filtered = chunk[mask].copy()
        
        if not filtered.empty:
            valid_ids = pd.to_numeric(filtered["in_response_to_tweet_id"], errors="coerce").dropna().astype(int)
            filtered["in_response_to_tweet_id"] = valid_ids
            brand_replies.append(filtered)
            parent_ids.update(valid_ids.tolist())

        total = sum(len(df) for df in brand_replies)
        if total >= MAX_PAIRS_NEEDED * 2:
            break

    if not brand_replies:
        print(f"No replies found for @{TARGET_BRAND}.")
        return

    all_brand = pd.concat(brand_replies, ignore_index=True)
    print(f"Found {len(all_brand)} brand replies. Tracking {len(parent_ids)} question IDs.")

    print("Step 2: Finding customer questions...")
    inbounds = []
    for chunk in pd.read_csv(DATASET_PATH, chunksize=CHUNK_SIZE, low_memory=False):
        chunk["tweet_id"] = pd.to_numeric(chunk["tweet_id"], errors="coerce")
        matched = chunk[chunk["tweet_id"].isin(parent_ids)].copy()
        if not matched.empty:
            inbounds.append(matched)
        if sum(len(df) for df in inbounds) >= len(parent_ids):
            break

    all_inbounds = pd.concat(inbounds, ignore_index=True)
    print(f"Found {len(all_inbounds)} customer parent tweets.")

    print("Step 3: Pairing customer questions with Apple replies...")
    merged = pd.merge(
        all_inbounds, 
        all_brand, 
        left_on="tweet_id", 
        right_on="in_response_to_tweet_id", 
        suffixes=("_cust", "_brand")
    )

    cleaned = pd.DataFrame({
        "customer_tweet_id": merged["tweet_id_cust"].astype("Int64"),
        "customer_text": merged["text_cust"].astype(str),
        "brand_reply_text": merged["text_brand"].astype(str)
    })

    cleaned = cleaned[
        (cleaned["customer_text"].str.strip().str.len() > 15) &
        (cleaned["brand_reply_text"].str.strip().str.len() > 15)
    ].drop_duplicates(subset=["customer_tweet_id"]).head(MAX_PAIRS_NEEDED)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    cleaned.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! Extracted {len(cleaned)} clean pairs to {OUTPUT_CSV}")

if __name__ == "__main__":
    extract_pairs()