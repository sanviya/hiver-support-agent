import pandas as pd

# Set display options so text doesn't get truncated in the terminal
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

df = pd.read_csv("evaluation/eval_results.csv")

# 1. Intent Mismatches
intent_errors = df[df["gold_intent"] != df["agent_intent"]][["customer_text", "gold_intent", "agent_intent"]]
print("=" * 60)
print(f"Total Intent Disagreements: {len(intent_errors)}")
print("=" * 60)
print(intent_errors.head(10))

# 2. Escalation Action Mismatches
action_errors = df[df["gold_action"] != df["agent_action"]][["customer_text", "gold_action", "agent_action"]]
print("\n" + "=" * 60)
print(f"Total Escalation Disagreements: {len(action_errors)}")
print("=" * 60)
print(action_errors.head(10))