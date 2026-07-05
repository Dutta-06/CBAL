import argparse
import json
import pandas as pd
from collections import Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", help="Path to execution log/json output", required=True)
    args = parser.parse_args()

    # Assuming you save your run decisions to a JSON lines file
    decisions = []
    with open(args.log_file, 'r') as f:
        for line in f:
            try:
                decisions.append(json.loads(line))
            except: continue

    df = pd.DataFrame(decisions)
    
    print("--- Error Analysis ---")
    print(f"Total Candidates Processed: {len(df)}")
    
    if 'action' in df.columns:
        print("\nAction Distribution:")
        print(df['action'].value_counts())

    if 'type' in df.columns and 'action' in df.columns:
        print("\nFix Rate by Error Type:")
        print(df.groupby('type')['action'].apply(lambda x: (x == 'MERGE').mean()))

if __name__ == "__main__":
    main()