import json
import pandas as pd
path = [
    "multinli_1.0_dev_matched.jsonl",
    "multinli_1.0_dev_mismatched.jsonl"
]

for p in path:
    with open(p) as fin:
        data = []
        for line in fin:
            d = json.loads(line.strip())
            data.append(d)
    df = pd.DataFrame(data)

    
    print(df.head())
    fname = p.replace(".jsonl", ".csv")
    df.to_csv(fname, index=False)