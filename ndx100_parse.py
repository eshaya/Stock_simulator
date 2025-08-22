import re
import pandas as pd

def parse_ndx2005(filename="ndx100_2005_tickers.txt"):
    tickers = []
    with open(filename) as f:
        for line in f:
            # extract ticker inside parentheses
            match = re.search(r"\(([^)]+)\)", line)
            if match:
                tickers.append(match.group(1).strip())
    return pd.DataFrame({"Ticker": tickers})

df = parse_ndx2005()
print(df.head())
print(f"Total tickers parsed: {len(df)}")
df.to_csv("ndx100_2005_tickers.csv", index=False)
