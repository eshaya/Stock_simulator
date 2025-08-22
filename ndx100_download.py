import yfinance as yf
import pandas as pd

def download_prices(ticker_list, start="2005-01-01", end="2025-01-01", out="ndx100_2005_prices.csv"):
    data = yf.download(ticker_list, start=start, end=end)["Adj Close"]
    data.to_csv(out)
    print(f"Saved {len(data)} rows x {len(data.columns)} tickers to {out}")

tickers = pd.read_csv("ndx100_2005_tickers.csv")["Ticker"].tolist()
download_prices(tickers)
