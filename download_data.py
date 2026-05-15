import os
import pandas as pd
import yfinance as yf

NIFTY500_CSV_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

os.makedirs("data", exist_ok=True)

symbols_df = pd.read_csv(NIFTY500_CSV_URL)

symbols = (
    symbols_df["Symbol"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

print(f"Downloading data for {len(symbols)} stocks")

for idx, symbol in enumerate(symbols, start=1):

    try:
        ticker = f"{symbol}.NS"

        print(f"[{idx}/{len(symbols)}] Downloading {ticker}")

        df = yf.download(
            ticker,
            start="2018-01-01",
            progress=False,
            auto_adjust=True,
            threads=False
        )

        if df.empty:
            print(f"{symbol}: Empty dataframe")
            continue

        df = df.reset_index()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).strip() for c in df.columns]

        required_cols = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        missing = [
            c for c in required_cols
            if c not in df.columns
        ]

        if missing:
            print(f"{symbol}: Missing columns {missing}")
            continue

        df = df[required_cols]

        df.to_csv(f"data/{symbol}.csv", index=False)

    except Exception as e:
        print(symbol, e)

print("Historical download completed")
