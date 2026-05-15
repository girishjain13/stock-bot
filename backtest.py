import os
import pandas as pd
import numpy as np

DATA_FOLDER = "data"

# ==============================
# STRATEGY SETTINGS
# ==============================

TARGET_PCT = 0.05
STOP_PCT = 0.01
MAX_HOLD_DAYS = 5

MIN_PRICE = 50
MIN_VOL_RATIO = 2.5

RSI_LOW = 60
RSI_HIGH = 63

MAX_DISTANCE_52W = 2.0

trades = []


# ==============================
# INDICATORS
# ==============================

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi_val = 100 - (100 / (1 + rs))

    return rsi_val.fillna(0)


def macd_hist(close):

    ema12 = ema(close, 12)
    ema26 = ema(close, 26)

    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)

    hist = macd_line - signal

    return hist


# ==============================
# ADD INDICATORS
# ==============================

def add_indicators(df):

    df = df.copy()

    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200)

    df["RSI14"] = rsi(df["Close"], 14)

    df["MACD_HIST"] = macd_hist(df["Close"])

    df["VOL_AVG20"] = df["Volume"].rolling(20).mean()

    df["AVG_PRICE20"] = df["Close"].rolling(20).mean()

    df["AVG_TRADED_VALUE20"] = (
        df["VOL_AVG20"] * df["AVG_PRICE20"]
    )

    df["52W_HIGH"] = df["High"].rolling(
        252,
        min_periods=100
    ).max()

    df["RET_20D"] = (
        df["Close"].pct_change(20) * 100
    )

    return df


# ==============================
# SIGNAL LOGIC
# ==============================

def signal(row, prev_row):

    vol_ratio = (
        row["Volume"] / row["VOL_AVG20"]
        if row["VOL_AVG20"] > 0
        else 0
    )

    dist_52w_high = (
        (row["52W_HIGH"] - row["Close"])
        / row["52W_HIGH"]
        * 100
    )

    ema_separation = (
        (row["EMA20"] - row["EMA50"])
        / row["EMA50"]
    )

    conditions = {

        # Price Filters
        "price": row["Close"] >= MIN_PRICE,

        # EMA Structure
        "ema20": row["Close"] > row["EMA20"],
        "ema50": row["Close"] > row["EMA50"],
        "ema200": row["Close"] > row["EMA200"],

        # Strong Trend Stack
        "stack":
        row["EMA20"] > row["EMA50"] > row["EMA200"],

        # Strong Momentum
        "strong_trend":
        row["Close"] > row["EMA20"] * 1.03,

        # EMA Separation
        "ema_separation":
        ema_separation > 0.02,

        # RSI Tight Range
        "rsi":
        RSI_LOW <= row["RSI14"] <= RSI_HIGH,

        # MACD Confirmation
        "macd":
        row["MACD_HIST"] > 0,

        # Rising Momentum
        "macd_rising":
        row["MACD_HIST"] > prev_row["MACD_HIST"],

        # Liquidity
        "liquidity":
        row["AVG_TRADED_VALUE20"] >= 20_00_00_000,

        # High Relative Volume
        "volume":
        vol_ratio >= MIN_VOL_RATIO,

        # Near 52W High
        "near_high":
        dist_52w_high <= MAX_DISTANCE_52W,

        # Strong 20D Momentum
        "relative_strength":
        row["RET_20D"] > 5,

        # Breakout Confirmation
        "breakout":
        row["Close"] >= row["52W_HIGH"] * 0.98,
    }

    return sum(conditions.values()) >= 12


# ==============================
# TRADE SIMULATION
# ==============================

def simulate_trade(df, signal_idx, symbol):

    if signal_idx + 1 >= len(df):
        return

    entry_row = df.iloc[signal_idx + 1]

    entry_price = entry_row["Open"]

    target_price = entry_price * (1 + TARGET_PCT)
    stop_price = entry_price * (1 - STOP_PCT)

    entry_date = entry_row["Date"]

    for i in range(
        signal_idx + 1,
        min(signal_idx + MAX_HOLD_DAYS, len(df) - 1)
    ):

        row = df.iloc[i]

        # TARGET HIT
        if row["High"] >= target_price:

            trades.append({
                "symbol": symbol,
                "entry_date": entry_date,
                "exit_date": row["Date"],
                "entry": round(entry_price, 2),
                "exit": round(target_price, 2),
                "return_pct": round(TARGET_PCT * 100, 2),
                "result": "TARGET"
            })

            return

        # STOP LOSS HIT
        if row["Low"] <= stop_price:

            trades.append({
                "symbol": symbol,
                "entry_date": entry_date,
                "exit_date": row["Date"],
                "entry": round(entry_price, 2),
                "exit": round(stop_price, 2),
                "return_pct": round(-STOP_PCT * 100, 2),
                "result": "STOP"
            })

            return

    # TIME EXIT
    final_row = df.iloc[
        min(signal_idx + MAX_HOLD_DAYS, len(df) - 1)
    ]

    pnl_pct = (
        (final_row["Close"] - entry_price)
        / entry_price
        * 100
    )

    trades.append({
        "symbol": symbol,
        "entry_date": entry_date,
        "exit_date": final_row["Date"],
        "entry": round(entry_price, 2),
        "exit": round(final_row["Close"], 2),
        "return_pct": round(pnl_pct, 2),
        "result": "TIME_EXIT"
    })


# ==============================
# LOAD FILES
# ==============================

files = [
    f for f in os.listdir(DATA_FOLDER)
    if f.endswith(".csv")
]

print(f"Backtesting {len(files)} stocks")


# ==============================
# RUN BACKTEST
# ==============================

for file in files:

    symbol = file.replace(".csv", "")

    try:

        df = pd.read_csv(f"{DATA_FOLDER}/{file}")

        if len(df) < 300:
            continue

        df["Date"] = pd.to_datetime(df["Date"])

        df = add_indicators(df)

        df = df.dropna().reset_index(drop=True)

        for idx in range(
            1,
            len(df) - MAX_HOLD_DAYS - 1
        ):

            row = df.iloc[idx]
            prev_row = df.iloc[idx - 1]

            if signal(row, prev_row):
                simulate_trade(df, idx, symbol)

    except Exception as e:
        print(symbol, e)


# ==============================
# RESULTS
# ==============================

trades_df = pd.DataFrame(trades)

trades_df.to_csv("trades.csv", index=False)

if len(trades_df) == 0:
    print("No trades generated")
    exit()

wins = trades_df[
    trades_df["return_pct"] > 0
]

losses = trades_df[
    trades_df["return_pct"] <= 0
]

win_rate = (
    len(wins)
    / len(trades_df)
    * 100
)

avg_return = trades_df[
    "return_pct"
].mean()

avg_win = (
    wins["return_pct"].mean()
    if len(wins) > 0
    else 0
)

avg_loss = (
    losses["return_pct"].mean()
    if len(losses) > 0
    else 0
)

profit_factor = (
    wins["return_pct"].sum()
    / abs(losses["return_pct"].sum())
    if len(losses) > 0
    else 0
)

# ==============================
# SUMMARY
# ==============================

print("\n========== BACKTEST SUMMARY ==========")

print(f"Total Trades: {len(trades_df)}")
print(f"Winning Trades: {len(wins)}")
print(f"Losing Trades: {len(losses)}")

print(f"Win Rate: {win_rate:.2f}%")

print(f"Average Return: {avg_return:.2f}%")

print(f"Average Win: {avg_win:.2f}%")
print(f"Average Loss: {avg_loss:.2f}%")

print(f"Profit Factor: {profit_factor:.2f}")

print("\nTrades saved to trades.csv")
