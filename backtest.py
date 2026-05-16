import os
import pandas as pd
import numpy as np

DATA_FOLDER = "data"

TARGET_PCT = 0.04
STOP_PCT = 0.025
MAX_HOLD_DAYS = 60

MIN_PRICE = 50
MIN_VOL_RATIO = 1.5

RSI_LOW = 52
RSI_HIGH = 56

trades = []


def ema(series, span):

    return series.ewm(
        span=span,
        adjust=False
    ).mean()


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

    return (
        100 - (100 / (1 + rs))
    ).fillna(0)


def add_indicators(df):

    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200)

    df["RSI14"] = rsi(df["Close"], 14)

    df["VOL_AVG20"] = (
        df["Volume"].rolling(20).mean()
    )

    df["RET_20D"] = (
        df["Close"].pct_change(20) * 100
    )

    df["ATR14"] = (
        (
            df["High"] - df["Low"]
        ).rolling(14).mean()
    )

    return df


def signal(row, prev_row):

    vol_ratio = (
        row["Volume"]
        / row["VOL_AVG20"]
        if row["VOL_AVG20"] > 0
        else 0
    )

    pullback_zone = (
        row["Close"]
        >= row["EMA20"] * 0.985
    ) and (
        row["Close"]
        <= row["EMA20"] * 1.02
    )

    conditions = {

        "price":
        row["Close"] >= MIN_PRICE,

        "ema_stack":
        row["EMA20"]
        > row["EMA50"]
        > row["EMA200"],

        "above_ema50":
        row["Close"] > row["EMA50"],

        "ema20_rising":
        row["EMA20"] > prev_row["EMA20"],

        "pullback":
        pullback_zone,

        "rsi":
        RSI_LOW <= row["RSI14"] <= RSI_HIGH,

        "relative_strength":
        row["RET_20D"] > 5,

        "volume":
        vol_ratio >= MIN_VOL_RATIO,

        "bullish_candle":
        row["Close"] > row["Open"],

        "recovery":
        row["Close"] > prev_row["Close"],

        "controlled_atr":
        (
            row["ATR14"]
            / row["Close"]
        ) < 0.035,
    }

    return sum(conditions.values()) >= 9


def simulate_trade(
    df,
    signal_idx,
    symbol
):

    if signal_idx + 1 >= len(df):
        return

    entry_row = df.iloc[
        signal_idx + 1
    ]

    entry_price = entry_row["Open"]

    target_price = (
        entry_price
        * (1 + TARGET_PCT)
    )

    stop_price = (
        entry_price
        * (1 - STOP_PCT)
    )

    entry_date = entry_row["Date"]

    for i in range(
        signal_idx + 1,
        min(
            signal_idx + MAX_HOLD_DAYS,
            len(df) - 1
        )
    ):

        row = df.iloc[i]

        if row["High"] >= target_price:

            trades.append({
                "symbol": symbol,
                "entry_date": entry_date,
                "exit_date": row["Date"],
                "return_pct":
                TARGET_PCT * 100,
                "result": "TARGET"
            })

            return

        if row["Low"] <= stop_price:

            trades.append({
                "symbol": symbol,
                "entry_date": entry_date,
                "exit_date": row["Date"],
                "return_pct":
                -STOP_PCT * 100,
                "result": "STOP"
            })

            return

    final_row = df.iloc[
        min(
            signal_idx + MAX_HOLD_DAYS,
            len(df) - 1
        )
    ]

    pnl_pct = (
        (
            final_row["Close"]
            - entry_price
        )
        / entry_price
        * 100
    )

    trades.append({
        "symbol": symbol,
        "entry_date": entry_date,
        "exit_date": final_row["Date"],
        "return_pct": pnl_pct,
        "result": "TIME_EXIT"
    })


files = [
    f for f in os.listdir(DATA_FOLDER)
    if f.endswith(".csv")
]

print(
    f"Backtesting {len(files)} stocks"
)

for file in files:

    symbol = file.replace(".csv", "")

    try:

        df = pd.read_csv(
            f"{DATA_FOLDER}/{file}"
        )

        if len(df) < 250:
            continue

        df["Date"] = pd.to_datetime(
            df["Date"]
        )

        df = add_indicators(df)

        df = df.dropna().reset_index(
            drop=True
        )

        for idx in range(
            1,
            len(df)
            - MAX_HOLD_DAYS
            - 1
        ):

            row = df.iloc[idx]

            prev_row = df.iloc[idx - 1]

            if signal(
                row,
                prev_row
            ):

                simulate_trade(
                    df,
                    idx,
                    symbol
                )

    except Exception as e:
        print(symbol, e)

trades_df = pd.DataFrame(trades)

trades_df.to_csv(
    "trades.csv",
    index=False
)

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

profit_factor = (
    wins["return_pct"].sum()
    / abs(losses["return_pct"].sum())
)

print(
    "\n========== BACKTEST SUMMARY =========="
)

print(
    f"Total Trades: {len(trades_df)}"
)

print(
    f"Win Rate: {win_rate:.2f}%"
)

print(
    f"Profit Factor: "
    f"{profit_factor:.2f}"
)

print(
    "\nTrades saved to trades.csv"
)
