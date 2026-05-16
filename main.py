import os
import time
import warnings
import requests
import pandas as pd
import numpy as np
import yfinance as yf

from datetime import date, timedelta

warnings.filterwarnings("ignore")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

NIFTY_SYMBOL = "^NSEI"

LOOKBACK_DAYS = 430

TARGET_PCT = 0.04
STOP_PCT = 0.025
MAX_HOLD_DAYS = 60

MIN_PRICE = 50
MIN_VOL_RATIO = 1.5

RSI_LOW = 52
RSI_HIGH = 58

TOP_N = 10

SLEEP_BETWEEN_SYMBOLS = 1.0


def send_telegram(message):

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(message)
        return

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    try:
        requests.post(
            url,
            json=payload,
            timeout=30
        )

    except Exception as e:
        print(e)


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

    tr = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift()),
            abs(df["Low"] - df["Close"].shift())
        )
    )

    df["ATR14"] = (
        tr.rolling(14).mean()
    )

    df["ATR_PCT"] = (
        df["ATR14"] / df["Close"]
    )

    return df


def fetch_history(symbol, start_dt, end_dt):

    try:

        df = yf.download(
            symbol,
            start=start_dt,
            end=end_dt,
            progress=False,
            auto_adjust=True,
            threads=False
        )

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = (
                df.columns.get_level_values(0)
            )

        df.columns = [
            str(c).strip()
            for c in df.columns
        ]

        return df

    except Exception:
        return pd.DataFrame()


def fetch_nifty500():

    url = (
        "https://archives.nseindia.com/"
        "content/indices/ind_nifty500list.csv"
    )

    df = pd.read_csv(url)

    return (
        df["Symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )


def market_is_bullish():

    today = date.today()

    start_dt = (
        today - timedelta(days=430)
    )

    nifty_df = fetch_history(
        NIFTY_SYMBOL,
        start_dt,
        today
    )

    if nifty_df.empty:
        return False

    nifty_df["EMA200"] = ema(
        nifty_df["Close"],
        200
    )

    latest = nifty_df.iloc[-1]

    return (
        latest["Close"]
        > latest["EMA200"]
    )


def analyze_stock(symbol, df):

    if df.empty or len(df) < 250:
        return None

    df = add_indicators(df)

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    vol_ratio = (
        latest["Volume"]
        / latest["VOL_AVG20"]
        if latest["VOL_AVG20"] > 0
        else 0
    )

    pullback_zone = (
        latest["Close"]
        >= latest["EMA20"] * 0.985
    ) and (
        latest["Close"]
        <= latest["EMA20"] * 1.02
    )

    conditions = {

        "price":
        latest["Close"] >= MIN_PRICE,

        "ema_stack":
        latest["EMA20"]
        > latest["EMA50"]
        > latest["EMA200"],

        "ema20_rising":
        latest["EMA20"] > prev["EMA20"],

        "pullback":
        pullback_zone,

        "rsi":
        RSI_LOW <= latest["RSI14"] <= RSI_HIGH,

        "relative_strength":
        latest["RET_20D"] > 5,

        "volume":
        vol_ratio >= MIN_VOL_RATIO,

        "bullish_candle":
        latest["Close"] > latest["Open"],

        "recovery":
        latest["Close"] > prev["Close"],

        "atr_contraction":
        latest["ATR_PCT"] < 0.03,
    }

    score = sum(conditions.values())

    if score < 8:
        return None

    ranking_score = (
        latest["RET_20D"] * 0.5
        + vol_ratio * 10
        + score * 5
    )

    entry = round(latest["Close"], 2)

    target = round(
        entry * (1 + TARGET_PCT),
        2
    )

    stop = round(
        entry * (1 - STOP_PCT),
        2
    )

    return {
        "symbol": symbol,
        "entry": entry,
        "target": target,
        "stop": stop,
        "score": round(ranking_score, 2),
        "rsi": round(
            latest["RSI14"],
            2
        )
    }


def run():

    if not market_is_bullish():

        send_telegram(
            "Market trend bearish. No trades."
        )

        return

    today = date.today()

    start_dt = (
        today - timedelta(days=LOOKBACK_DAYS)
    )

    symbols = fetch_nifty500()

    candidates = []

    for idx, symbol in enumerate(
        symbols,
        start=1
    ):

        try:

            print(
                f"[{idx}/{len(symbols)}] {symbol}"
            )

            yf_symbol = f"{symbol}.NS"

            df = fetch_history(
                yf_symbol,
                start_dt,
                today
            )

            result = analyze_stock(
                symbol,
                df
            )

            if result:
                candidates.append(result)

        except Exception as e:
            print(symbol, e)

        time.sleep(
            SLEEP_BETWEEN_SYMBOLS
        )

    candidates = sorted(
        candidates,
        key=lambda x: -x["score"]
    )[:TOP_N]

    if not candidates:

        send_telegram(
            "No quality setups found."
        )

        return

    message = (
        "📈 Ranked Swing Setups\n\n"
    )

    for idx, c in enumerate(
        candidates,
        start=1
    ):

        message += (
            f"{idx}. {c['symbol']}\n"
            f"Entry: ₹{c['entry']}\n"
            f"Target: ₹{c['target']}\n"
            f"Stop: ₹{c['stop']}\n"
            f"RSI: {c['rsi']}\n"
            f"Score: {c['score']}\n\n"
        )

    send_telegram(message)


if __name__ == "__main__":
    run()
