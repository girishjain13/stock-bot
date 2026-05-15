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

NIFTY500_CSV_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

LOOKBACK_DAYS = 430
TOP_N = 5

MIN_PRICE = 50
MIN_AVG_TRADED_VALUE = 20_00_00_000
MIN_VOL_RATIO = 1.2

RSI_LOW = 55
RSI_HIGH = 68

MAX_DISTANCE_52W = 15.0

SLEEP_BETWEEN_SYMBOLS = 0.5


def log(msg):
    print(msg, flush=True)


def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log(message)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }

    try:
        r = requests.post(url, json=payload, timeout=30)

        if r.status_code != 200:
            log(f"Telegram failed: {r.status_code} | {r.text}")
        else:
            log("Telegram message sent")

    except Exception as e:
        log(f"Telegram error: {e}")


def fetch_nifty500_symbols():
    df = pd.read_csv(NIFTY500_CSV_URL)

    if "Symbol" not in df.columns:
        raise Exception("Symbol column missing")

    symbols = (
        df["Symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    return symbols


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

    return macd_line, signal, hist


def add_indicators(df):
    df = df.copy()

    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200)

    df["RSI14"] = rsi(df["Close"], 14)

    _, _, hist = macd_hist(df["Close"])
    df["MACD_HIST"] = hist

    df["VOL_AVG20"] = df["Volume"].rolling(20).mean()

    df["AVG_PRICE20"] = df["Close"].rolling(20).mean()

    df["AVG_TRADED_VALUE20"] = (
        df["VOL_AVG20"] * df["AVG_PRICE20"]
    )

    df["52W_HIGH"] = df["High"].rolling(
        252,
        min_periods=100
    ).max()

    df["RET_20D"] = df["Close"].pct_change(20) * 100

    return df


def fetch_stock_history(symbol, start_dt, end_dt):
    try:
        yf_symbol = f"{symbol}.NS"

        df = yf.download(
            yf_symbol,
            start=start_dt,
            end=end_dt,
            progress=False,
            auto_adjust=False,
            threads=False
        )

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()

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
            log(f"{symbol}: Missing columns {missing}")
            return pd.DataFrame()

        df = df[required_cols]

        df = df.dropna()

        return df

    except Exception as e:
        log(f"{symbol} fetch failed: {e}")
        return pd.DataFrame()


def fetch_nifty_history(start_dt, end_dt):
    try:
        df = yf.download(
            "^NSEI",
            start=start_dt,
            end=end_dt,
            progress=False,
            auto_adjust=False,
            threads=False
        )

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()

        df.columns = [str(c).strip() for c in df.columns]

        required_cols = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close"
        ]

        missing = [
            c for c in required_cols
            if c not in df.columns
        ]

        if missing:
            log(f"NIFTY missing columns {missing}")
            return pd.DataFrame()

        df = df[required_cols]

        df = df.dropna()

        return df

    except Exception as e:
        log(f"NIFTY fetch failed: {e}")
        return pd.DataFrame()


def is_market_bullish(nifty_df):
    if nifty_df.empty or len(nifty_df) < 60:
        return False, None, None

    nifty_df = nifty_df.copy()

    nifty_df["EMA50"] = ema(
        nifty_df["Close"],
        50
    )

    latest = nifty_df.iloc[-1]

    bullish = latest["Close"] > latest["EMA50"]

    return bullish, latest["Close"], latest["EMA50"]


def score_candidate(latest, rel_strength):
    score = 0

    if latest["Close"] > latest["EMA20"]:
        score += 1

    if latest["Close"] > latest["EMA50"]:
        score += 1

    if latest["Close"] > latest["EMA200"]:
        score += 1

    if latest["EMA20"] > latest["EMA50"] > latest["EMA200"]:
        score += 2

    if RSI_LOW <= latest["RSI14"] <= RSI_HIGH:
        score += 1

    if latest["MACD_HIST"] > 0:
        score += 1

    if latest["VOL_RATIO"] >= MIN_VOL_RATIO:
        score += 1

    if latest["DIST_52W_HIGH"] <= MAX_DISTANCE_52W:
        score += 1

    if rel_strength > 0:
        score += 2

    return score


def analyze_stock(symbol, stock_df, nifty_20d_ret):
    if stock_df.empty or len(stock_df) < 220:
        return None

    df = add_indicators(stock_df)

    latest = df.iloc[-1]
    prev = df.iloc[-2]

    if pd.isna(latest["VOL_AVG20"]):
        return None

    vol_ratio = (
        latest["Volume"] / latest["VOL_AVG20"]
        if latest["VOL_AVG20"] > 0
        else 0
    )

    dist_52w_high = (
        (latest["52W_HIGH"] - latest["Close"])
        / latest["52W_HIGH"]
        * 100
    )

    rel_strength = (
        latest["RET_20D"] - nifty_20d_ret
    )

    checks = {
        "price": latest["Close"] >= MIN_PRICE,
        "ema20": latest["Close"] > latest["EMA20"],
        "ema50": latest["Close"] > latest["EMA50"],
        "ema200": latest["Close"] > latest["EMA200"],
        "stack": latest["EMA20"] > latest["EMA50"] > latest["EMA200"],
        "rsi": RSI_LOW <= latest["RSI14"] <= RSI_HIGH,
        "macd": latest["MACD_HIST"] > 0,
        "macd_rising": latest["MACD_HIST"] > prev["MACD_HIST"],
        "liquidity": latest["AVG_TRADED_VALUE20"] >= MIN_AVG_TRADED_VALUE,
        "volume": vol_ratio >= MIN_VOL_RATIO,
        "near_high": dist_52w_high <= MAX_DISTANCE_52W,
        "rs": rel_strength > 0
    }

    passed = sum(checks.values()) >= 9

    if not passed:
        return None

    latest["VOL_RATIO"] = vol_ratio
    latest["DIST_52W_HIGH"] = dist_52w_high

    score = score_candidate(latest, rel_strength)

    entry = round(latest["Close"], 2)

    result = {
        "symbol": symbol,
        "entry": entry,
        "target": round(entry * 1.05, 2),
        "stop": round(entry * 0.965, 2),
        "rsi": round(latest["RSI14"], 2),
        "vol_ratio": round(vol_ratio, 2),
        "dist_52w_high": round(dist_52w_high, 2),
        "rel_strength_20d": round(rel_strength, 2),
        "score": score
    }

    return result


def build_alert(candidates, nifty_close, nifty_ema50):
    header = (
        f"📈 NSE 500 Swing Scanner\n"
        f"Nifty50: {nifty_close:.2f}\n"
        f"EMA50: {nifty_ema50:.2f}\n\n"
    )

    lines = []

    for idx, c in enumerate(candidates, start=1):
        lines.append(
            f"{idx}. {c['symbol']}\n"
            f"Entry: ₹{c['entry']}\n"
            f"Target: ₹{c['target']}\n"
            f"Stop: ₹{c['stop']}\n"
            f"RSI: {c['rsi']}\n"
            f"Volume Ratio: {c['vol_ratio']}x\n"
            f"Relative Strength: {c['rel_strength_20d']}%\n"
            f"Score: {c['score']}\n"
        )

    return header + "\n".join(lines)


def run():
    today = date.today()

    start_dt = today - timedelta(days=LOOKBACK_DAYS)

    log("Fetching NIFTY history")

    nifty_df = fetch_nifty_history(
        start_dt,
        today
    )

    bullish, nifty_close, nifty_ema50 = is_market_bullish(
        nifty_df
    )

    if not bullish:
        send_telegram(
            f"Market not bullish.\n"
            f"NIFTY: {nifty_close}\n"
            f"EMA50: {nifty_ema50}"
        )
        return

    nifty_df = add_indicators(nifty_df)

    nifty_20d_ret = nifty_df.iloc[-1]["RET_20D"]

    symbols = fetch_nifty500_symbols()

    log(f"Scanning {len(symbols)} stocks")

    candidates = []

    for idx, symbol in enumerate(symbols, start=1):

        try:
            log(f"[{idx}/{len(symbols)}] {symbol}")

            stock_df = fetch_stock_history(
                symbol,
                start_dt,
                today
            )

            result = analyze_stock(
                symbol,
                stock_df,
                nifty_20d_ret
            )

            if result:
                candidates.append(result)

        except Exception as e:
            log(f"{symbol} failed: {e}")

        time.sleep(SLEEP_BETWEEN_SYMBOLS)

    if not candidates:
        send_telegram("No stocks passed filters today")
        return

    candidates = sorted(
        candidates,
        key=lambda x: (
            -x["score"],
            -x["rel_strength_20d"]
        )
    )[:TOP_N]

    message = build_alert(
        candidates,
        nifty_close,
        nifty_ema50
    )

    send_telegram(message)


if __name__ == "__main__":
    run()
