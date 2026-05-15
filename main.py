import os
import time
import math
import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
from nsepython import equity_history, index_history

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

NIFTY500_CSV_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"

LOOKBACK_DAYS = 430
TOP_N = 5
MIN_PRICE = 50
MIN_AVG_TRADED_VALUE = 20_00_00_000  # ₹20 crore
MIN_VOL_RATIO = 1.2
RSI_LOW = 55
RSI_HIGH = 68
MAX_DISTANCE_52W = 15.0
SLEEP_BETWEEN_SYMBOLS = 0.20


def log(msg):
    print(msg, flush=True)


def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log("Telegram credentials missing. Printing message instead:")
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
            log("Telegram message sent.")
    except Exception as e:
        log(f"Telegram error: {e}")


def fetch_nifty500_symbols():
    df = pd.read_csv(NIFTY500_CSV_URL)
    if "Symbol" not in df.columns:
        raise ValueError("Nifty 500 CSV format changed: 'Symbol' column not found.")
    symbols = df["Symbol"].dropna().astype(str).str.strip().unique().tolist()
    return symbols


def to_ddmmyyyy(dt: date) -> str:
    return dt.strftime("%d-%m-%Y")


def normalize_equity_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    rename_map = {
        "CH_TIMESTAMP": "Date",
        "Date": "Date",
        "CH_OPENING_PRICE": "Open",
        "Open Price": "Open",
        "Open": "Open",
        "CH_TRADE_HIGH_PRICE": "High",
        "High Price": "High",
        "High": "High",
        "CH_TRADE_LOW_PRICE": "Low",
        "Low Price": "Low",
        "Low": "Low",
        "CH_CLOSING_PRICE": "Close",
        "Close Price": "Close",
        "Close": "Close",
        "CH_LAST_TRADED_PRICE": "Close",
        "CH_TOT_TRADED_QTY": "Volume",
        "Total Traded Quantity": "Volume",
        "Volume": "Volume",
    }
    out = out.rename(columns=rename_map)

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        return pd.DataFrame()

    out = out[required].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", dayfirst=True)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna().sort_values("Date").reset_index(drop=True)
    return out


def normalize_index_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    rename_map = {
        "HistoricalDate": "Date",
        "Date": "Date",
        "OPEN": "Open",
        "HIGH": "High",
        "LOW": "Low",
        "CLOSE": "Close",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
    }
    out = out.rename(columns=rename_map)

    if "Date" not in out.columns or "Close" not in out.columns:
        return pd.DataFrame()

    keep = [c for c in ["Date", "Open", "High", "Low", "Close"] if c in out.columns]
    out = out[keep].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce", dayfirst=True)
    for c in keep:
        if c != "Date":
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna().sort_values("Date").reset_index(drop=True)
    return out


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

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


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200)
    df["RSI14"] = rsi(df["Close"], 14)
    _, _, hist = macd_hist(df["Close"])
    df["MACD_HIST"] = hist
    df["VOL_AVG20"] = df["Volume"].rolling(20).mean()
    df["AVG_PRICE20"] = df["Close"].rolling(20).mean()
    df["AVG_TRADED_VALUE20"] = df["VOL_AVG20"] * df["AVG_PRICE20"]
    df["52W_HIGH"] = df["High"].rolling(252, min_periods=100).max()
    df["RET_20D"] = df["Close"].pct_change(20) * 100
    return df


def fetch_stock_history(symbol: str, start_dt: date, end_dt: date) -> pd.DataFrame:
    raw = equity_history(symbol, "EQ", to_ddmmyyyy(start_dt), to_ddmmyyyy(end_dt))
    return normalize_equity_history(pd.DataFrame(raw))


def fetch_nifty_history(start_dt: date, end_dt: date) -> pd.DataFrame:
    raw = index_history("NIFTY 50", start_dt.strftime("%d-%b-%Y"), end_dt.strftime("%d-%b-%Y"))
    return normalize_index_history(pd.DataFrame(raw))


def is_market_bullish(nifty_df: pd.DataFrame):
    if nifty_df.empty or len(nifty_df) < 60:
        return False, None, None

    nifty_df = nifty_df.copy()
    nifty_df["EMA50"] = ema(nifty_df["Close"], 50)

    latest = nifty_df.iloc[-1]
    return float(latest["Close"]) > float(latest["EMA50"]), float(latest["Close"]), float(latest["EMA50"])


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


def analyze_stock(symbol: str, stock_df: pd.DataFrame, nifty_20d_ret: float):
    if stock_df.empty or len(stock_df) < 220:
        return None, f"{symbol}: insufficient history"

    df = add_indicators(stock_df)
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    if pd.isna(latest["VOL_AVG20"]) or pd.isna(latest["52W_HIGH"]):
        return None, f"{symbol}: indicators incomplete"

    vol_ratio = float(latest["Volume"] / latest["VOL_AVG20"]) if latest["VOL_AVG20"] > 0 else 0.0
    dist_52w_high = float((latest["52W_HIGH"] - latest["Close"]) / latest["52W_HIGH"] * 100) if latest["52W_HIGH"] > 0 else 999.0
    rel_strength = float(latest["RET_20D"] - nifty_20d_ret)

    checks = {
        "price_min": latest["Close"] >= MIN_PRICE,
        "above_ema20": latest["Close"] > latest["EMA20"],
        "above_ema50": latest["Close"] > latest["EMA50"],
        "above_ema200": latest["Close"] > latest["EMA200"],
        "ema_stack": latest["EMA20"] > latest["EMA50"] > latest["EMA200"],
        "rsi_band": RSI_LOW <= latest["RSI14"] <= RSI_HIGH,
        "macd_positive": latest["MACD_HIST"] > 0,
        "macd_rising": latest["MACD_HIST"] > prev["MACD_HIST"],
        "liquidity": latest["AVG_TRADED_VALUE20"] >= MIN_AVG_TRADED_VALUE,
        "vol_ratio": vol_ratio >= MIN_VOL_RATIO,
        "near_52w": dist_52w_high <= MAX_DISTANCE_52W,
        "rel_strength": rel_strength > 0,
    }

    passed = all(checks.values())

    debug_line = (
        f"{symbol} | Close={latest['Close']:.2f} | "
        f"EMA20={latest['EMA20']:.2f} EMA50={latest['EMA50']:.2f} EMA200={latest['EMA200']:.2f} | "
        f"RSI={latest['RSI14']:.2f} | "
        f"MACDh={latest['MACD_HIST']:.4f}/{prev['MACD_HIST']:.4f} | "
        f"VolRatio={vol_ratio:.2f} | "
        f"Dist52W={dist_52w_high:.2f}% | "
        f"RS20={rel_strength:.2f}% | "
        f"ADV20={latest['AVG_TRADED_VALUE20'] / 1e7:.2f}Cr | "
        f"PASS={passed}"
    )

    if not passed:
        return None, debug_line

    latest_data = latest.copy()
    latest_data["VOL_RATIO"] = vol_ratio
    latest_data["DIST_52W_HIGH"] = dist_52w_high
    latest_data["REL_STRENGTH_20D"] = rel_strength

    score = score_candidate(latest_data, rel_strength)
    entry = float(latest["Close"])
    target = round(entry * 1.05, 2)
    stop = round(entry * (1 - 0.035), 2)

    result = {
        "symbol": symbol,
        "entry": round(entry, 2),
        "target": target,
        "stop": stop,
        "rsi": round(float(latest["RSI14"]), 2),
        "vol_ratio": round(vol_ratio, 2),
        "dist_52w_high": round(dist_52w_high, 2),
        "rel_strength_20d": round(rel_strength, 2),
        "score": score
    }

    return result, debug_line


def build_alert(candidates, nifty_close, nifty_ema50):
    header = (
        f"📈 <b>NSE 500 Swing Scanner</b>\n"
        f"Nifty50: {nifty_close:.2f} | EMA50: {nifty_ema50:.2f}\n"
        f"Top candidates today: {len(candidates)}\n\n"
    )

    lines = []
    for i, c in enumerate(candidates, start=1):
        lines.append(
            f"<b>{i}. {c['symbol']}</b>\n"
            f"Entry: ₹{c['entry']}\n"
            f"Target: ₹{c['target']} (+5%)\n"
            f"Stop: ₹{c['stop']} (-3.5%)\n"
            f"RSI: {c['rsi']} | VolRatio: {c['vol_ratio']}x\n"
            f"Dist 52W High: {c['dist_52w_high']}%\n"
            f"20D Relative Strength: {c['rel_strength_20d']}%\n"
            f"Score: {c['score']}\n"
        )

    return header + "\n".join(lines)


def build_no_trade_message(reason, nifty_close=None, nifty_ema50=None):
    parts = [f"ℹ️ NSE 500 scanner ran successfully.\n{reason}"]
    if nifty_close is not None and nifty_ema50 is not None:
        parts.append(f"Nifty50: {nifty_close:.2f} | EMA50: {nifty_ema50:.2f}")
    return "\n".join(parts)


def run():
    today = date.today()
    start_dt = today - timedelta(days=LOOKBACK_DAYS)
    end_dt = today

    log("Fetching Nifty 50 history...")
    nifty_df = fetch_nifty_history(start_dt, end_dt)
    bullish, nifty_close, nifty_ema50 = is_market_bullish(nifty_df)

    if not bullish:
        msg = build_no_trade_message(
            "Market regime filter blocked longs today.",
            nifty_close,
            nifty_ema50
        )
        send_telegram(msg)
        return

    nifty_df = add_indicators(nifty_df.rename(columns={"Close": "Close", "High": "High", "Low": "Low", "Open": "Open"}))
    nifty_20d_ret = float(nifty_df.iloc[-1]["RET_20D"])

    log("Fetching Nifty 500 constituents...")
    symbols = fetch_nifty500_symbols()
    log(f"Total symbols fetched: {len(symbols)}")

    candidates = []

    for idx, symbol in enumerate(symbols, start=1):
        try:
            log(f"[{idx}/{len(symbols)}] Fetching {symbol}")
            stock_df = fetch_stock_history(symbol, start_dt, end_dt)
            result, debug_line = analyze_stock(symbol, stock_df, nifty_20d_ret)
            log(debug_line)
            if result:
                candidates.append(result)
        except Exception as e:
            log(f"{symbol}: ERROR - {e}")

        time.sleep(SLEEP_BETWEEN_SYMBOLS)

    if not candidates:
        msg = build_no_trade_message(
            "No stocks passed all filters today.",
            nifty_close,
            nifty_ema50
        )
        send_telegram(msg)
        return

    candidates = sorted(
        candidates,
        key=lambda x: (-x["score"], -x["rel_strength_20d"], x["dist_52w_high"])
    )[:TOP_N]

    message = build_alert(candidates, nifty_close, nifty_ema50)
    send_telegram(message)


if __name__ == "__main__":
    run()
