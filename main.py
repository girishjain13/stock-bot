import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os

# ================= CONFIG =================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# 🔍 DEBUG (TEMPORARY - REMOVE LATER)
print("DEBUG TOKEN:", TELEGRAM_TOKEN)
print("DEBUG CHAT ID:", TELEGRAM_CHAT_ID)

SECTOR_MAP = {
    "^CNXIT": ["TCS.NS", "INFY.NS"],
    "^NSEBANK": ["HDFCBANK.NS", "ICICIBANK.NS"],
    "^CNXAUTO": ["TATAMOTORS.NS", "M&M.NS"],
    "^CNXFMCG": ["HUL.NS", "ITC.NS"],
}

# ================= TELEGRAM =================
def send_telegram(msg):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ Missing Telegram credentials")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    try:
        response = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": msg
        })

        print("Telegram response:", response.text)

    except Exception as e:
        print("Telegram error:", str(e))


# ================= MARKET FILTER =================
def is_market_bullish():
    try:
        df = yf.download("^NSEI", period="3mo", progress=False)
        df["EMA50"] = df["Close"].ewm(span=50).mean()
        return df.iloc[-1]["Close"] > df.iloc[-1]["EMA50"]
    except Exception as e:
        print("Market check error:", e)
        return False


# ================= SECTOR =================
def get_top_sector():
    perf = {}
    for sector in SECTOR_MAP:
        try:
            df = yf.download(sector, period="3d", progress=False)
            if len(df) >= 2:
                change = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100
                perf[sector] = float(change)
        except:
            continue

    if not perf:
        return None, {}

    return max(perf, key=perf.get), perf


# ================= INDICATORS =================
def add_indicators(df):
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA200"] = df["Close"].ewm(span=200).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["VOL_AVG"] = df["Volume"].rolling(20).mean()

    df["TR"] = np.maximum(df["High"] - df["Low"],
                         np.maximum(abs(df["High"] - df["Close"].shift()),
                                    abs(df["Low"] - df["Close"].shift())))
    df["ATR"] = df["TR"].rolling(14).mean()

    return df


# ================= NEWS =================
def get_news_score(symbol):
    if not NEWS_API_KEY:
        return 0

    try:
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}"
        articles = requests.get(url, timeout=5).json().get("articles", [])[:3]

        score = 0
        for a in articles:
            t = a.get("title", "").lower()
            if any(x in t for x in ["growth","profit","upgrade","order"]):
                score += 1
            if any(x in t for x in ["loss","fraud","downgrade","fall"]):
                score -= 1
        return score
    except:
        return 0


# ================= SCORING =================
def score_stock(df, news_score):
    latest = df.iloc[-1]
    score = 0

    if latest["EMA20"] > latest["EMA50"] > latest["EMA200"]:
        score += 3

    pullback = (
        latest["Close"] > latest["EMA50"] and
        latest["Close"] < latest["EMA20"] * 1.02 and
        latest["RSI"] > 50
    )
    if pullback:
        score += 3

    if 60 <= latest["RSI"] <= 68:
        score += 2

    if latest["Volume"] > 1.5 * latest["VOL_AVG"]:
        score += 2

    candle = (
        latest["Close"] > latest["Open"] and
        (latest["Close"] - latest["Low"]) > (latest["High"] - latest["Close"])
    )
    if candle:
        score += 2

    score += news_score
    return score


# ================= TRADE =================
def generate_trade(df):
    latest = df.iloc[-1]
    entry = round(latest["Close"], 2)
    sl = round(entry - (1.2 * latest["ATR"]), 2)
    target = round(entry + (entry - sl) * 1.8, 2)
    rr = round((target - entry) / (entry - sl), 2)
    return entry, target, sl, rr


# ================= MAIN =================
def run():
    if not is_market_bullish():
        send_telegram("⚠️ Market not favorable. No trade today.")
        return

    top_sector, _ = get_top_sector()
    if not top_sector:
        send_telegram("⚠️ Error fetching sector data.")
        return

    best = None

    for stock in SECTOR_MAP[top_sector]:
        try:
            df = yf.download(stock, period="6mo", progress=False)
            if len(df) < 100:
                continue

            df = add_indicators(df)
            news = get_news_score(stock.replace(".NS",""))
            score = score_stock(df, news)

            if score < 7:
                continue

            entry, target, sl, rr = generate_trade(df)
            if rr < 1.5:
                continue

            if not best or score > best["score"]:
                best = {
                    "stock": stock.replace(".NS",""),
                    "score": score,
                    "entry": entry,
                    "target": target,
                    "sl": sl,
                    "rr": rr
                }

        except Exception as e:
            print("Error with stock:", stock, e)
            continue

    if not best:
        send_telegram("No high-confidence trade today.")
        return

    msg = f"""📈 High-Probability Trade

Stock: {best['stock']}
Entry: ₹{best['entry']}
Target: ₹{best['target']}
Stop Loss: ₹{best['sl']}
R:R: {best['rr']}

⚠️ Rule-based system.
"""

    send_telegram(msg)


if __name__ == "__main__":
    run()