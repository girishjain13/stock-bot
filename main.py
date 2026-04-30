import requests
import pandas as pd
import numpy as np
import yfinance as yf
import concurrent.futures
import os

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ================= TELEGRAM =================
def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg})


# ================= NSE FETCH =================
def fetch_nse_symbols():
    url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }

    session = requests.Session()
    session.get("https://www.nseindia.com", headers=headers)

    data = session.get(url, headers=headers).json()

    symbols = []
    for item in data["data"]:
        sym = item["symbol"] + ".NS"
        symbols.append(sym)

    return symbols


# ================= MARKET FILTER =================
def is_market_bullish():
    df = yf.download("^NSEI", period="6mo", progress=False)
    df["EMA100"] = df["Close"].ewm(span=100).mean()
    return float(df["Close"].iloc[-1]) > float(df["EMA100"].iloc[-1])


# ================= INDICATORS =================
def add_indicators(df):
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA200"] = df["Close"].ewm(span=200).mean()

    df["VOL_AVG"] = df["Volume"].rolling(20).mean()

    return df


# ================= FAST DATA FETCH =================
def fetch_stock_data(stock):
    try:
        df = yf.download(stock, period="3mo", progress=False)
        if len(df) < 50:
            return None
        df = add_indicators(df)

        latest = df.iloc[-1]

        # Pre-filter (important for speed)
        if (
            latest["Close"] > latest["EMA50"] and
            latest["Volume"] > latest["VOL_AVG"]
        ):
            return stock
    except:
        return None

    return None


# ================= BUILD DYNAMIC UNIVERSE =================
def get_dynamic_universe(symbols):
    selected = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(fetch_stock_data, symbols)

    for r in results:
        if r:
            selected.append(r)

    return selected[:120]  # cap for speed


# ================= SECTOR STRENGTH =================
SECTOR_INDEX = {
    "IT": "^CNXIT",
    "BANK": "^NSEBANK",
    "AUTO": "^CNXAUTO",
    "FMCG": "^CNXFMCG",
    "PHARMA": "^CNXPHARMA"
}

def get_strongest_sector():
    perf = {}

    for name, symbol in SECTOR_INDEX.items():
        try:
            df = yf.download(symbol, period="5d", progress=False)
            change = (df["Close"].iloc[-1] - df["Close"].iloc[-3]) / df["Close"].iloc[-3] * 100
            perf[name] = float(change)
        except:
            continue

    return max(perf, key=perf.get)


# ================= SECTOR FILTER =================
def filter_by_sector(stocks, sector):
    keywords = {
        "IT": ["TCS","INFY","WIPRO","HCLTECH","TECHM"],
        "BANK": ["BANK","SBIN","ICICI","HDFC","AXIS"],
        "AUTO": ["AUTO","MOTOR","M&M","MARUTI"],
        "FMCG": ["HUL","ITC","NESTLE","DABUR"],
        "PHARMA": ["PHARMA","DRREDDY","CIPLA"]
    }

    keys = keywords.get(sector, [])
    return [s for s in stocks if any(k in s for k in keys)]


# ================= SCORING =================
def score_stock(df):
    latest = df.iloc[-1]
    score = 0

    if latest["EMA20"] > latest["EMA50"] > latest["EMA200"]:
        score += 3

    return score


# ================= TRADE =================
def generate_trade(df):
    latest = df.iloc[-1]

    entry = round(latest["Close"], 2)
    atr = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]

    sl = round(entry - (2 * atr), 2)
    target = round(entry * 1.10, 2)

    rr = round((target - entry) / (entry - sl), 2)

    return entry, target, sl, rr


# ================= MAIN =================
def run():

    if not is_market_bullish():
        send_telegram("⚠️ Market not favorable.")
        return

    print("Fetching NSE universe...")
    symbols = fetch_nse_symbols()

    print("Filtering active stocks...")
    universe = get_dynamic_universe(symbols)

    if not universe:
        send_telegram("No strong stocks today.")
        return

    sector = get_strongest_sector()
    sector_stocks = filter_by_sector(universe, sector)

    best = None

    for stock in sector_stocks:
        try:
            df = yf.download(stock, period="6mo", progress=False)
            df = add_indicators(df)

            score = score_stock(df)
            entry, target, sl, rr = generate_trade(df)

            if rr < 1.5:
                continue

            if not best or score > best["score"]:
                best = {
                    "stock": stock.replace(".NS",""),
                    "entry": entry,
                    "target": target,
                    "sl": sl,
                    "rr": rr
                }

        except:
            continue

    if not best:
        send_telegram("No trade setup.")
        return

    msg = f"""📈 Dynamic NSE Trade

Sector: {sector}
Stock: {best['stock']}

Entry: ₹{best['entry']}
Target: ₹{best['target']} (~10%)
Stop Loss: ₹{best['sl']}
R:R: {best['rr']}

⏳ Holding: 2–3 months
"""

    send_telegram(msg)


if __name__ == "__main__":
    run()