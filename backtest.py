import os
files = [
    f for f in os.listdir(DATA_FOLDER)
    if f.endswith(".csv")
]

print(f"Backtesting {len(files)} stocks")

for file in files:
    symbol = file.replace(".csv", "")

    try:
        df = pd.read_csv(f"{DATA_FOLDER}/{file}")

        if len(df) < 300:
            continue

        df["Date"] = pd.to_datetime(df["Date"])

        df = add_indicators(df)

        df = df.dropna().reset_index(drop=True)

        for idx in range(1, len(df) - MAX_HOLD_DAYS - 1):

            row = df.iloc[idx]
            prev_row = df.iloc[idx - 1]

            if signal(row, prev_row):
                simulate_trade(df, idx, symbol)

    except Exception as e:
        print(symbol, e)



trades_df = pd.DataFrame(trades)

trades_df.to_csv("trades.csv", index=False)


if len(trades_df) == 0:
    print("No trades generated")
    exit()


wins = trades_df[trades_df["return_pct"] > 0]
losses = trades_df[trades_df["return_pct"] <= 0]

win_rate = len(wins) / len(trades_df) * 100
avg_return = trades_df["return_pct"].mean()
avg_win = wins["return_pct"].mean() if len(wins) > 0 else 0
avg_loss = losses["return_pct"].mean() if len(losses) > 0 else 0

profit_factor = (
    wins["return_pct"].sum()
    / abs(losses["return_pct"].sum())
    if len(losses) > 0
    else 0
)

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
