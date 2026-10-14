from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class PortfolioConfig:
    initial_cash: float = 100_000.0
    commission_rate: float = 0.0005
    fixed_spread_bps: float = 2.0
    slippage_impact_factor: float = 0.1
    max_active_positions: int = 5    # 5 slot da 20% ciascuno
    atr_stop_multiplier: float = 2.5


def prepare_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(ticker, start=start, end=end, progress=False)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    # Calcolo ATR
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=14).mean().bfill()

    # Indicatori Trend Following
    lookback = 50
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["SMA200_slope"] = df["SMA200"] - df["SMA200"].shift(20)
    df["Max50"] = df["Close"].shift(1).rolling(lookback).max()
    df["SMA50"] = df["Close"].rolling(lookback).mean()

    # Segnali di Trend Following
    buy_cond = (df["Close"] > df["Max50"]) & (df["Close"] > df["SMA200"]) & (df["SMA200_slope"] > 0)
    exit_cond = df["Close"] < df["SMA50"]

    signal = pd.Series(0.0, index=df.index)
    state = 0.0
    for i in range(len(df)):
        if buy_cond.iloc[i]:
            state = 1.0
        elif exit_cond.iloc[i]:
            state = 0.0
        signal.iloc[i] = state
    df["Signal"] = signal
    return df


def run_portfolio_backtest(tickers: list[str], start_date: str, end_date: str, cfg: PortfolioConfig):
    print(f"Scaricamento dati multi-settore per: {', '.join(tickers)}...")
    stock_data = {}
    for t in tickers:
        df = prepare_stock_data(t, "2017-01-01", end_date)
        stock_data[t] = df.loc[start_date:].copy()

    # Allineamento su date di contrattazione comuni
    common_index = stock_data[tickers[0]].index
    for t in tickers[1:]:
        common_index = common_index.intersection(stock_data[t].index)

    for t in tickers:
        stock_data[t] = stock_data[t].loc[common_index]

    n = len(common_index)
    portfolio_equity = np.zeros(n)
    cash = cfg.initial_cash
    positions = {t: 0.0 for t in tickers}
    peak_prices = {t: 0.0 for t in tickers}
    frictions_total = 0.0

    for i in range(1, n):
        current_equity = cash
        for t in tickers:
            current_equity += positions[t] * stock_data[t]["Close"].iloc[i - 1]

        slot_capital = current_equity / cfg.max_active_positions

        for t in tickers:
            df_t = stock_data[t]
            open_p = df_t["Open"].iloc[i]
            prev_close = df_t["Close"].iloc[i - 1]
            atr = df_t["ATR"].iloc[i]
            raw_sig = df_t["Signal"].iloc[i - 1]

            # 1. Trailing Stop agganciato alla volatilita
            stopped_out = False
            if positions[t] > 0:
                peak_prices[t] = max(peak_prices[t], prev_close)
                stop_threshold = peak_prices[t] - (atr * cfg.atr_stop_multiplier)
                if prev_close < stop_threshold:
                    stopped_out = True

            # 2. Decisione sul target delle quote
            target_shares = positions[t]
            if raw_sig == 1.0 and not stopped_out and positions[t] == 0.0:
                target_shares = np.floor(slot_capital / open_p)
                peak_prices[t] = open_p
            elif stopped_out or raw_sig == 0.0:
                target_shares = 0.0
                peak_prices[t] = 0.0

            trade_shares = target_shares - positions[t]

            # 3. Esecuzione ordini e attriti
            if trade_shares != 0:
                half_spread = open_p * (cfg.fixed_spread_bps / 10000.0)
                rel_vol = atr / open_p
                slippage = open_p * rel_vol * cfg.slippage_impact_factor

                exec_price = (open_p + half_spread + slippage) if trade_shares > 0 else (open_p - half_spread - slippage)
                fee = abs(trade_shares) * exec_price * cfg.commission_rate

                cash -= (trade_shares * exec_price) + fee
                positions[t] = target_shares
                frictions_total += fee + (abs(exec_price - open_p) * abs(trade_shares))

        # Calcolo Equity giornaliera
        e = cash
        for t in tickers:
            e += positions[t] * stock_data[t]["Close"].iloc[i]
        portfolio_equity[i] = e

    portfolio_equity[0] = cfg.initial_cash
    res = pd.DataFrame({"equity": portfolio_equity}, index=common_index)
    return res, frictions_total


# Configurazione Paniere Multi-Settore
# AAPL, NVDA (Tech), XLE (Energia), GLD (Oro), XLV (Sanita)
tickers = ["AAPL", "NVDA", "XLE", "GLD", "XLV"]
cfg = PortfolioConfig(initial_cash=100_000.0, max_active_positions=5)

results, frictions = run_portfolio_backtest(
    tickers=tickers,
    start_date="2018-01-01",
    end_date="2025-01-01",
    cfg=cfg
)

# Calcolo Metriche
ret = (results["equity"].iloc[-1] / results["equity"].iloc[0]) - 1
peak = results["equity"].cummax()
max_dd = ((results["equity"] - peak) / peak).min()
daily_ret = results["equity"].pct_change().fillna(0)
sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0

print("\n=== METRICHE PORTAFOGLIO MULTI-SETTORE (2018-2025) ===")
print(f"Capitale Iniziale: ${cfg.initial_cash:,.2f}")
print(f"Capitale Finale: ${results['equity'].iloc[-1]:,.2f}")
print(f"Rendimento Totale Netto: {ret * 100:.2f}%")
print(f"Max Drawdown (Rischio Massimo): {max_dd * 100:.2f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Attrito Totale Speso: ${frictions:,.2f}")

# Visualizzazione
plt.figure(figsize=(12, 6))
plt.plot(results.index, results["equity"], label="Portafoglio Multi-Settore (Equity)", color="#2ca02c", linewidth=2)
plt.title("Backtest Multi-Settore: Tech, Energia, Oro, Sanità (2018-2025)", fontsize=13, weight="bold")
plt.ylabel("Valore Portafoglio ($)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()