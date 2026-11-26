import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

# --- CONFIGURAZIONE GENERALE ---
TICKERS = ["AAPL", "NVDA", "XLE", "GLD", "XLV"]
BENCHMARK_TICKER = "SPY"
INITIAL_CAPITAL = 100000.0
START_DATE = "2026-10-15"  # Data esatta di inizio del primo test
HISTORY_FILE = "arena_history.json"

# --- MODELLO COMMISSIONI & SLIPPAGE REALISTICO (Broker USA Pro) ---
COMM_PER_SHARE = 0.005     # $0.005 per azione
MIN_COMMISSION = 1.00      # Minimo $1.00 ad eseguito
SLIPPAGE_PCT = 0.0003      # 0.03% di slippage medio sul prezzo di esecuzione

STRATEGIES = {
    "trend_following": "1. Trend Following (Donchian 50d)",
    "mean_reversion": "2. Mean Reversion (RSI-2)",
    "vol_breakout": "3. Volatility Breakout (Bollinger)",
    "risk_parity": "4. Risk Parity (1/ATR)",
    "rel_momentum": "5. Momentum Relativo (Top 2)",
    "ensemble_50_50": "6. Ensemble 50/50 (Trend + Reversion)",
    "buy_and_hold": "7. Buy & Hold (5 Asset)",
    "sp500": "Benchmark S&P 500 (SPY)"
}


def calc_trade_cost(shares, price):
    """Calcola commissione broker + slippage stimato in dollari."""
    if shares <= 0:
        return 0.0
    comm = max(MIN_COMMISSION, shares * COMM_PER_SHARE)
    slippage = (shares * price) * SLIPPAGE_PCT
    return round(comm + slippage, 2)


def compute_indicators(df):
    """Calcola indicatori tecnici su serie storica pulita."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["SMA5"] = close.rolling(5).mean()
    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()

    # Donchian 50d (shift(1) evita look-ahead bias)
    df["DonchianHigh50"] = high.rolling(50).max().shift(1)

    # ATR 14 giorni
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    # RSI a 2 periodi
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(2).mean()
    loss = (-delta.clip(upper=0)).rolling(2).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI2"] = (100 - (100 / (1 + rs))).fillna(50)

    # Bande di Bollinger (20, 2)
    std20 = close.rolling(20).std()
    df["BB_Upper"] = df["SMA20"] + 2 * std20
    df["BB_Lower"] = df["SMA20"] - 2 * std20
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["SMA20"]
    df["BB_Width_Min60"] = df["BB_Width"].rolling(60).min().shift(1)

    # Rendimento a 60 sedute per Momentum Relativo
    df["ROC60"] = close.pct_change(60)

    return df


def download_market_data():
    """Scarica 420 giorni e allinea tutti i titoli al calendario ufficiale di SPY."""
    all_tickers = TICKERS + [BENCHMARK_TICKER]
    start_fetch = (datetime.strptime(START_DATE, "%Y-%m-%d") - timedelta(days=420)).strftime("%Y-%m-%d")
    raw = yf.download(all_tickers, start=start_fetch, progress=False, group_by="ticker", auto_adjust=True)

    # Estrae e normalizza prima il calendario ufficiale di SPY
    spy_raw = raw[BENCHMARK_TICKER].copy()
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = spy_raw.columns.get_level_values(-1)
    spy_raw.index = pd.to_datetime(spy_raw.index).tz_localize(None).normalize()
    spy_raw = spy_raw[~spy_raw.index.duplicated(keep="last")].dropna(subset=["Close"])
    base_index = spy_raw.index

    data = {}
    for t in all_tickers:
        df = raw[t].copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df[~df.index.duplicated(keep="last")]
        # Allinea esattamente alle date di SPY riempiendo eventuali singoli buchi
        df = df.reindex(base_index).ffill().bfill()
        data[t] = compute_indicators(df)
    return data


def init_portfolio(capital=INITIAL_CAPITAL):
    return {
        "cash": capital,
        "positions": {},
        "total_fees": 0.0,
        "trades_count": 0
    }


def portfolio_net_value(pf, prices):
    val = pf["cash"]
    for t, pos in pf["positions"].items():
        val += pos["shares"] * prices[t]
    return val


def build_position_details(pf, prices):
    details = []
    for t, pos in pf["positions"].items():
        curr = prices[t]
        tot_val = pos["shares"] * curr
        pnl_pct = ((curr - pos["entry_price"]) / pos["entry_price"]) * 100
        pnl_usd = tot_val - (pos["shares"] * pos["entry_price"])
        details.append({
            "ticker": t,
            "shares": pos["shares"],
            "entry_price": round(pos["entry_price"], 2),
            "close_price": round(curr, 2),
            "total_value": round(tot_val, 2),
            "pnl_pct": round(pnl_pct, 2),
            "pnl_usd": round(pnl_usd, 2)
        })
    return details


def execute_sell(pf, ticker, price):
    pos = pf["positions"][ticker]
    fee = calc_trade_cost(pos["shares"], price)
    pf["cash"] += (pos["shares"] * price) - fee
    pf["total_fees"] += fee
    pf["trades_count"] += 1
    del pf["positions"][ticker]
    return fee


def execute_buy(pf, ticker, budget, price, extra_fields=None):
    shares = int(budget // (price * (1 + SLIPPAGE_PCT) + COMM_PER_SHARE))
    if shares <= 0:
        return 0.0
    fee = calc_trade_cost(shares, price)
    pf["cash"] -= (shares * price + fee)
    pf["total_fees"] += fee
    pf["trades_count"] += 1
    pos_data = {"shares": shares, "entry_price": price}
    if extra_fields:
        pos_data.update(extra_fields)
    pf["positions"][ticker] = pos_data
    return fee


def run_arena():
    print("[+] Scaricamento dati reali da Yahoo Finance...")
    data = download_market_data()

    spy_df = data[BENCHMARK_TICKER]
    trading_days = spy_df[spy_df.index >= pd.Timestamp(START_DATE)].index

    pfs = {
        "trend_following": init_portfolio(),
        "mean_reversion": init_portfolio(),
        "vol_breakout": init_portfolio(),
        "risk_parity": init_portfolio(),
        "rel_momentum": init_portfolio(),
        "buy_and_hold": init_portfolio(),
        "sp500": init_portfolio()
    }

    history_days = []
    prev_equities = {k: INITIAL_CAPITAL for k in STRATEGIES.keys()}
    slot_budget = INITIAL_CAPITAL / len(TICKERS)

    for day_idx, dt in enumerate(trading_days):
        date_str = dt.strftime("%Y-%m-%d")
        prices = {t: float(data[t].loc[dt, "Close"]) for t in TICKERS + [BENCHMARK_TICKER]}
        day_fees = {k: 0.0 for k in STRATEGIES.keys()}

        # 1. TREND FOLLOWING (Donchian 50d + SMA200 + Trailing Stop 2.5 ATR)
        pf1 = pfs["trend_following"]
        for t in TICKERS:
            row = data[t].loc[dt]
            price = prices[t]
            atr = float(row["ATR14"])
            if t in pf1["positions"]:
                pos = pf1["positions"][t]
                pos["high_since_entry"] = max(pos["high_since_entry"], price)
                pos["stop"] = max(pos["stop"], pos["high_since_entry"] - 2.5 * atr)
                if price < pos["stop"] or price < float(row["SMA50"]):
                    day_fees["trend_following"] += execute_sell(pf1, t, price)
            else:
                if price > float(row["DonchianHigh50"]) and price > float(row["SMA200"]):
                    budget = min(pf1["cash"], slot_budget)
                    day_fees["trend_following"] += execute_buy(
                        pf1, t, budget, price,
                        {"high_since_entry": price, "stop": price - 2.5 * atr}
                    )

        # 2. MEAN REVERSION (RSI-2 < 12 sopra SMA200, uscita > SMA5 o stop -7%)
        pf2 = pfs["mean_reversion"]
        for t in TICKERS:
            row = data[t].loc[dt]
            price = prices[t]
            if t in pf2["positions"]:
                pos = pf2["positions"][t]
                if price > float(row["SMA5"]) or price < pos["entry_price"] * 0.93:
                    day_fees["mean_reversion"] += execute_sell(pf2, t, price)
            else:
                if float(row["RSI2"]) < 12 and price > float(row["SMA200"]):
                    budget = min(pf2["cash"], slot_budget)
                    day_fees["mean_reversion"] += execute_buy(pf2, t, budget, price)

        # 3. VOLATILITY BREAKOUT (Bollinger Squeeze + Rottura Banda Superiore)
        pf3 = pfs["vol_breakout"]
        for t in TICKERS:
            row = data[t].loc[dt]
            price = prices[t]
            atr = float(row["ATR14"])
            if t in pf3["positions"]:
                pos = pf3["positions"][t]
                pos["high_since_entry"] = max(pos["high_since_entry"], price)
                pos["stop"] = max(pos["stop"], pos["high_since_entry"] - 2.0 * atr)
                if price < pos["stop"] or price < float(row["SMA20"]):
                    day_fees["vol_breakout"] += execute_sell(pf3, t, price)
            else:
                squeeze = float(row["BB_Width"]) <= float(row["BB_Width_Min60"]) * 1.25
                if squeeze and price > float(row["BB_Upper"]) and price > float(row["SMA200"]):
                    budget = min(pf3["cash"], slot_budget)
                    day_fees["vol_breakout"] += execute_buy(
                        pf3, t, budget, price,
                        {"high_since_entry": price, "stop": price - 2.0 * atr}
                    )

        # 4. RISK PARITY (Inverse Volatility 1/ATR, ribilanciamento ogni 20 sedute)
        pf4 = pfs["risk_parity"]
        if day_idx % 20 == 0:
            for t in list(pf4["positions"].keys()):
                day_fees["risk_parity"] += execute_sell(pf4, t, prices[t])

            inv_vols = {}
            for t in TICKERS:
                atr_pct = float(data[t].loc[dt, "ATR14"]) / prices[t]
                inv_vols[t] = 1.0 / max(atr_pct, 0.002)
            total_inv_vol = sum(inv_vols.values())
            avail_cash = pf4["cash"] * 0.995

            for t in TICKERS:
                budget = avail_cash * (inv_vols[t] / total_inv_vol)
                day_fees["risk_parity"] += execute_buy(pf4, t, budget, prices[t])

        # 5. MOMENTUM RELATIVO (Top 2 titoli per rendimento a 60 giorni, ribilanciamento ogni 20 sedute)
        pf5 = pfs["rel_momentum"]
        if day_idx % 20 == 0:
            for t in list(pf5["positions"].keys()):
                day_fees["rel_momentum"] += execute_sell(pf5, t, prices[t])

            ranked = sorted(
                [(t, float(data[t].loc[dt, "ROC60"])) for t in TICKERS],
                key=lambda x: x[1],
                reverse=True
            )
            winners = [t for t, roc in ranked[:2] if roc > 0]
            if winners:
                avail_cash = pf5["cash"] * 0.995
                per_winner = avail_cash / len(winners)
                for t in winners:
                    day_fees["rel_momentum"] += execute_buy(pf5, t, per_winner, prices[t])

        # 7. BUY & HOLD (5 Asset dal Giorno 1)
        pf7 = pfs["buy_and_hold"]
        if day_idx == 0:
            for t in TICKERS:
                budget = (INITIAL_CAPITAL * 0.995) / len(TICKERS)
                day_fees["buy_and_hold"] += execute_buy(pf7, t, budget, prices[t])

        # 8. BENCHMARK S&P 500 (SPY dal Giorno 1)
        pf_spy = pfs["sp500"]
        if day_idx == 0:
            day_fees["sp500"] += execute_buy(pf_spy, BENCHMARK_TICKER, INITIAL_CAPITAL * 0.998, prices[BENCHMARK_TICKER])

        # Calcolo Equity Nette
        eq1 = portfolio_net_value(pf1, prices)
        eq2 = portfolio_net_value(pf2, prices)
        eq3 = portfolio_net_value(pf3, prices)
        eq4 = portfolio_net_value(pf4, prices)
        eq5 = portfolio_net_value(pf5, prices)
        # 6. ENSEMBLE 50/50 (50% Trend Following + 50% Mean Reversion)
        eq6 = 0.5 * eq1 + 0.5 * eq2
        day_fees["ensemble_50_50"] = round(0.5 * day_fees["trend_following"] + 0.5 * day_fees["mean_reversion"], 2)
        eq7 = portfolio_net_value(pf7, prices)
        eq_spy = portfolio_net_value(pf_spy, prices)

        equities = {
            "trend_following": round(eq1, 2),
            "mean_reversion": round(eq2, 2),
            "vol_breakout": round(eq3, 2),
            "risk_parity": round(eq4, 2),
            "rel_momentum": round(eq5, 2),
            "ensemble_50_50": round(eq6, 2),
            "buy_and_hold": round(eq7, 2),
            "sp500": round(eq_spy, 2)
        }

        daily_returns = {}
        for k, eq in equities.items():
            ret = ((eq - prev_equities[k]) / prev_equities[k]) * 100
            daily_returns[k] = round(ret, 2)
            prev_equities[k] = eq

        # Dettaglio posizioni per Ensemble 50/50
        ens_positions = []
        for p in build_position_details(pf1, prices):
            ens_positions.append({
                **p,
                "ticker": f"{p['ticker']} [Trend 50%]",
                "total_value": round(p["total_value"] * 0.5, 2),
                "pnl_usd": round(p["pnl_usd"] * 0.5, 2)
            })
        for p in build_position_details(pf2, prices):
            ens_positions.append({
                **p,
                "ticker": f"{p['ticker']} [Rev 50%]",
                "total_value": round(p["total_value"] * 0.5, 2),
                "pnl_usd": round(p["pnl_usd"] * 0.5, 2)
            })

        history_days.append({
            "date": date_str,
            "equities": equities,
            "daily_returns": daily_returns,
            "daily_fees": {k: round(v, 2) for k, v in day_fees.items()},
            "positions": {
                "trend_following": build_position_details(pf1, prices),
                "mean_reversion": build_position_details(pf2, prices),
                "vol_breakout": build_position_details(pf3, prices),
                "risk_parity": build_position_details(pf4, prices),
                "rel_momentum": build_position_details(pf5, prices),
                "ensemble_50_50": ens_positions,
                "buy_and_hold": build_position_details(pf7, prices),
                "sp500": build_position_details(pf_spy, prices)
            }
        })

    # Riepilogo Finale & Impatto Commissioni
    summary = {}
    last_eq = history_days[-1]["equities"]
    ens_fees = round(0.5 * pf1["total_fees"] + 0.5 * pf2["total_fees"], 2)
    ens_trades = int(round(0.5 * pf1["trades_count"] + 0.5 * pf2["trades_count"]))

    fee_map = {
        "trend_following": (pf1["total_fees"], pf1["trades_count"]),
        "mean_reversion": (pf2["total_fees"], pf2["trades_count"]),
        "vol_breakout": (pf3["total_fees"], pf3["trades_count"]),
        "risk_parity": (pf4["total_fees"], pf4["trades_count"]),
        "rel_momentum": (pf5["total_fees"], pf5["trades_count"]),
        "ensemble_50_50": (ens_fees, ens_trades),
        "buy_and_hold": (pf7["total_fees"], pf7["trades_count"]),
        "sp500": (pf_spy["total_fees"], pf_spy["trades_count"])
    }

    for k, label in STRATEGIES.items():
        net_eq = last_eq[k]
        fees_paid, n_trades = fee_map[k]
        gross_eq = round(net_eq + fees_paid, 2)
        net_ret_pct = ((net_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        gross_ret_pct = ((gross_eq - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        fee_impact_pct = (fees_paid / INITIAL_CAPITAL) * 100

        summary[k] = {
            "name": label,
            "gross_equity": gross_eq,
            "final_equity": net_eq,
            "gross_return_pct": round(gross_ret_pct, 2),
            "total_return_pct": round(net_ret_pct, 2),
            "total_fees_usd": round(fees_paid, 2),
            "fee_impact_pct": round(fee_impact_pct, 3),
            "trades_count": n_trades
        }

    output = {
        "last_updated": history_days[-1]["date"],
        "initial_capital": INITIAL_CAPITAL,
        "strategies_meta": STRATEGIES,
        "summary": summary,
        "history": history_days
    }

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"[✓] Arena salvata fino al {history_days[-1]['date']} ({len(history_days)} sedute reali di borsa).")


if __name__ == "__main__":
    run_arena()
