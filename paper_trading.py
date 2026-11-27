import json
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

# --- CONFIGURAZIONE GENERALE ---
TICKERS = ["AAPL", "NVDA", "XLE", "GLD", "XLV"]
BENCHMARK_TICKER = "SPY"
INITIAL_CAPITAL = 100000.0
START_DATE = "2018-01-01"
HISTORY_FILE = "arena_history.json"

# --- MODELLO COMMISSIONI & SLIPPAGE ---
COMM_PER_SHARE = 0.005
MIN_COMMISSION = 1.00
SLIPPAGE_PCT = 0.0003

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
    if shares <= 0 or pd.isna(price):
        return 0.0
    comm = max(MIN_COMMISSION, shares * COMM_PER_SHARE)
    slippage = (shares * price) * SLIPPAGE_PCT
    return round(comm + slippage, 2)


def compute_indicators(df):
    """Calcola indicatori e SEGNALI GREZZI (leciti alla chiusura del giorno).
    L'esecuzione dei segnali avviene SEMPRE il giorno dopo, all'Open
    (vedi run_arena): qui ci limitiamo a definire cosa e' 'vero' a fine giornata."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["SMA5"] = close.rolling(5).mean()
    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()

    df["DonchianHigh50"] = high.rolling(50).max().shift(1)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(2).mean()
    loss = (-delta.clip(upper=0)).rolling(2).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI2"] = (100 - (100 / (1 + rs))).fillna(50)

    std20 = close.rolling(20).std()
    df["BB_Upper"] = df["SMA20"] + 2 * std20
    df["BB_Lower"] = df["SMA20"] - 2 * std20
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["SMA20"]
    df["BB_Width_Min60"] = df["BB_Width"].rolling(60).min().shift(1)

    df["ROC60"] = close.pct_change(60)

    # Segnali grezzi, noti alla chiusura del giorno stesso (NON eseguiti qui)
    df["Sig_Trend_Buy"] = (close > df["DonchianHigh50"]) & (close > df["SMA200"])
    df["Sig_Trend_ExitSMA"] = close < df["SMA50"]

    df["Sig_MeanRev_Buy"] = (df["RSI2"] < 12) & (close > df["SMA200"])
    df["Sig_MeanRev_ExitSMA"] = close > df["SMA5"]

    squeeze = df["BB_Width"] <= df["BB_Width_Min60"] * 1.25
    df["Sig_VolBreak_Buy"] = squeeze & (close > df["BB_Upper"]) & (close > df["SMA200"])
    df["Sig_VolBreak_ExitSMA"] = close < df["SMA20"]

    return df


def fetch_single_ticker(ticker, start_fetch, retries=3):
    for attempt in range(1, retries + 1):
        try:
            df = yf.Ticker(ticker).history(start=start_fetch, auto_adjust=True)
            if df is not None and not df.empty and "Close" in df.columns:
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                df = df[~df.index.duplicated(keep="last")].dropna(subset=["Close"])
                if not df.empty:
                    return df
        except Exception as exc:
            print(f"[!] Tentativo {attempt}/{retries} fallito per {ticker}: {exc}")
        time.sleep(1.5 * attempt)
    raise RuntimeError(f"Impossibile scaricare i dati per {ticker} dopo {retries} tentativi.")


def download_market_data():
    all_tickers = TICKERS + [BENCHMARK_TICKER]
    start_fetch = (datetime.strptime(START_DATE, "%Y-%m-%d") - timedelta(days=420)).strftime("%Y-%m-%d")

    raw_map = {}
    for t in all_tickers:
        print(f"    -> Scaricamento {t}...")
        raw_map[t] = fetch_single_ticker(t, start_fetch)
        time.sleep(0.4)

    base_index = raw_map[BENCHMARK_TICKER].index
    data = {}
    for t in all_tickers:
        df = raw_map[t].reindex(base_index).ffill().bfill()
        data[t] = compute_indicators(df)
    return data


def init_portfolio(capital=INITIAL_CAPITAL):
    return {"cash": capital, "positions": {}, "total_fees": 0.0, "trades_count": 0}


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
    if pd.isna(price) or pd.isna(budget) or price <= 0 or budget <= 0:
        return 0.0
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

    # --- GIORNO 0: nessun "ieri" disponibile, nessun segnale possibile.
    # Apriamo solo le posizioni statiche (Buy&Hold, benchmark SPY) all'Open. ---
    dt0 = trading_days[0]
    open0 = {t: float(data[t].loc[dt0, "Open"]) for t in TICKERS + [BENCHMARK_TICKER]}
    close0 = {t: float(data[t].loc[dt0, "Close"]) for t in TICKERS + [BENCHMARK_TICKER]}

    pf7 = pfs["buy_and_hold"]
    for t in TICKERS:
        budget = (INITIAL_CAPITAL * 0.995) / len(TICKERS)
        execute_buy(pf7, t, budget, open0[t])

    pf_spy = pfs["sp500"]
    execute_buy(pf_spy, BENCHMARK_TICKER, INITIAL_CAPITAL * 0.998, open0[BENCHMARK_TICKER])

    equities0 = {
        "trend_following": INITIAL_CAPITAL,
        "mean_reversion": INITIAL_CAPITAL,
        "vol_breakout": INITIAL_CAPITAL,
        "risk_parity": INITIAL_CAPITAL,
        "rel_momentum": INITIAL_CAPITAL,
        "ensemble_50_50": INITIAL_CAPITAL,
        "buy_and_hold": round(portfolio_net_value(pf7, close0), 2),
        "sp500": round(portfolio_net_value(pf_spy, close0), 2)
    }
    daily_returns0 = {k: round(((equities0[k] - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100, 2) for k in STRATEGIES}
    for k in STRATEGIES:
        prev_equities[k] = equities0[k]

    history_days.append({
        "date": dt0.strftime("%Y-%m-%d"),
        "equities": equities0,
        "daily_returns": daily_returns0,
        "daily_fees": {
            "trend_following": 0.0, "mean_reversion": 0.0, "vol_breakout": 0.0,
            "risk_parity": 0.0, "rel_momentum": 0.0, "ensemble_50_50": 0.0,
            "buy_and_hold": round(pf7["total_fees"], 2),
            "sp500": round(pf_spy["total_fees"], 2)
        },
        "positions": {
            "trend_following": [], "mean_reversion": [], "vol_breakout": [],
            "risk_parity": [], "rel_momentum": [], "ensemble_50_50": [],
            "buy_and_hold": build_position_details(pf7, close0),
            "sp500": build_position_details(pf_spy, close0)
        }
    })

    # --- DAL GIORNO 1 IN POI: segnale letto da IERI, eseguito OGGI all'Open ---
    for day_idx in range(1, len(trading_days)):
        dt = trading_days[day_idx]
        dt_prev = trading_days[day_idx - 1]

        open_p = {t: float(data[t].loc[dt, "Open"]) for t in TICKERS + [BENCHMARK_TICKER]}
        close_p = {t: float(data[t].loc[dt, "Close"]) for t in TICKERS + [BENCHMARK_TICKER]}
        day_fees = {k: 0.0 for k in STRATEGIES.keys()}

        # 1. TREND FOLLOWING
        pf1 = pfs["trend_following"]
        for t in TICKERS:
            row_prev = data[t].loc[dt_prev]
            prev_close = float(row_prev["Close"])
            atr_prev = float(row_prev["ATR14"])
            open_price = open_p[t]

            if t in pf1["positions"]:
                pos = pf1["positions"][t]
                pos["high_since_entry"] = max(pos["high_since_entry"], prev_close)
                pos["stop"] = max(pos["stop"], pos["high_since_entry"] - 2.5 * atr_prev)
                if bool(row_prev["Sig_Trend_ExitSMA"]) or prev_close < pos["stop"]:
                    day_fees["trend_following"] += execute_sell(pf1, t, open_price)
            else:
                if bool(row_prev["Sig_Trend_Buy"]):
                    budget = min(pf1["cash"], slot_budget)
                    day_fees["trend_following"] += execute_buy(
                        pf1, t, budget, open_price,
                        {"high_since_entry": prev_close, "stop": prev_close - 2.5 * atr_prev}
                    )

        # 2. MEAN REVERSION
        pf2 = pfs["mean_reversion"]
        for t in TICKERS:
            row_prev = data[t].loc[dt_prev]
            prev_close = float(row_prev["Close"])
            open_price = open_p[t]

            if t in pf2["positions"]:
                pos = pf2["positions"][t]
                if bool(row_prev["Sig_MeanRev_ExitSMA"]) or prev_close < pos["entry_price"] * 0.93:
                    day_fees["mean_reversion"] += execute_sell(pf2, t, open_price)
            else:
                if bool(row_prev["Sig_MeanRev_Buy"]):
                    budget = min(pf2["cash"], slot_budget)
                    day_fees["mean_reversion"] += execute_buy(pf2, t, budget, open_price)

        # 3. VOLATILITY BREAKOUT
        pf3 = pfs["vol_breakout"]
        for t in TICKERS:
            row_prev = data[t].loc[dt_prev]
            prev_close = float(row_prev["Close"])
            atr_prev = float(row_prev["ATR14"])
            open_price = open_p[t]

            if t in pf3["positions"]:
                pos = pf3["positions"][t]
                pos["high_since_entry"] = max(pos["high_since_entry"], prev_close)
                pos["stop"] = max(pos["stop"], pos["high_since_entry"] - 2.0 * atr_prev)
                if bool(row_prev["Sig_VolBreak_ExitSMA"]) or prev_close < pos["stop"]:
                    day_fees["vol_breakout"] += execute_sell(pf3, t, open_price)
            else:
                if bool(row_prev["Sig_VolBreak_Buy"]):
                    budget = min(pf3["cash"], slot_budget)
                    day_fees["vol_breakout"] += execute_buy(
                        pf3, t, budget, open_price,
                        {"high_since_entry": prev_close, "stop": prev_close - 2.0 * atr_prev}
                    )

        # 4. RISK PARITY (ribilancio periodico su ATR di ieri, eseguito oggi all'Open)
        pf4 = pfs["risk_parity"]
        if (day_idx - 1) % 20 == 0:
            for t in list(pf4["positions"].keys()):
                day_fees["risk_parity"] += execute_sell(pf4, t, open_p[t])

            inv_vols = {}
            for t in TICKERS:
                row_prev = data[t].loc[dt_prev]
                atr_pct = float(row_prev["ATR14"]) / float(row_prev["Close"])
                inv_vols[t] = 1.0 / max(atr_pct, 0.002)
            total_inv_vol = sum(inv_vols.values())
            avail_cash = pf4["cash"] * 0.995

            for t in TICKERS:
                budget = avail_cash * (inv_vols[t] / total_inv_vol)
                day_fees["risk_parity"] += execute_buy(pf4, t, budget, open_p[t])

        # 5. MOMENTUM RELATIVO (ranking su ROC60 di ieri, eseguito oggi all'Open)
        pf5 = pfs["rel_momentum"]
        if (day_idx - 1) % 20 == 0:
            for t in list(pf5["positions"].keys()):
                day_fees["rel_momentum"] += execute_sell(pf5, t, open_p[t])

            ranked = sorted(
                [(t, float(data[t].loc[dt_prev, "ROC60"])) for t in TICKERS],
                key=lambda x: x[1],
                reverse=True
            )
            winners = [t for t, roc in ranked[:2] if roc > 0]
            if winners:
                avail_cash = pf5["cash"] * 0.995
                per_winner = avail_cash / len(winners)
                for t in winners:
                    day_fees["rel_momentum"] += execute_buy(pf5, t, per_winner, open_p[t])

        # Equity marcate alla Close di oggi
        eq1 = portfolio_net_value(pf1, close_p)
        eq2 = portfolio_net_value(pf2, close_p)
        eq3 = portfolio_net_value(pf3, close_p)
        eq4 = portfolio_net_value(pf4, close_p)
        eq5 = portfolio_net_value(pf5, close_p)
        eq6 = 0.5 * eq1 + 0.5 * eq2
        day_fees["ensemble_50_50"] = round(0.5 * day_fees["trend_following"] + 0.5 * day_fees["mean_reversion"], 2)
        eq7 = portfolio_net_value(pf7, close_p)
        eq_spy = portfolio_net_value(pf_spy, close_p)

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

        ens_positions = []
        for p in build_position_details(pf1, close_p):
            ens_positions.append({
                **p, "ticker": f"{p['ticker']} [Trend 50%]",
                "total_value": round(p["total_value"] * 0.5, 2),
                "pnl_usd": round(p["pnl_usd"] * 0.5, 2)
            })
        for p in build_position_details(pf2, close_p):
            ens_positions.append({
                **p, "ticker": f"{p['ticker']} [Rev 50%]",
                "total_value": round(p["total_value"] * 0.5, 2),
                "pnl_usd": round(p["pnl_usd"] * 0.5, 2)
            })

        history_days.append({
            "date": dt.strftime("%Y-%m-%d"),
            "equities": equities,
            "daily_returns": daily_returns,
            "daily_fees": {k: round(v, 2) for k, v in day_fees.items()},
            "positions": {
                "trend_following": build_position_details(pf1, close_p),
                "mean_reversion": build_position_details(pf2, close_p),
                "vol_breakout": build_position_details(pf3, close_p),
                "risk_parity": build_position_details(pf4, close_p),
                "rel_momentum": build_position_details(pf5, close_p),
                "ensemble_50_50": ens_positions,
                "buy_and_hold": build_position_details(pf7, close_p),
                "sp500": build_position_details(pf_spy, close_p)
            }
        })

    # --- Riepilogo finale & impatto commissioni ---
    summary = {}
    last_eq = history_days[-1]["equities"]
    ens_fees = round(0.5 * pfs["trend_following"]["total_fees"] + 0.5 * pfs["mean_reversion"]["total_fees"], 2)
    ens_trades = int(round(0.5 * pfs["trend_following"]["trades_count"] + 0.5 * pfs["mean_reversion"]["trades_count"]))

    fee_map = {
        "trend_following": (pfs["trend_following"]["total_fees"], pfs["trend_following"]["trades_count"]),
        "mean_reversion": (pfs["mean_reversion"]["total_fees"], pfs["mean_reversion"]["trades_count"]),
        "vol_breakout": (pfs["vol_breakout"]["total_fees"], pfs["vol_breakout"]["trades_count"]),
        "risk_parity": (pfs["risk_parity"]["total_fees"], pfs["risk_parity"]["trades_count"]),
        "rel_momentum": (pfs["rel_momentum"]["total_fees"], pfs["rel_momentum"]["trades_count"]),
        "ensemble_50_50": (ens_fees, ens_trades),
        "buy_and_hold": (pfs["buy_and_hold"]["total_fees"], pfs["buy_and_hold"]["trades_count"]),
        "sp500": (pfs["sp500"]["total_fees"], pfs["sp500"]["trades_count"])
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

    print(f"[OK] Arena salvata fino al {history_days[-1]['date']} ({len(history_days)} sedute reali di borsa).")


if __name__ == "__main__":
    run_arena()
