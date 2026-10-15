import os
import json
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf

PORTFOLIO_FILE = "paper_portfolio.json"
HISTORY_FILE = "daily_history.json"

@dataclass
class Position:
    shares: int
    entry_price: float
    peak_price: float
    entry_date: str

class PaperTrader:
    def __init__(self, initial_cash: float = 100_000.0, max_positions: int = 5, atr_mult: float = 2.5):
        self.initial_cash = initial_cash
        self.max_positions = max_positions
        self.atr_mult = atr_mult
        self.cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.load_state()

    def load_state(self):
        if os.path.exists(PORTFOLIO_FILE):
            try:
                with open(PORTFOLIO_FILE, "r") as f:
                    data = json.load(f)
                    self.cash = data.get("cash", self.initial_cash)
                    self.positions = {
                        ticker: Position(**pos_data) 
                        for ticker, pos_data in data.get("positions", {}).items()
                    }
                print(f"[+] Portafoglio caricato da {PORTFOLIO_FILE}")
            except Exception as e:
                print(f"[!] Errore lettura file stato: {e}. Inizializzazione nuovo conto.")
        else:
            print("[+] Nuovo portafoglio virtuale.")

    def save_state(self, current_equity: float):
        today_str = datetime.now().strftime("%Y-%m-%d")
        
        # 1. Salva lo stato delle posizioni
        payload = {
            "cash": self.cash,
            "positions": {t: asdict(pos) for t, pos in self.positions.items()},
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(payload, f, indent=4)
            
        # 2. Aggiorna lo storico giornaliero per il calendario
        history = {}
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r") as f:
                    history = json.load(f)
            except Exception:
                history = {}

        # Prendi l'ultimo valore registrato per calcolare la percentuale di oggi
        prev_equity = self.initial_cash
        if history:
            dates = sorted(history.keys())
            # se oggi e gia presente, prendi quello precedente
            valid_dates = [d for d in dates if d != today_str]
            if valid_dates:
                prev_equity = history[valid_dates[-1]]["equity"]

        daily_pct = ((current_equity - prev_equity) / prev_equity) * 100 if prev_equity > 0 else 0.0

        history[today_str] = {
            "equity": round(current_equity, 2),
            "return_pct": round(daily_pct, 2)
        }

        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)
        print(f"[+] Storico salvato su {HISTORY_FILE} (Oggi: {daily_pct:+.2f}%)")

    def analyze_ticker(self, ticker: str):
        raw = yf.download(ticker, period="2y", interval="1d", progress=False)
        if raw.empty:
            return None
        
        df = raw[["Open", "High", "Low", "Close"]].copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()

        hl = df["High"] - df["Low"]
        hc = (df["High"] - df["Close"].shift(1)).abs()
        lc = (df["Low"] - df["Close"].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(14).mean().bfill()

        df["SMA200"] = df["Close"].rolling(200).mean()
        df["SMA200_slope"] = df["SMA200"] - df["SMA200"].shift(20)
        df["Max50"] = df["Close"].shift(1).rolling(50).max()
        df["SMA50"] = df["Close"].rolling(50).mean()

        last = df.iloc[-1]
        return {
            "current_price": float(last["Close"]),
            "atr": float(last["ATR"]),
            "sma50": float(last["SMA50"]),
            "sma200": float(last["SMA200"]),
            "sma200_slope": float(last["SMA200_slope"]),
            "max50": float(last["Max50"]),
            "breakout": bool(last["Close"] > last["Max50"]),
            "bull_regime": bool(last["Close"] > last["SMA200"] and last["SMA200_slope"] > 0),
            "below_sma50": bool(last["Close"] < last["SMA50"])
        }

    def generate_and_run(self, watchlist: list[str]):
        portfolio_val = self.cash
        analysis = {}

        for t in watchlist:
            info = self.analyze_ticker(t)
            if info:
                analysis[t] = info
                if t in self.positions:
                    portfolio_val += self.positions[t].shares * info["current_price"]

        slot_size = portfolio_val / self.max_positions
        orders = []

        print("\n" + "="*80)
        print(f" REPORT OPERATIVO PAPER TRADING - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f" Liquidità Libera: ${self.cash:,.2f} | Valore Totale Portafoglio: ${portfolio_val:,.2f}")
        print("="*80)
        print(f"{'TICKER':<7} {'PREZZO':<10} {'POSIZIONE':<12} {'AZIONE':<10} {'NOTE/MOTIVAZIONE'}")
        print("-" * 80)

        for t in watchlist:
            if t not in analysis:
                continue

            info = analysis[t]
            price = info["current_price"]
            atr = info["atr"]
            pos = self.positions.get(t)

            action = "INATTIVO"
            reason = "Nessun segnale di trend"

            if pos is not None:
                new_peak = max(pos.peak_price, price)
                pos.peak_price = new_peak
                stop_loss = new_peak - (atr * self.atr_mult)

                if price < stop_loss:
                    action = "VENDI"
                    reason = f"Trailing Stop Scattato (${stop_loss:.2f})"
                    orders.append((t, "SELL", pos.shares, price))
                elif info["below_sma50"]:
                    action = "VENDI"
                    reason = f"Chiusura sotto SMA50 (${info['sma50']:.2f})"
                    orders.append((t, "SELL", pos.shares, price))
                else:
                    action = "MANTIENI"
                    pnl_pct = ((price - pos.entry_price) / pos.entry_price) * 100
                    reason = f"PnL: {pnl_pct:+.2f}% | Stop: ${stop_loss:.2f}"
            else:
                if info["breakout"] and info["bull_regime"]:
                    if len(self.positions) < self.max_positions and self.cash >= price:
                        shares_to_buy = int(slot_size // price)
                        total_cost = shares_to_buy * price
                        if total_cost <= self.cash and shares_to_buy > 0:
                            action = "COMPRA"
                            reason = f"Breakout Donchian 50d ({shares_to_buy} quote)"
                            orders.append((t, "BUY", shares_to_buy, price))
                        else:
                            action = "ATTENDI"
                            reason = "Fondi insufficienti"
                    else:
                        action = "ATTENDI"
                        reason = "Slot di portafoglio esauriti"
                else:
                    reason = "Sotto i massimi o trend negativo"

            pos_desc = f"{pos.shares} quote" if pos else "Liquido"
            print(f"{t:<7} ${price:<9.2f} {pos_desc:<12} {action:<10} {reason}")

        print("="*80)

        # Esecuzione ordini se presenti
        for ticker, side, shares, price in orders:
            if side == "SELL":
                self.cash += shares * price
                del self.positions[ticker]
                print(f"[VENDITA] Eseguita vendita di {shares} quote di {ticker}")
            elif side == "BUY":
                self.cash -= shares * price
                self.positions[ticker] = Position(shares=shares, entry_price=price, peak_price=price, entry_date=datetime.now().strftime("%Y-%m-%d"))
                print(f"[ACQUISTO] Eseguito acquisto di {shares} quote di {ticker}")

        # Ricalcola l'equity finale dopo gli ordini
        final_equity = self.cash
        for t, p in self.positions.items():
            if t in analysis:
                final_equity += p.shares * analysis[t]["current_price"]

        self.save_state(final_equity)


if __name__ == "__main__":
    watchlist = ["AAPL", "NVDA", "XLE", "GLD", "XLV"]
    bot = PaperTrader(initial_cash=100_000.0, max_positions=5, atr_mult=2.5)
    bot.generate_and_run(watchlist)
