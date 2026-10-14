import os
import json
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf

PORTFOLIO_FILE = "paper_portfolio.json"

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
            print("[+] Nessun file di stato trovato. Creato nuovo portafoglio virtuale.")

    def save_state(self):
        payload = {
            "cash": self.cash,
            "positions": {t: asdict(pos) for t, pos in self.positions.items()},
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(payload, f, indent=4)
        print(f"[+] Stato portafoglio salvato su {PORTFOLIO_FILE}")

    def analyze_ticker(self, ticker: str):
        # Scarica storico sufficiente per SMA200 e slope
        raw = yf.download(ticker, period="2y", interval="1d", progress=False)
        if raw.empty:
            return None
        
        df = raw[["Open", "High", "Low", "Close"]].copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()

        # ATR
        hl = df["High"] - df["Low"]
        hc = (df["High"] - df["Close"].shift(1)).abs()
        lc = (df["Low"] - df["Close"].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(14).mean().bfill()

        # Trend & Donchian
        df["SMA200"] = df["Close"].rolling(200).mean()
        df["SMA200_slope"] = df["SMA200"] - df["SMA200"].shift(20)
        df["Max50"] = df["Close"].shift(1).rolling(50).max()
        df["SMA50"] = df["Close"].rolling(50).mean()

        last = df.iloc[-1]
        prev = df.iloc[-2]

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

    def generate_signals(self, watchlist: list[str]):
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
            target_shares = 0

            if pos is not None:
                # Gestione posizione aperta
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
                # Valutazione nuovo ingresso
                if info["breakout"] and info["bull_regime"]:
                    if len(self.positions) < self.max_positions and self.cash >= price:
                        shares_to_buy = int(slot_size // price)
                        total_cost = shares_to_buy * price
                        if total_cost <= self.cash and shares_to_buy > 0:
                            action = "COMPRA"
                            reason = f"Breakout Donchian 50d + Trend SMA200 ({shares_to_buy} quote)"
                            orders.append((t, "BUY", shares_to_buy, price))
                        else:
                            action = "ATTENDI"
                            reason = "Fondi insufficienti per allocazione"
                    else:
                        action = "ATTENDI"
                        reason = "Slot di portafoglio esauriti"
                else:
                    if not info["bull_regime"]:
                        reason = "Sotto SMA200 o pendenza ribassista"
                    else:
                        reason = "Sotto il massimo a 50 giorni"

            pos_desc = f"{pos.shares} quote" if pos else "Liquido"
            print(f"{t:<7} ${price:<9.2f} {pos_desc:<12} {action:<10} {reason}")

        print("="*80)
        return orders

    def execute_orders(self, orders: list):
        if not orders:
            print("\nNessun ordine operativo da eseguire.")
            return

        print("\n--- ESECUZIONE SIMULATA ORDINI ---")
        for ticker, side, shares, price in orders:
            if side == "SELL":
                revenue = shares * price
                self.cash += revenue
                del self.positions[ticker]
                print(f"[VENDITA] Eseguita vendita di {shares} quote di {ticker} a ${price:.2f} (+${revenue:,.2f})")
            elif side == "BUY":
                cost = shares * price
                self.cash -= cost
                self.positions[ticker] = Position(
                    shares=shares,
                    entry_price=price,
                    peak_price=price,
                    entry_date=datetime.now().strftime("%Y-%m-%d")
                )
                print(f"[ACQUISTO] Eseguito acquisto di {shares} quote di {ticker} a ${price:.2f} (-${cost:,.2f})")
        
        self.save_state()


if __name__ == "__main__":
    watchlist = ["AAPL", "NVDA", "XLE", "GLD", "XLV"]
    bot = PaperTrader(initial_cash=100_000.0, max_positions=5, atr_mult=2.5)
    
    orders = bot.generate_signals(watchlist)
    if orders:
        bot.execute_orders(orders)
    else:
        # Se non ci sono nuovi ordini, salva comunque l'aggiornamento dei peak_prices
        bot.save_state()