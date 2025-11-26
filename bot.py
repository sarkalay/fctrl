# pure_scalper_v1.py
# မင်း terminal ထဲ paste ပြီး run ပစ်လိုက်ရုံပဲ
# python3 pure_scalper_v1.py

import ccxt.async_support as ccxt
import asyncio
import time
import json
from datetime import datetime

# ================== CONFIG ==================
SYMBOLS = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'AVAX/USDT:USDT', '1000PEPE/USDT:USDT']
LEVERAGE = 20
POSITION_USD = 35          # တစ်ခါ $35
TP_PCT = 0.0048            # +0.48% gross → net ~0.40% after fee
SL_PCT = 0.0028            # -0.28%
MAX_POSITIONS = 10
COOLDOWN_SEC = 3           # တူညီတဲ့ coin ကို ပြန်မဝင်ခင်

# Paper wallet
balance = 500.0
initial_balance = balance
positions = {}
cooldown = {}

exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

async def fetch_ohlcv(symbol, timeframe='15s', limit=100):
    try:
        return await exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except:
        return None

def ema(data, period):
    close = [x[4] for x in data]
    k = 2/(period + 1)
    ema_val = close[0]
    for price in close[1:]:
        ema_val = price * k + ema_val * (1 - k)
    return ema_val

async def scalper():
    global balance
    while True:
        try:
            for symbol in SYMBOLS:
                if symbol in cooldown and time.time() < cooldown[symbol]:
                    continue
                if len(positions) >= MAX_POSITIONS:
                    continue

                ohlcv_15s = await fetch_ohlcv(symbol, '15s', 50)
                ohlcv_1m  = await fetch_ohlcv(symbol, '1m', 50)
                if not ohlcv_15s or not ohlcv_1m:
                    continue

                price = ohlcv_15s[-1][4]
                ema8_15s  = ema(ohlcv_15s, 8)
                ema21_15s = ema(ohlcv_15s, 21)
                ema8_1m   = ema(ohlcv_1m, 8)
                rsi = 100 - (100 / (1 + 
                    sum(max(ohlcv_15s[i][4]-ohlcv_15s[i-1][4],0) for i in range(-14,0)) /
                    sum(abs(ohlcv_15s[i][4]-ohlcv_15s[i-1][4]) for i in range(-14,0) or [1])
                ))

                # LONG SIGNAL
                if (ema8_15s > ema21_15s and 
                    ema8_1m > ema8_1m * 1.0001 and   # 1m လည်း တက်နေတာ confirm
                    rsi < 38 and price < ema8_15s * 1.002):
                    
                    qty = (POSITION_USD * LEVERAGE) / price
                    positions[symbol] = {
                        'side': 'LONG',
                        'entry': price,
                        'qty': qty,
                        'tp': price * (1 + TP_PCT),
                        'sl': price * (1 - SL_PCT),
                        'time': time.time()
                    }
                    cooldown[symbol] = time.time() + COOLDOWN_SEC
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚀 LONG  {symbol} @ {price:.4f}")

                # SHORT SIGNAL
                elif (ema8_15s < ema21_15s and 
                      ema8_1m < ema8_1m * 0.9999 and
                      rsi > 62 and price > ema8_15s * 0.998):
                    
                    qty = (POSITION_USD * LEVERAGE) / price
                    positions[symbol] = {
                        'side': 'SHORT',
                        'entry': price,
                        'qty': qty,
                        'tp': price * (1 - TP_PCT),
                        'sl': price * (1 + SL_PCT),
                        'time': time.time()
                    }
                    cooldown[symbol] = time.time() + COOLDOWN_SEC
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔥 SHORT {symbol} @ {price:.4f}")

            # Check exits
            to_remove = []
            for symbol, pos in positions.items():
                price = (await fetch_ohlcv(symbol, '15s', 1))[0][4]
                pnl_pct = (price - pos['entry']) / pos['entry'] * (1 if pos['side']=='LONG' else -1)

                if (pos['side']=='LONG' and (price >= pos['tp'] or price <= pos['sl'])) or \
                   (pos['side']=='SHORT' and (price <= pos['tp'] or price >= pos['sl'])):
                    profit = POSITION_USD * LEVERAGE * pnl_pct * 0.998   # after fee
                    balance += profit
                    winlose = "✅ WIN" if pnl_pct > 0 else "❌ LOSS"
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {winlose} {symbol} | PnL: {profit:+.2f} | Balance: {balance:.1f}")
                    to_remove.append(symbol)

            for s in to_remove:
                del positions[s]

            # Dashboard
            if int(time.time()) % 15 == 0:
                daily_pnl = ((balance / initial_balance) - 1) * 100
                print(f"💰 BALANCE: ${balance:.1f} | Today: {daily_pnl:+.2f}% | Positions: {len(positions)}")

            await asyncio.sleep(0.8)  # ~75 requests/min → safe

        except Exception as e:
            print("Error:", e)
            await asyncio.sleep(5)

print("🚀 Pure Scalper v1.0 STARTED - Paper Trading Mode")
print("Starting balance: $500")
asyncio.run(scalper())
