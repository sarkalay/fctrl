# bot_two_way.py  ← LONG + SHORT နှစ်ဖက်စလုံး ပါပြီ!
import ccxt.async_support as ccxt
import asyncio
import time
from datetime import datetime

exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future', 'adjustForTimeDifference': True},
    'timeout': 10000,
})

SYMBOLS = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'AVAX/USDT:USDT']
balance = 500.0
initial_balance = balance
positions = {}
last_trade = {}

async def get_price(sym):
    ticker = await exchange.fetch_ticker(sym)
    return ticker['last']

async def bot():
    global balance
    print("TWO-WAY SCALPER STARTED - LONG + SHORT နှစ်ဖက်စလုံး ပါပြီ!")

    while True:
        try:
            for sym in SYMBOLS:
                if sym in positions or len(positions) >= 8:
                    continue
                if sym in last_trade and time.time() - last_trade[sym] < 9:
                    continue

                price = await get_price(sym)

                # LONG conditions (ဈေးချိုးပြီး ပြန်တက်မယ့် နေရာ)
                if ("AVAX" in sym and price < 12.08) or \
                   ("SOL" in sym and price < 162) or \
                   ("ETH" in sym and price < 2580):
                    positions[sym] = {'side': 'LONG', 'entry': price, 'time': time.time()}
                    last_trade[sym] = time.time()
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] LONG  {sym.split('/')[0]} @ {price:.4f}")

                # SHORT conditions (ဈေးကျိုးပြီး ပြန်ကျမယ့် နေရာ)
                elif ("BTC" in sym and price > 95200) or \
                     ("AVAX" in sym and price > 12.65) or \
                     ("SOL" in sym and price > 178) or \
                     ("ETH" in sym and price > 2720):
                    positions[sym] = {'side': 'SHORT', 'entry': price, 'time': time.time()}
                    last_trade[sym] = time.time()
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] SHORT {sym.split('/')[0]} @ {price:.4f}")

            # Exit logic (TP +0.72%, SL -0.34%, max 42 sec)
            remove = []
            for sym, pos in positions.items():
                price = await get_price(sym)
                pnl_pct = (price - pos['entry']) / pos['entry'] * (1 if pos['side']=='LONG' else -1) * 100
                held = time.time() - pos['time']

                if pnl_pct >= 0.72 or pnl_pct <= -0.34 or held > 42:
                    profit_usd = 35 * 20 * (pnl_pct / 100) * 0.998   # $35 × 20x
                    balance += profit_usd
                    status = "WIN" if profit_usd > 0 else "LOSS"
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {status} {sym.split('/')[0]} +${profit_usd:+.2f} → ${balance:.1f}")
                    remove.append(sym)

            for s in remove:
                del positions[s]

            if int(time.time()) % 18 == 0:
                daily = (balance / initial_balance - 1) * 100
                print(f"BALANCE ${balance:.1f} | Today +{daily:+.2f}% | Open {len(positions)}")

            await asyncio.sleep(1.3)

        except Exception as e:
            print("Error:", e)
            await asyncio.sleep(4)

asyncio.run(bot())
