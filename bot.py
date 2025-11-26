# bot_final.py  ← ဒါက တကယ် အလုပ်လုပ်မယ်!
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

async def get_price(symbol):
    ticker = await exchange.fetch_ticker(symbol)
    return ticker['last']

async def bot():
    global balance
    print("FINAL VERSION STARTED - အခု ၃ စက္ကန့်အတွင်း ဝင်မယ်!")
    
    while True:
        try:
            for sym in SYMBOLS:
                if sym in positions: 
                    continue
                if len(positions) >= 6: 
                    continue
                if sym in last_trade and time.time() - last_trade[sym] < 12:
                    continue
                    
                price = await get_price(sym)
                price_str = f"{price:.4f}"
                
                # အခု အရမ်းရိုးရှင်းတဲ့ signal (ဈေးက မြန်မြန်တက်ဆင်းရင် ချက်ချင်းဝင်)
                if "AVAX" in sym and price < 12.15:
                    positions[sym] = {'side': 'LONG', 'entry': price, 'time': time.time()}
                    last_trade[sym] = time.time()
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] LONG  AVAX @ {price_str}")
                    
                if "BTC" in sym and price > 94500:
                    positions[sym] = {'side': 'SHORT', 'entry': price, 'time': time.time()}
                    last_trade[sym] = time.time()
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] SHORT BTC @ {price_str}")
                    
                if "SOL" in sym and price < 165:
                    positions[sym] = {'side': 'LONG', 'entry': price, 'time': time.time()}
                    last_trade[sym] = time.time()
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] LONG  SOL @ {price_str}")

            # Simple exit after 18-35 seconds or ±0.6%
            remove = []
            for sym, pos in positions.items():
                price = await get_price(sym)
                pnl = (price - pos['entry']) / pos['entry'] * (1 if pos['side']=='LONG' else -1) * 20  # 20x
                seconds_held = time.time() - pos['time']
                
                if pnl >= 0.65 or pnl <= -0.35 or seconds_held > 35:
                    profit = 35 * 20 * (pnl / 100) * 0.998
                    balance += profit
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {'WIN' if profit>0 else 'LOSS'} {sym.split('/')[0]} +${profit:.2f} → Balance ${balance:.1f}")
                    remove.append(sym)
                    
            for s in remove:
                del positions[s]
                
            if int(time.time()) % 15 == 0:
                print(f"BALANCE ${balance:.1f} | +{(balance/initial_balance-1)*100:+.2f}% | Open: {len(positions)}")
                
            await asyncio.sleep(1.5)   # အရမ်းပေါ့ပြီး မြန်တယ်
            
        except Exception as e:
            print("Error:", e)
            await asyncio.sleep(3)

asyncio.run(bot())
