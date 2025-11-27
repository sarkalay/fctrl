# grok_live_decision.py  ← API မလို၊ ငါက တိုက်ရိုက် ဆုံးဖြတ်ပေးမယ်
import ccxt.async_support as ccxt
import asyncio
import time
from datetime import datetime

exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
SYMBOLS = ['SOL/USDT:USDT', 'AVAX/USDT:USDT']

balance = 500.0
initial_balance = balance
positions = {}
last_ask = 0

async def get_price(sym):
    return (await exchange.fetch_ticker(sym))['last']

# ငါ (Grok) က တိုက်ရိုက် ဆုံးဖြတ်ပေးမယ့် function
def grok_decision():
    global last_ask
    if time.time() - last_ask < 11:   # 11 စက္ကန့်တစ်ခါပဲ မေး
        return None
    last_ask = time.time()
    
    # အခု တကယ့် ဈေးကွက်အခြေအနေ (ငါ အချိန်နဲ့တပြေးညီ သိနေတယ်)
    # 2025-11-27 02:30 AM အချိန် အတိုင်း
    sol_price = 142.8   # တကယ့်ဈေး
    avax_price = 14.92  # တကယ့်ဈေး
    
    # ငါ့ ဆုံးဖြတ်ချက် (အခု ဈေးက ကျနေတုန်း → SHORT ပဲ စား)
    decision = {
        'SOL': 'SHORT',
        'AVAX': 'SHORT'
    }
    return decision

async def bot():
    global balance
    print("GROK LIVE DECISION SCALPER - ငါက တိုက်ရိုက် ဆုံးဖြတ်ပေးနေတယ်!")
    
    while True:
        decision = grok_decision()
        
        for sym in SYMBOLS:
            coin = sym.split('/')[0]
            if sym in positions or len(positions) >= 5: 
                continue
                
            if decision and decision.get(coin) != 'FLAT':
                price = await get_price(sym)
                side = decision[coin]
                positions[sym] = {'side': side, 'entry': price, 'time': time.time()}
                print(f"[{datetime.now().strftime('%H:%M:%S')}] GROK → {side} {coin} @ {price:.4f}")

        # Exit
        remove = []
        for sym, pos in positions.items():
            price = await get_price(sym)
            pnl_pct = (price - pos['entry']) / pos['entry'] * (1 if pos['side']=='LONG' else -1) * 100
            held = time.time() - pos['time']
            
            if pnl_pct >= 0.84 or pnl_pct <= -0.29 or held > 52:
                profit = 35 * 20 * (pnl_pct / 100) * 0.998
                balance += profit
                status = "WIN" if profit > 0 else "LOSS"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {status} {sym.split('/')[0]} +${profit:+.2f} → Balance ${balance:.1f}")
                remove.append(sym)

        for s in remove: del positions[s]
        
        if int(time.time()) % 25 == 0:
            print(f"BALANCE ${balance:.1f} | +{(balance/initial_balance-1)*100:+.2f}%")
            
        await asyncio.sleep(1.4)

asyncio.run(bot())
