import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Try to import learn script
try:
    from learn_script import SelfLearningAITrader
    LEARN_SCRIPT_AVAILABLE = True
    print("✅ Learn script loaded successfully!")
except ImportError as e:
    print(f"❌ Learn script import failed: {e}")
    LEARN_SCRIPT_AVAILABLE = False

import requests
import json
import time
import re
import math
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz
import pandas as pd
import types

# Colorama setup
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    print("Warning: Colorama not installed. Run: pip install colorama")

# Load environment variables
load_dotenv()

# Global color variables for fallback
if not COLORAMA_AVAILABLE:
    class DummyColors:
        def __getattr__(self, name):
            return ''
    
    Fore = DummyColors()
    Back = DummyColors() 
    Style = DummyColors()

# ==================== V5 FREEDOM AI TRADER ====================
def should_close_trade(trade, current_price, atr_14):
    """SMART EXIT SYSTEM - AI Controlled"""
    if trade['direction'] == 'LONG':
        pnl_pct = (current_price - trade['entry_price']) / trade['entry_price'] * 100 * trade['leverage']
    else:  # SHORT
        pnl_pct = (trade['entry_price'] - current_price) / trade['entry_price'] * 100 * trade['leverage']
    
    # Peak PnL tracking
    if 'peak_pnl' not in trade or pnl_pct > trade['peak_pnl']:
        trade['peak_pnl'] = pnl_pct

    peak = trade['peak_pnl']

    # 1. Quick Profit Taking
    if peak >= 8.0 and not trade.get('partial_done', False):
        trade['partial_done'] = True
        return {
            "should_close": True,
            "partial_percent": 50,
            "close_type": "QUICK_PROFIT",
            "reason": f"Lock 50% profit at +{peak:.1f}%",
            "confidence": 100
        }

    # 2. Breakeven Protection
    if peak >= 10.0 and not trade.get('breakeven_done', False):
        trade['breakeven_done'] = True
        return {
            "should_close": False,
            "move_sl_to": trade['entry_price'],
            "close_type": "BREAKEVEN",
            "reason": f"Breakeven activated at +{peak:.1f}%",
            "confidence": 100
        }

    # 3. Hard Stop Loss
    if pnl_pct <= -6.0:
        return {
            "should_close": True,
            "partial_percent": 100,
            "close_type": "STOP_LOSS",
            "reason": "Hard stop loss hit",
            "confidence": 100
        }

    # 4. Profit Protection
    if peak >= 15.0:
        profit_floor = peak * 0.70  # 70% of peak
        if pnl_pct <= profit_floor and trade.get('partial_done', False):
            return {
                "should_close": True,
                "partial_percent": 100,
                "close_type": "PROFIT_PROTECTION",
                "reason": f"Protecting profits from {peak:.1f}% to {pnl_pct:.1f}%",
                "confidence": 100
            }

    return {"should_close": False}

# Use conditional inheritance with proper method placement
if LEARN_SCRIPT_AVAILABLE:
    class FreedomAITrader(SelfLearningAITrader):
        def __init__(self):
            super().__init__()
            self._initialize_trading()
else:
    class FreedomAITrader(object):
        def __init__(self):
            self.mistakes_history = []
            self.learned_patterns = {}
            self.performance_stats = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'common_mistakes': {},
                'improvement_areas': []
            }
            self._initialize_trading()

def _initialize_trading(self):
    """Initialize trading with FREEDOM approach"""
    self.binance_api_key = os.getenv('BINANCE_API_KEY')
    self.binance_secret = os.getenv('BINANCE_SECRET_KEY')
    self.openrouter_key = os.getenv('OPENROUTER_API_KEY')
    
    self.Fore = Fore
    self.Back = Back
    self.Style = Style
    self.COLORAMA_AVAILABLE = COLORAMA_AVAILABLE
    
    self.thailand_tz = pytz.timezone('Asia/Bangkok')
    
    # 🎯 FREEDOM TRADING PARAMETERS
    self.total_budget = 500
    self.available_budget = 500
    self.max_position_size_percent = 15  # Increased for more freedom
    self.max_concurrent_trades = 6  # More concurrent trades
    
    # Multiple pairs for more opportunities
    self.available_pairs = [
        "SOLUSDT", "BTCUSDT", "ETHUSDT", "ADAUSDT", "MATICUSDT"
    ]
    
    self.ai_opened_trades = {}
    self.pending_entries = {}  # For partial entry system
    
    # Trading history
    self.real_trade_history_file = "freedom_ai_trading_history.json"
    self.real_trade_history = self.load_real_trade_history()
    
    self.real_total_trades = 0
    self.real_winning_trades = 0
    self.real_total_pnl = 0.0
    
    # Precision settings
    self.quantity_precision = {}
    self.price_precision = {}
    
    # FREEDOM SETTINGS
    self.allow_reverse_positions = True
    self.monitoring_interval = 120  # 2 minutes for faster action
    self.ai_aggressiveness = "BALANCED"  # BALANCED, AGGRESSIVE, CONSERVATIVE
    
    # Validate APIs
    self.validate_api_keys()
    
    # Initialize Binance client
    try:
        self.binance = Client(self.binance_api_key, self.binance_secret)
        self.print_color(f"🤖 FREEDOM AI TRADER ACTIVATED! 🤖", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color(f"💰 TOTAL BUDGET: ${self.total_budget}", self.Fore.GREEN + self.Style.BRIGHT)
        self.print_color(f"🎯 AGGRESSIVENESS: {self.ai_aggressiveness}", self.Fore.MAGENTA + self.Style.BRIGHT)
        self.print_color(f"⏰ MONITORING: 2 MINUTE INTERVAL", self.Fore.RED + self.Style.BRIGHT)
        self.print_color(f"📊 Max Positions: {self.max_concurrent_trades}", self.Fore.YELLOW + self.Style.BRIGHT)
        self.print_color(f"🎯 TRADING PAIRS: {len(self.available_pairs)}", self.Fore.BLUE + self.Style.BRIGHT)
        if LEARN_SCRIPT_AVAILABLE:
            self.print_color(f"🧠 SELF-LEARNING AI: ENABLED", self.Fore.MAGENTA + self.Style.BRIGHT)
    except Exception as e:
        self.print_color(f"Binance initialization failed: {e}", self.Fore.RED)
        self.binance = None
    
    self.validate_config()
    if self.binance:
        self.setup_futures()
        self.load_symbol_precision()

# Add the method to both classes
FreedomAITrader._initialize_trading = _initialize_trading

def set_aggressiveness(self, level="BALANCED"):
    """Set how aggressive AI should be"""
    self.ai_aggressiveness = level
    self.print_color(f"🎯 AI Aggressiveness set to: {level}", self.Fore.GREEN + self.Style.BRIGHT)

def get_aggressiveness_prompt(self):
    """Get prompt based on aggressiveness level"""
    prompts = {
        "CONSERVATIVE": """
You are a WISE, patient trader. You wait for high-probability setups.
- Max position: $25
- Min confidence: 65%
- Prefer clear trend alignment
- Take profits early
- Priority: Capital preservation
""",
        
        "BALANCED": """
You are a SMART, balanced trader. You take calculated risks.
- Max position: $40  
- Min confidence: 55%
- Good risk-reward ratios
- Mix of trend and counter-trend
- Priority: Consistent growth
""",
        
        "AGGRESSIVE": """
You are a BOLD, opportunistic trader. You create opportunities.
- Max position: $60
- Min confidence: 45% 
- Action over perfection
- Quick entries and exits
- Priority: Maximum opportunity capture
""",
        
        "VIKING": """
You are CRYPTO VIKING! Fearless and decisive.
- Max position: $80
- Min confidence: 35%
- Better to trade and learn than wait
- Trust your instincts
- Small losses are tuition fees
- Priority: Learning and aggressive growth
"""
    }
    return prompts.get(self.ai_aggressiveness, prompts["BALANCED"])

def format_market_analysis(self, mtf):
    """Format market analysis for AI"""
    if not mtf:
        return "Market data unavailable"
    
    analysis = "MULTI-TIMEFRAME ANALYSIS:\n"
    for tf in ['5m', '15m', '1h', '4h']:
        if tf in mtf:
            data = mtf[tf]
            trend = data.get('trend', 'N/A')
            crossover = data.get('crossover', 'NONE')
            rsi = data.get('rsi', 50)
            vol_spike = "🔥" if data.get('vol_spike') else "➖"
            
            analysis += f"  {tf.upper()}: {trend} | {crossover} | RSI:{rsi} | Vol:{vol_spike}\n"
    
    return analysis

def get_ai_trading_decision(self, pair, market_data, current_trade=None):
    """AI with TRUE FREEDOM to make decisions"""
    max_retries = 2
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            if not self.openrouter_key:
                return self.get_freedom_fallback_decision(pair, market_data)
            
            current_price = market_data.get('current_price', 0)
            mtf = market_data.get('mtf_analysis', {})
            
            # Learning context
            learning_context = ""
            if LEARN_SCRIPT_AVAILABLE and hasattr(self, 'get_learning_enhanced_prompt'):
                learning_context = self.get_learning_enhanced_prompt(pair, market_data)

            # Current position context
            position_context = ""
            if current_trade:
                current_pnl = self.calculate_current_pnl(current_trade, current_price)
                position_context = f"""
CURRENT POSITION:
- Direction: {current_trade['direction']}
- Entry: ${current_trade['entry_price']:.4f}
- Current PnL: {current_pnl:.1f}%
- Consider reverse if trend changed?
"""

            # FREEDOM PROMPT - No restrictive rules
            prompt = f"""
{self.get_aggressiveness_prompt()}

YOU ARE AN AUTONOMOUS AI TRADER WITH COMPLETE FREEDOM.

BUDGET: ${self.available_budget:.2f}
TRADING PAIR: {pair}
CURRENT PRICE: ${current_price:.6f}

{self.format_market_analysis(mtf)}
{position_context}
{learning_context}

YOUR MISSION:
Make your OWN trading decision. You have COMPLETE freedom to:
- Enter when YOU see opportunity (no perfect setup needed)
- Use any position size that makes sense
- Take calculated risks
- Trade against the trend if justified
- Learn through action

NO RESTRICTIVE RULES - Trust your analysis and instincts.

Return JSON with your decision:
{{
    "decision": "LONG" | "SHORT" | "HOLD" | "REVERSE_LONG" | "REVERSE_SHORT",
    "position_size_usd": number (10-80 based on confidence),
    "entry_price": number,
    "leverage": number (3-8),
    "confidence": 0-100,
    "reasoning": "Your honest reasoning",
    "should_reverse": boolean
}}
"""
            headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com",
                "X-Title": "Freedom AI Trader"
            }
            
            data = {
                "model": "deepseek/deepseek-chat-v3.1",
                "messages": [
                    {"role": "system", "content": "You are a fearless AI trader with complete freedom. Make bold but calculated decisions."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.4,  # Slightly higher for more creativity
                "max_tokens": 800
            }
            
            self.print_color(f"🧠 Freedom AI Analyzing {pair}...", self.Fore.MAGENTA + self.Style.BRIGHT)
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=45)
            
            if response.status_code == 200:
                result = response.json()
                ai_response = result['choices'][0]['message']['content'].strip()
                return self.parse_freedom_ai_response(ai_response, pair, current_price, current_trade)
            else:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                    
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
    
    return self.get_freedom_fallback_decision(pair, market_data)

def parse_freedom_ai_response(self, ai_response, pair, current_price, current_trade=None):
    """Parse AI's free-form response"""
    try:
        json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            decision_data = json.loads(json_str)
            
            decision = decision_data.get('decision', 'HOLD').upper()
            position_size = float(decision_data.get('position_size_usd', 0))
            leverage = int(decision_data.get('leverage', 5))
            confidence = float(decision_data.get('confidence', 50))
            reasoning = decision_data.get('reasoning', 'Freedom AI Analysis')
            
            # Validate and adjust based on aggressiveness
            if position_size > 80:
                position_size = 80
            elif position_size < 10 and decision != "HOLD":
                position_size = 15  # Minimum meaningful position
                
            if leverage > 8:
                leverage = 8
            elif leverage < 3:
                leverage = 5
                
            return {
                "decision": decision,
                "position_size_usd": position_size,
                "entry_price": current_price,
                "leverage": leverage,
                "confidence": confidence,
                "reasoning": reasoning,
                "should_reverse": decision.startswith('REVERSE_')
            }
        
        # If no JSON found, try to extract decision from text
        decision = "HOLD"
        if "LONG" in ai_response.upper() and "SHORT" not in ai_response.upper():
            decision = "LONG"
        elif "SHORT" in ai_response.upper() and "LONG" not in ai_response.upper():
            decision = "SHORT"
            
        return {
            "decision": decision,
            "position_size_usd": 25,
            "entry_price": current_price,
            "leverage": 5,
            "confidence": 50,
            "reasoning": "Extracted from AI text response",
            "should_reverse": False
        }
        
    except Exception as e:
        self.print_color(f"Freedom AI response parsing failed: {e}", self.Fore.RED)
        return self.get_freedom_fallback_decision(pair, {'current_price': current_price})

def get_freedom_fallback_decision(self, pair, market_data):
    """Fallback that encourages more trading"""
    current_price = market_data['current_price']
    mtf = market_data.get('mtf_analysis', {})
    
    # Simplified scoring - more likely to trade
    score = 0
    
    h1_data = mtf.get('1h', {})
    m15_data = mtf.get('15m', {})
    
    # Basic trend analysis
    if h1_data.get('trend') == 'BULLISH':
        score += 2
    elif h1_data.get('trend') == 'BEARISH':
        score -= 2
        
    # Crossover signals
    if m15_data.get('crossover') == 'GOLDEN':
        score += 3
    elif m15_data.get('crossover') == 'DEATH':
        score -= 3
        
    # RSI conditions (wider range)
    h1_rsi = h1_data.get('rsi', 50)
    if h1_rsi < 40:
        score += 2
    elif h1_rsi > 60:
        score -= 2
        
    # Make decision (lower thresholds)
    if score >= 2:  # Lower threshold for entry
        return {
            "decision": "LONG",
            "position_size_usd": 30,
            "entry_price": current_price,
            "leverage": 5,
            "confidence": 60,
            "reasoning": f"Fallback: Bullish score {score}",
            "should_reverse": False
        }
    elif score <= -2:
        return {
            "decision": "SHORT", 
            "position_size_usd": 30,
            "entry_price": current_price,
            "leverage": 5,
            "confidence": 60,
            "reasoning": f"Fallback: Bearish score {score}",
            "should_reverse": False
        }
    else:
        return {
            "decision": "HOLD",
            "position_size_usd": 0,
            "entry_price": current_price,
            "leverage": 5,
            "confidence": 40,
            "reasoning": f"Fallback: Neutral score {score}",
            "should_reverse": False
        }

def calculate_current_pnl(self, trade, current_price):
    """Calculate current PnL percentage"""
    try:
        if trade['direction'] == 'LONG':
            pnl_percent = ((current_price - trade['entry_price']) / trade['entry_price']) * 100 * trade['leverage']
        else:
            pnl_percent = ((trade['entry_price'] - current_price) / trade['entry_price']) * 100 * trade['leverage']
        return pnl_percent
    except:
        return 0

def execute_freedom_trade(self, pair, ai_decision):
    """Execute trade with freedom approach"""
    try:
        decision = ai_decision["decision"]
        position_size_usd = ai_decision["position_size_usd"]
        leverage = ai_decision["leverage"]
        confidence = ai_decision["confidence"]
        reasoning = ai_decision["reasoning"]
        
        # Handle reverse positions
        if decision.startswith('REVERSE_'):
            if pair in self.ai_opened_trades:
                current_trade = self.ai_opened_trades[pair]
                return self.execute_reverse_position(pair, ai_decision, current_trade)
            else:
                self.print_color(f"❌ Cannot reverse: No active position for {pair}", self.Fore.RED)
                return False
        
        if decision == "HOLD" or position_size_usd <= 0:
            self.print_color(f"🟡 Freedom AI decides to HOLD {pair}", self.Fore.YELLOW)
            return False
        
        # Check if we can open position
        if pair in self.ai_opened_trades:
            self.print_color(f"🚫 Cannot open {pair}: Position already exists", self.Fore.RED)
            return False
        
        if len(self.ai_opened_trades) >= self.max_concurrent_trades:
            self.print_color(f"🚫 Cannot open {pair}: Max concurrent trades reached", self.Fore.RED)
            return False
            
        if position_size_usd > self.available_budget:
            self.print_color(f"🚫 Cannot open {pair}: Insufficient budget", self.Fore.RED)
            return False
        
        # Calculate quantity
        entry_price = ai_decision["entry_price"]
        quantity = self.calculate_quantity(pair, entry_price, position_size_usd, leverage)
        if quantity is None:
            return False
        
        # Display trade decision
        direction_color = self.Fore.GREEN + self.Style.BRIGHT if decision == 'LONG' else self.Fore.RED + self.Style.BRIGHT
        direction_icon = "🟢 LONG" if decision == 'LONG' else "🔴 SHORT"
        
        self.print_color(f"\n🤖 FREEDOM AI TRADE EXECUTION", self.Fore.CYAN + self.Style.BRIGHT)
        self.print_color("=" * 80, self.Fore.CYAN)
        self.print_color(f"{direction_icon} {pair}", direction_color)
        self.print_color(f"POSITION SIZE: ${position_size_usd:.2f}", self.Fore.GREEN + self.Style.BRIGHT)
        self.print_color(f"LEVERAGE: {leverage}x ⚡", self.Fore.RED + self.Style.BRIGHT)
        self.print_color(f"CONFIDENCE: {confidence}%", self.Fore.YELLOW + self.Style.BRIGHT)
        self.print_color(f"REASONING: {reasoning}", self.Fore.WHITE)
        self.print_color("=" * 80, self.Fore.CYAN)
        
        # Execute live trade
        if self.binance:
            entry_side = 'BUY' if decision == 'LONG' else 'SELL'
            
            try:
                self.binance.futures_change_leverage(symbol=pair, leverage=leverage)
            except Exception as e:
                self.print_color(f"Leverage change failed: {e}", self.Fore.YELLOW)
            
            order = self.binance.futures_create_order(
                symbol=pair,
                side=entry_side,
                type='MARKET',
                quantity=quantity
            )
        
        # Update budget and track trade
        self.available_budget -= position_size_usd
        
        self.ai_opened_trades[pair] = {
            "pair": pair,
            "direction": decision,
            "entry_price": entry_price,
            "quantity": quantity,
            "position_size_usd": position_size_usd,
            "leverage": leverage,
            "entry_time": time.time(),
            "status": 'ACTIVE',
            'ai_confidence': confidence,
            'ai_reasoning': reasoning,
            'entry_time_th': self.get_thailand_time(),
            'has_tp_sl': False,
            'peak_pnl': 0
        }
        
        self.print_color(f"✅ FREEDOM TRADE EXECUTED: {pair} {decision} | Size: ${position_size_usd:.2f}", self.Fore.GREEN + self.Style.BRIGHT)
        return True
        
    except Exception as e:
        self.print_color(f"❌ Trade execution failed: {e}", self.Fore.RED)
        return False

# Add all other necessary methods (shortened for brevity)
def load_real_trade_history(self):
    try:
        if os.path.exists(self.real_trade_history_file):
            with open(self.real_trade_history_file, 'r') as f:
                history = json.load(f)
                self.real_total_trades = len(history)
                self.real_winning_trades = len([t for t in history if t.get('pnl', 0) > 0])
                self.real_total_pnl = sum(t.get('pnl', 0) for t in history)
                return history
        return []
    except Exception as e:
        self.print_color(f"Error loading trade history: {e}", self.Fore.RED)
        return []

def save_real_trade_history(self):
    try:
        with open(self.real_trade_history_file, 'w') as f:
            json.dump(self.real_trade_history, f, indent=2)
    except Exception as e:
        self.print_color(f"Error saving trade history: {e}", self.Fore.RED)

def add_trade_to_history(self, trade_data):
    try:
        trade_data['close_time'] = self.get_thailand_time()
        trade_data['close_timestamp'] = time.time()
        trade_data['trade_type'] = 'REAL'
        
        if 'exit_price' not in trade_data:
            current_price = self.get_current_price(trade_data['pair'])
            trade_data['exit_price'] = current_price
        
        if 'peak_pnl_pct' not in trade_data:
            if 'peak_pnl' in trade_data:
                trade_data['peak_pnl_pct'] = trade_data['peak_pnl']
            else:
                if trade_data['direction'] == 'LONG':
                    peak_pct = ((trade_data['exit_price'] - trade_data['entry_price']) / trade_data['entry_price']) * 100 * trade_data.get('leverage', 1)
                else:
                    peak_pct = ((trade_data['entry_price'] - trade_data['exit_price']) / trade_data['entry_price']) * 100 * trade_data.get('leverage', 1)
                trade_data['peak_pnl_pct'] = max(0, peak_pct)
        
        if trade_data.get('partial_percent', 100) < 100:
            trade_data['display_type'] = f"PARTIAL_{trade_data['partial_percent']}%"
        else:
            trade_data['display_type'] = "FULL_CLOSE"
        
        self.real_trade_history.append(trade_data)
        
        if LEARN_SCRIPT_AVAILABLE:
            self.learn_from_mistake(trade_data)
            self.adaptive_learning_adjustment()
        
        self.performance_stats['total_trades'] += 1
        pnl = trade_data.get('pnl', 0)
        self.real_total_pnl += pnl
        if pnl > 0:
            self.real_winning_trades += 1
            self.performance_stats['winning_trades'] += 1
        else:
            self.performance_stats['losing_trades'] += 1
            
        if len(self.real_trade_history) > 200:
            self.real_trade_history = self.real_trade_history[-200:]
        self.save_real_trade_history()
        
        try:
            from data_collector import log_trade_for_ml
            log_trade_for_ml(trade_data)
        except:
            pass
        
        if trade_data.get('partial_percent', 100) < 100:
            self.print_color(f"📝 Partial close saved: {trade_data['pair']} {trade_data['partial_percent']}% | P&L: ${pnl:.2f}", self.Fore.CYAN)
        else:
            self.print_color(f"📝 Trade saved: {trade_data['pair']} | P&L: ${pnl:.2f}", self.Fore.CYAN)
            
    except Exception as e:
        self.print_color(f"Error adding trade to history: {e}", self.Fore.RED)

def get_thailand_time(self):
    now_utc = datetime.now(pytz.utc)
    thailand_time = now_utc.astimezone(self.thailand_tz)
    return thailand_time.strftime('%Y-%m-%d %H:%M:%S')

def print_color(self, text, color="", style=""):
    if self.COLORAMA_AVAILABLE:
        print(f"{style}{color}{text}")
    else:
        print(text)

def validate_api_keys(self):
    issues = []
    
    if not self.binance_api_key or self.binance_api_key == "your_binance_api_key_here":
        issues.append("Binance API Key not configured")
    
    if not self.binance_secret or self.binance_secret == "your_binance_secret_key_here":
        issues.append("Binance Secret Key not configured")
        
    if not self.openrouter_key or self.openrouter_key == "your_openrouter_api_key_here":
        issues.append("OpenRouter API Key not configured - AI will use fallback decisions")
    
    if issues:
        self.print_color("🚨 CONFIGURATION ISSUES FOUND:", self.Fore.RED + self.Style.BRIGHT)
        for issue in issues:
            self.print_color(f"   ❌ {issue}", self.Fore.RED)
    
    return len(issues) == 0

def validate_config(self):
    if not all([self.binance_api_key, self.binance_secret, self.openrouter_key]):
        self.print_color("Missing API keys!", self.Fore.RED)
        return False
    try:
        if self.binance:
            self.binance.futures_exchange_info()
            self.print_color("✅ Binance connection successful!", self.Fore.GREEN + self.Style.BRIGHT)
        else:
            self.print_color("Binance client not available - Paper Trading only", self.Fore.YELLOW)
            return True
    except Exception as e:
        self.print_color(f"Binance connection failed: {e}", self.Fore.RED)
        return False
    return True

def setup_futures(self):
    if not self.binance:
        return
        
    try:
        for pair in self.available_pairs:
            try:
                self.binance.futures_change_leverage(symbol=pair, leverage=5)
                self.binance.futures_change_margin_type(symbol=pair, marginType='ISOLATED')
            except Exception as e:
                self.print_color(f"Leverage setup failed for {pair}: {e}", self.Fore.YELLOW)
        self.print_color("✅ Futures setup completed!", self.Fore.GREEN + self.Style.BRIGHT)
    except Exception as e:
        self.print_color(f"Futures setup failed: {e}", self.Fore.RED)

def load_symbol_precision(self):
    if not self.binance:
        for pair in self.available_pairs:
            try:
                response = requests.get(f'https://api.binance.com/api/v3/exchangeInfo?symbol={pair}')
                if response.status_code == 200:
                    data = response.json()
                    symbol_info = next((s for s in data['symbols'] if s['symbol'] == pair), None)
                    if symbol_info:
                        for f in symbol_info['filters']:
                            if f['filterType'] == 'LOT_SIZE':
                                step_size = f['stepSize']
                                qty_precision = len(step_size.split('.')[1].rstrip('0')) if '.' in step_size else 0
                                self.quantity_precision[pair] = qty_precision
                            elif f['filterType'] == 'PRICE_FILTER':
                                tick_size = f['tickSize']
                                price_precision = len(tick_size.split('.')[1].rstrip('0')) if '.' in tick_size else 0
                                self.price_precision[pair] = price_precision
                else:
                    self.quantity_precision[pair] = 3
                    self.price_precision[pair] = 4
            except:
                self.quantity_precision[pair] = 3
                self.price_precision[pair] = 4
        self.print_color("Symbol precision loaded from Binance API", self.Fore.GREEN)
        return
        
    try:
        exchange_info = self.binance.futures_exchange_info()
        for symbol in exchange_info['symbols']:
            pair = symbol['symbol']
            if pair not in self.available_pairs:
                continue
            for f in symbol['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    step_size = f['stepSize']
                    qty_precision = len(step_size.split('.')[1].rstrip('0')) if '.' in step_size else 0
                    self.quantity_precision[pair] = qty_precision
                elif f['filterType'] == 'PRICE_FILTER':
                    tick_size = f['tickSize']
                    price_precision = len(tick_size.split('.')[1].rstrip('0')) if '.' in tick_size else 0
                    self.price_precision[pair] = price_precision
        self.print_color("✅ Symbol precision loaded", self.Fore.GREEN + self.Style.BRIGHT)
    except Exception as e:
        self.print_color(f"Error loading symbol precision: {e}", self.Fore.RED)

def get_current_price(self, pair):
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            if self.binance:
                ticker = self.binance.futures_symbol_ticker(symbol=pair)
                return float(ticker['price'])
            
            response = requests.get(
                f'https://api.binance.com/api/v3/ticker/price?symbol={pair}',
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return float(data['price'])
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
    
    fallback_prices = {
        "SOLUSDT": 140.0, "BTCUSDT": 45000.0, "ETHUSDT": 2500.0,
        "ADAUSDT": 0.5, "MATICUSDT": 0.8
    }
    return fallback_prices.get(pair, 100.0)

def calculate_quantity(self, pair, entry_price, position_size_usd, leverage):
    try:
        if entry_price <= 0:
            return None
            
        notional_value = position_size_usd * leverage
        quantity = notional_value / entry_price
        
        precision = self.quantity_precision.get(pair, 3)
        quantity = round(quantity, precision)
        
        if quantity <= 0:
            return None
            
        self.print_color(f"📊 Position: ${position_size_usd} | Leverage: {leverage}x | Quantity: {quantity}", self.Fore.CYAN)
        return quantity
        
    except Exception as e:
        self.print_color(f"Quantity calculation failed: {e}", self.Fore.RED)
        return None

def get_price_history(self, pair, limit=50):
    try:
        if not self.binance:
            return self._get_mtf_data_via_api(pair, limit)
        
        intervals = {
            '5m': (Client.KLINE_INTERVAL_5MINUTE, 50),
            '15m': (Client.KLINE_INTERVAL_15MINUTE, 50),
            '1h': (Client.KLINE_INTERVAL_1HOUR, 50),
            '4h': (Client.KLINE_INTERVAL_4HOUR, 30)
        }

        mtf = {}
        current_price = self.get_current_price(pair)

        for name, (interval, lim) in intervals.items():
            klines = self.binance.futures_klines(symbol=pair, interval=interval, limit=lim)
            if not klines:
                continue

            closes = [float(k[4]) for k in klines]
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            volumes = [float(k[5]) for k in klines]

            ema9 = self.calculate_ema(closes, 9)
            ema21 = self.calculate_ema(closes, 21)
            rsi = self.calculate_rsi(closes, 14)[-1] if len(closes) > 14 else 50

            crossover = 'NONE'
            if len(ema9) >= 2 and len(ema21) >= 2:
                if ema9[-2] < ema21[-2] and ema9[-1] > ema21[-1]:
                    crossover = 'GOLDEN'
                elif ema9[-2] > ema21[-2] and ema9[-1] < ema21[-1]:
                    crossover = 'DEATH'

            vol_spike = self.calculate_volume_spike(volumes)

            mtf[name] = {
                'current_price': closes[-1],
                'change_1h': ((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) > 1 else 0,
                'ema9': round(ema9[-1], 6) if ema9[-1] else 0,
                'ema21': round(ema21[-1], 6) if ema21[-1] else 0,
                'trend': 'BULLISH' if ema9[-1] > ema21[-1] else 'BEARISH',
                'crossover': crossover,
                'rsi': round(rsi, 1),
                'vol_spike': vol_spike,
                'support': round(min(lows[-10:]), 6),
                'resistance': round(max(highs[-10:]), 6)
            }

        main = mtf.get('1h', {})
        return {
            'current_price': current_price,
            'price_change': main.get('change_1h', 0),
            'support_levels': [mtf['1h']['support'], mtf['4h']['support']] if '4h' in mtf else [],
            'resistance_levels': [mtf['1h']['resistance'], mtf['4h']['resistance']] if '4h' in mtf else [],
            'mtf_analysis': mtf
        }

    except Exception as e:
        self.print_color(f"MTF Analysis error: {e}", self.Fore.RED)
        return self._get_mtf_data_via_api(pair, limit)

def _get_mtf_data_via_api(self, pair, limit=50):
    try:
        intervals = {
            '5m': '5m',
            '15m': '15m', 
            '1h': '1h',
            '4h': '4h'
        }
        
        mtf = {}
        current_price = self.get_current_price(pair)
        
        for name, interval in intervals.items():
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                'symbol': pair,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                klines = response.json()
                
                closes = [float(k[4]) for k in klines]
                highs = [float(k[2]) for k in klines]
                lows = [float(k[3]) for k in klines]
                volumes = [float(k[5]) for k in klines]
                
                ema9 = self.calculate_ema(closes, 9)
                ema21 = self.calculate_ema(closes, 21)
                rsi = self.calculate_rsi(closes, 14)[-1] if len(closes) > 14 else 50
                
                crossover = 'NONE'
                if len(ema9) >= 2 and len(ema21) >= 2:
                    if ema9[-2] < ema21[-2] and ema9[-1] > ema21[-1]:
                        crossover = 'GOLDEN'
                    elif ema9[-2] > ema21[-2] and ema9[-1] < ema21[-1]:
                        crossover = 'DEATH'
                
                vol_spike = self.calculate_volume_spike(volumes)
                
                mtf[name] = {
                    'current_price': closes[-1],
                    'change_1h': ((closes[-1] - closes[-2]) / closes[-2] * 100) if len(closes) > 1 else 0,
                    'ema9': round(ema9[-1], 6) if ema9[-1] else 0,
                    'ema21': round(ema21[-1], 6) if ema21[-1] else 0,
                    'trend': 'BULLISH' if ema9[-1] > ema21[-1] else 'BEARISH',
                    'crossover': crossover,
                    'rsi': round(rsi, 1),
                    'vol_spike': vol_spike,
                    'support': round(min(lows[-10:]), 6),
                    'resistance': round(max(highs[-10:]), 6)
                }
        
        main = mtf.get('1h', {})
        return {
            'current_price': current_price,
            'price_change': main.get('change_1h', 0),
            'support_levels': [mtf['1h']['support'], mtf['4h']['support']] if '4h' in mtf else [],
            'resistance_levels': [mtf['1h']['resistance'], mtf['4h']['resistance']] if '4h' in mtf else [],
            'mtf_analysis': mtf
        }
        
    except Exception as e:
        self.print_color(f"API MTF Analysis error: {e}", self.Fore.RED)
        return {
            'current_price': self.get_current_price(pair),
            'price_change': 0,
            'support_levels': [],
            'resistance_levels': [],
            'mtf_analysis': {}
        }

def calculate_ema(self, data, period):
    if len(data) < period:
        return [None] * len(data)
    df = pd.Series(data)
    return df.ewm(span=period, adjust=False).mean().tolist()

def calculate_rsi(self, data, period=14):
    if len(data) < period + 1:
        return [50] * len(data)
    df = pd.Series(data)
    delta = df.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).tolist()

def calculate_volume_spike(self, volumes, window=10):
    if len(volumes) < window + 1:
        return False
    avg_vol = np.mean(volumes[-window-1:-1])
    current_vol = volumes[-1]
    return current_vol > avg_vol * 1.8

def execute_reverse_position(self, pair, ai_decision, current_trade):
    try:
        self.print_color(f"🔄 ATTEMPTING REVERSE POSITION FOR {pair}", self.Fore.YELLOW + self.Style.BRIGHT)
        
        close_success = self.close_trade_immediately(pair, current_trade, "REVERSE_POSITION")
        
        if close_success:
            time.sleep(2)
            
            if pair in self.ai_opened_trades:
                del self.ai_opened_trades[pair]
            
            self.print_color(f"🔍 Asking AI to confirm reverse position for {pair}...", self.Fore.BLUE)
            market_data = self.get_price_history(pair)
            
            new_ai_decision = self.get_ai_trading_decision(pair, market_data, None)
            
            if new_ai_decision["decision"] in ["LONG", "SHORT"] and new_ai_decision["position_size_usd"] > 0:
                current_direction = current_trade['direction']
                if current_direction == "LONG":
                    correct_reverse_direction = "SHORT"
                else:
                    correct_reverse_direction = "LONG"
                
                self.print_color(f"✅ AI CONFIRMED: Opening {correct_reverse_direction} {pair}", self.Fore.CYAN + self.Style.BRIGHT)
                
                reverse_decision = new_ai_decision.copy()
                reverse_decision["decision"] = correct_reverse_direction
                
                return self.execute_freedom_trade(pair, reverse_decision)
            else:
                self.print_color(f"🔄 AI changed mind, not opening reverse position for {pair}", self.Fore.YELLOW)
                return False
        else:
            self.print_color(f"❌ Reverse position failed: Could not close current trade", self.Fore.RED)
            return False
            
    except Exception as e:
        self.print_color(f"❌ Reverse position execution failed: {e}", self.Fore.RED)
        return False

def close_trade_immediately(self, pair, trade, close_reason="AI_DECISION", partial_percent=100):
    try:
        current_price = self.get_current_price(pair)
        
        if trade['direction'] == 'LONG':
            pnl = (current_price - trade['entry_price']) * trade['quantity'] * (partial_percent / 100)
        else:
            pnl = (trade['entry_price'] - current_price) * trade['quantity'] * (partial_percent / 100)
        
        peak_pnl_pct = 0.0
        if 'peak_pnl' in trade:
            peak_pnl_pct = trade['peak_pnl']
        else:
            if trade['direction'] == 'LONG':
                peak_pnl_pct = ((current_price - trade['entry_price']) / trade['entry_price']) * 100 * trade['leverage']
            else:
                peak_pnl_pct = ((trade['entry_price'] - current_price) / trade['entry_price']) * 100 * trade['leverage']
            peak_pnl_pct = max(0, peak_pnl_pct)
        
        if partial_percent < 100:
            remaining_quantity = trade['quantity'] * (1 - partial_percent / 100)
            closed_quantity = trade['quantity'] * (partial_percent / 100)
            closed_position_size = trade['position_size_usd'] * (partial_percent / 100)
            
            trade['quantity'] = remaining_quantity
            trade['position_size_usd'] = trade['position_size_usd'] * (1 - partial_percent / 100)
            
            partial_trade = trade.copy()
            partial_trade['status'] = 'PARTIAL_CLOSE'
            partial_trade['exit_price'] = current_price
            partial_trade['pnl'] = pnl
            partial_trade['close_reason'] = close_reason
            partial_trade['close_time'] = self.get_thailand_time()
            partial_trade['partial_percent'] = partial_percent
            partial_trade['closed_quantity'] = closed_quantity
            partial_trade['closed_position_size'] = closed_position_size
            partial_trade['peak_pnl_pct'] = round(peak_pnl_pct, 3)
            
            self.available_budget += closed_position_size + pnl
            self.add_trade_to_history(partial_trade)
            
            pnl_color = self.Fore.GREEN if pnl > 0 else self.Fore.RED
            self.print_color(f"✅ Partial Close | {pair} | {partial_percent}% | P&L: ${pnl:.2f} | Reason: {close_reason}", pnl_color)
            self.print_color(f"📊 Remaining: {remaining_quantity:.4f} {pair} (${trade['position_size_usd']:.2f})", self.Fore.CYAN)
            
            return True
            
        else:
            trade['status'] = 'CLOSED'
            trade['exit_price'] = current_price
            trade['pnl'] = pnl
            trade['close_reason'] = close_reason
            trade['close_time'] = self.get_thailand_time()
            trade['partial_percent'] = 100
            trade['peak_pnl_pct'] = round(peak_pnl_pct, 3)
            
            self.available_budget += trade['position_size_usd'] + pnl
            self.add_trade_to_history(trade.copy())
            
            pnl_color = self.Fore.GREEN if pnl > 0 else self.Fore.RED
            self.print_color(f"✅ Full Close | {pair} | P&L: ${pnl:.2f} | Reason: {close_reason}", pnl_color)
            
            if pair in self.ai_opened_trades:
                del self.ai_opened_trades[pair]
            
            return True
            
    except Exception as e:
        self.print_color(f"❌ Close failed: {e}", self.Fore.RED)
        return False

def get_ai_close_decision(self, pair, trade):
    try:
        current_price = self.get_current_price(pair)
        current_pnl = self.calculate_current_pnl(trade, current_price)
        
        if 'peak_pnl' not in trade:
            trade['peak_pnl'] = current_pnl
        if current_pnl > trade['peak_pnl']:
            trade['peak_pnl'] = current_pnl
        
        peak = trade['peak_pnl']

        if current_pnl <= -6.0:
            return {
                "should_close": True, 
                "close_type": "STOP_LOSS", 
                "close_reason": "Hard -6% stop loss", 
                "confidence": 100,
                "partial_percent": 100
            }

        if peak >= 8.0 and not trade.get('partial_done', False):
            trade['partial_done'] = True
            return {
                "should_close": True,
                "partial_percent": 50,
                "close_type": "QUICK_PROFIT",
                "reason": f"Take 50% profit at +{peak:.1f}%",
                "confidence": 100
            }

        if peak >= 10.0 and not trade.get('breakeven_done', False):
            trade['breakeven_done'] = True
            return {
                "should_close": False,
                "move_sl_to": trade['entry_price'],
                "close_type": "BREAKEVEN",
                "reason": f"Breakeven at +{peak:.1f}%",
                "confidence": 100
            }

        if peak >= 15.0:
            profit_floor = peak * 0.70
            if current_pnl <= profit_floor and trade.get('partial_done', False):
                return {
                    "should_close": True,
                    "partial_percent": 100,
                    "close_type": "PROFIT_PROTECTION",
                    "reason": f"Protect profits {peak:.1f}%→{current_pnl:.1f}%",
                    "confidence": 100
                }

        return {"should_close": False}

    except Exception as e:
        return {"should_close": False}

def monitor_positions(self):
    try:
        closed_trades = []
        for pair, trade in list(self.ai_opened_trades.items()):
            if trade['status'] != 'ACTIVE':
                continue
            
            if not trade.get('has_tp_sl', True):
                self.print_color(f"🔍 Freedom AI Checking {pair}...", self.Fore.BLUE)
                close_decision = self.get_ai_close_decision(pair, trade)
                
                if close_decision.get("should_close", False):
                    close_type = close_decision.get("close_type", "AI_DECISION")
                    confidence = close_decision.get("confidence", 0)
                    reasoning = close_decision.get("reason", "No reason provided")
                    partial_percent = close_decision.get("partial_percent", 100)
                    
                    full_close_reason = f"FREEDOM AI: {close_type} - {reasoning}"
                    
                    self.print_color(f"🎯 Freedom AI Decision: CLOSE {pair}", self.Fore.YELLOW + self.Style.BRIGHT)
                    self.print_color(f"📝 Close Type: {close_type} | Partial: {partial_percent}%", self.Fore.CYAN)
                    self.print_color(f"💡 Confidence: {confidence}% | Reasoning: {reasoning}", self.Fore.WHITE)
                    
                    success = self.close_trade_immediately(pair, trade, full_close_reason, partial_percent)
                    if success and partial_percent == 100:
                        closed_trades.append(pair)
                else:
                    if close_decision.get('confidence', 0) > 0:
                        reasoning = close_decision.get('reason', 'No reason provided')
                        self.print_color(f"🔍 Freedom AI wants to HOLD {pair} (Confidence: {close_decision.get('confidence', 0)}%)", self.Fore.GREEN)
                
        return closed_trades
                
    except Exception as e:
        self.print_color(f"Freedom AI Monitoring error: {e}", self.Fore.RED)
        return []

def display_dashboard(self):
    self.print_color(f"\n🤖 FREEDOM AI TRADING DASHBOARD - {self.get_thailand_time()}", self.Fore.CYAN + self.Style.BRIGHT)
    self.print_color("=" * 90, self.Fore.CYAN)
    self.print_color(f"🎯 MODE: FREEDOM AI | AGGRESSIVENESS: {self.ai_aggressiveness}", self.Fore.YELLOW + self.Style.BRIGHT)
    self.print_color(f"⏰ MONITORING: 2 MINUTE INTERVAL", self.Fore.RED + self.Style.BRIGHT)
    
    if hasattr(self, 'last_mtf') and self.last_mtf:
        self.print_color(" MULTI-TIMEFRAME SUMMARY", self.Fore.MAGENTA + self.Style.BRIGHT)
        for tf in ['15m', '1h', '4h']:
            if tf in self.last_mtf:
                data = self.last_mtf[tf]
                color = self.Fore.GREEN if data.get('trend') == 'BULLISH' else self.Fore.RED
                signal = f" | {data.get('crossover', '')}" if 'crossover' in data else ""
                rsi_text = f" | RSI: {data.get('rsi', 50)}" if 'rsi' in data else ""
                self.print_color(f"  {tf.upper()}: {data.get('trend', 'N/A')}{signal}{rsi_text}", color)
        self.print_color("   " + "-" * 60, self.Fore.CYAN)
    
    if LEARN_SCRIPT_AVAILABLE and hasattr(self, 'mistakes_history'):
        total_lessons = len(self.mistakes_history)
        if total_lessons > 0:
            self.print_color(f"🧠 AI HAS LEARNED FROM {total_lessons} MISTAKES", self.Fore.MAGENTA + self.Style.BRIGHT)
    
    active_count = 0
    total_unrealized = 0
    
    for pair, trade in self.ai_opened_trades.items():
        if trade['status'] == 'ACTIVE':
            active_count += 1
            current_price = self.get_current_price(pair)
            
            direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
            
            if trade['direction'] == 'LONG':
                unrealized_pnl = (current_price - trade['entry_price']) * trade['quantity']
            else:
                unrealized_pnl = (trade['entry_price'] - current_price) * trade['quantity']
                
            total_unrealized += unrealized_pnl
            pnl_color = self.Fore.GREEN + self.Style.BRIGHT if unrealized_pnl >= 0 else self.Fore.RED + self.Style.BRIGHT
            
            self.print_color(f"{direction_icon} {pair}", self.Fore.WHITE + self.Style.BRIGHT)
            self.print_color(f"   Size: ${trade['position_size_usd']:.2f} | Leverage: {trade['leverage']}x ⚡", self.Fore.WHITE)
            self.print_color(f"   Entry: ${trade['entry_price']:.4f} | Current: ${current_price:.4f}", self.Fore.WHITE)
            self.print_color(f"   P&L: ${unrealized_pnl:.2f}", pnl_color)
            self.print_color(f"   🎯 FREEDOM AI EXIT SYSTEM ACTIVE", self.Fore.YELLOW)
            self.print_color("   " + "-" * 60, self.Fore.CYAN)
    
    if active_count == 0:
        self.print_color("No active positions", self.Fore.YELLOW)
    else:
        total_color = self.Fore.GREEN + self.Style.BRIGHT if total_unrealized >= 0 else self.Fore.RED + self.Style.BRIGHT
        self.print_color(f"📊 Active Positions: {active_count}/{self.max_concurrent_trades} | Total Unrealized P&L: ${total_unrealized:.2f}", total_color)

def show_trade_history(self, limit=10):
    if not self.real_trade_history:
        self.print_color("No trade history found", self.Fore.YELLOW)
        return
    
    self.print_color(f"\n📊 TRADING HISTORY (Last {min(limit, len(self.real_trade_history))} trades)", self.Fore.CYAN + self.Style.BRIGHT)
    self.print_color("=" * 120, self.Fore.CYAN)
    
    recent_trades = self.real_trade_history[-limit:]
    for i, trade in enumerate(reversed(recent_trades)):
        pnl = trade.get('pnl', 0)
        pnl_color = self.Fore.GREEN + self.Style.BRIGHT if pnl > 0 else self.Fore.RED + self.Style.BRIGHT if pnl < 0 else self.Fore.YELLOW
        direction_icon = "🟢 LONG" if trade['direction'] == 'LONG' else "🔴 SHORT"
        position_size = trade.get('position_size_usd', 0)
        leverage = trade.get('leverage', 1)
        
        display_type = trade.get('display_type', 'FULL_CLOSE')
        if display_type.startswith('PARTIAL'):
            type_indicator = f" | {display_type}"
        else:
            type_indicator = " | FULL"
        
        self.print_color(f"{i+1:2d}. {direction_icon} {trade['pair']}{type_indicator}", pnl_color)
        self.print_color(f"     Size: ${position_size:.2f} | Leverage: {leverage}x | P&L: ${pnl:.2f}", pnl_color)
        self.print_color(f"     Entry: ${trade.get('entry_price', 0):.4f} | Exit: ${trade.get('exit_price', 0):.4f} | {trade.get('close_reason', 'N/A')}", self.Fore.YELLOW)
        
        if trade.get('partial_percent', 100) < 100:
            closed_qty = trade.get('closed_quantity', 0)
            self.print_color(f"     🔸 Partial: {trade['partial_percent']}% ({closed_qty:.4f}) closed", self.Fore.CYAN)

def show_trading_stats(self):
    if self.real_total_trades == 0:
        return
        
    win_rate = (self.real_winning_trades / self.real_total_trades) * 100
    avg_trade = self.real_total_pnl / self.real_total_trades
    
    self.print_color(f"\n📈 TRADING STATISTICS", self.Fore.GREEN + self.Style.BRIGHT)
    self.print_color("=" * 60, self.Fore.GREEN)
    self.print_color(f"Total Trades: {self.real_total_trades} | Winning Trades: {self.real_winning_trades}", self.Fore.WHITE)
    self.print_color(f"Win Rate: {win_rate:.1f}%", self.Fore.GREEN + self.Style.BRIGHT if win_rate > 50 else self.Fore.YELLOW)
    self.print_color(f"Total P&L: ${self.real_total_pnl:.2f}", self.Fore.GREEN + self.Style.BRIGHT if self.real_total_pnl > 0 else self.Fore.RED + self.Style.BRIGHT)
    self.print_color(f"Average P&L per Trade: ${avg_trade:.2f}", self.Fore.WHITE)
    self.print_color(f"Available Budget: ${self.available_budget:.2f}", self.Fore.CYAN + self.Style.BRIGHT)

def run_trading_cycle(self):
    try:
        self.monitor_positions()
        self.display_dashboard()
        
        if hasattr(self, 'cycle_count') and self.cycle_count % 4 == 0:
            self.show_trade_history(8)
            self.show_trading_stats()
        
        if hasattr(self, 'cycle_count') and self.cycle_count % 3 == 0 and LEARN_SCRIPT_AVAILABLE:
            if hasattr(self, 'show_advanced_learning_progress'):
                self.show_advanced_learning_progress()
        
        self.print_color(f"\n🔍 FREEDOM AI SCANNING {len(self.available_pairs)} PAIRS...", self.Fore.BLUE + self.Style.BRIGHT)
        
        qualified_signals = 0
        for pair in self.available_pairs:
            if self.available_budget > 50:  # Lower minimum budget requirement
                market_data = self.get_price_history(pair)
                self.last_mtf = market_data.get('mtf_analysis', {})
                
                ai_decision = self.get_ai_trading_decision(pair, market_data)
                
                if ai_decision["decision"] != "HOLD" and ai_decision["position_size_usd"] > 0:
                    qualified_signals += 1
                    direction = ai_decision['decision']
                    
                    if direction.startswith('REVERSE_'):
                        self.print_color(f"🔄 REVERSE SIGNAL: {pair} {direction} | Size: ${ai_decision['position_size_usd']:.2f}", self.Fore.YELLOW + self.Style.BRIGHT)
                    else:
                        self.print_color(f"🎯 TRADE SIGNAL: {pair} {direction} | Size: ${ai_decision['position_size_usd']:.2f} | Leverage: {ai_decision['leverage']}x", self.Fore.GREEN + self.Style.BRIGHT)
                    
                    success = self.execute_freedom_trade(pair, ai_decision)
                    if success:
                        time.sleep(1)
            
        if qualified_signals == 0:
            self.print_color("No qualified Freedom AI signals this cycle", self.Fore.YELLOW)
            
    except Exception as e:
        self.print_color(f"Trading cycle error: {e}", self.Fore.RED)

def start_trading(self):
    self.print_color("🚀 STARTING FREEDOM AI TRADER!", self.Fore.CYAN + self.Style.BRIGHT)
    self.print_color("💰 AI MANAGING $500 PORTFOLIO", self.Fore.GREEN + self.Style.BRIGHT)
    self.print_color(f"🎯 AGGRESSIVENESS: {self.ai_aggressiveness}", self.Fore.MAGENTA + self.Style.BRIGHT)
    self.print_color("🔄 REVERSE POSITION: ENABLED", self.Fore.MAGENTA + self.Style.BRIGHT)
    self.print_color("⏰ MONITORING: 2 MINUTE INTERVAL", self.Fore.RED + self.Style.BRIGHT)
    self.print_color(f"📊 TRADING PAIRS: {len(self.available_pairs)}", self.Fore.BLUE + self.Style.BRIGHT)
    self.print_color("🎯 FREEDOM AI: NO RESTRICTIVE RULES", self.Fore.YELLOW + self.Style.BRIGHT)
    if LEARN_SCRIPT_AVAILABLE:
        self.print_color("🧠 SELF-LEARNING AI: ENABLED", self.Fore.MAGENTA + self.Style.BRIGHT)
    
    self.cycle_count = 0
    while True:
        try:
            self.cycle_count += 1
            self.print_color(f"\n🔄 FREEDOM AI CYCLE {self.cycle_count}", self.Fore.CYAN + self.Style.BRIGHT)
            self.print_color("=" * 60, self.Fore.CYAN)
            self.run_trading_cycle()
            self.print_color(f"⏳ Next Freedom AI analysis in 2 minutes...", self.Fore.BLUE)
            time.sleep(self.monitoring_interval)
            
        except KeyboardInterrupt:
            self.print_color(f"\n🛑 TRADING STOPPED", self.Fore.RED + self.Style.BRIGHT)
            self.show_trade_history(15)
            self.show_trading_stats()
            break
        except Exception as e:
            self.print_color(f"Main loop error: {e}", self.Fore.RED)
            time.sleep(self.monitoring_interval)

# Add all methods to the class
methods = [
    set_aggressiveness, get_aggressiveness_prompt, format_market_analysis,
    get_ai_trading_decision, parse_freedom_ai_response, get_freedom_fallback_decision,
    execute_freedom_trade, load_real_trade_history, save_real_trade_history,
    add_trade_to_history, get_thailand_time, print_color, validate_api_keys,
    validate_config, setup_futures, load_symbol_precision, get_current_price,
    calculate_quantity, get_price_history, _get_mtf_data_via_api,
    calculate_ema, calculate_rsi, calculate_volume_spike,
    execute_reverse_position, close_trade_immediately, get_ai_close_decision,
    monitor_positions, display_dashboard, show_trade_history, show_trading_stats,
    run_trading_cycle, start_trading, calculate_current_pnl
]

for method in methods:
    setattr(FreedomAITrader, method.__name__, method)

# Paper trading version
class FreedomAIPaperTrader:
    def __init__(self, real_bot):
        self.real_bot = real_bot
        self.Fore = real_bot.Fore
        self.Back = real_bot.Back
        self.Style = real_bot.Style
        self.COLORAMA_AVAILABLE = real_bot.COLORAMA_AVAILABLE
        
        self.allow_reverse_positions = True
        self.monitoring_interval = 120
        
        self.paper_balance = 500
        self.available_budget = 500
        self.paper_positions = {}
        self.paper_history_file = "freedom_ai_paper_trading_history.json"
        self.paper_history = self.load_paper_history()
        self.available_pairs = ["SOLUSDT", "BTCUSDT", "ETHUSDT", "ADAUSDT", "MATICUSDT"]
        self.max_concurrent_trades = 6
        self.ai_aggressiveness = "BALANCED"
        
        self.real_bot.print_color("🤖 FREEDOM AI PAPER TRADER INITIALIZED!", self.Fore.GREEN + self.Style.BRIGHT)
        self.real_bot.print_color(f"💰 Virtual Budget: ${self.paper_balance}", self.Fore.CYAN + self.Style.BRIGHT)
        self.real_bot.print_color(f"🎯 AGGRESSIVENESS: {self.ai_aggressiveness}", self.Fore.MAGENTA + self.Style.BRIGHT)
    
    # Paper trading methods would be similar to real trading but with virtual execution
    # ... (paper trading methods implementation would go here)

# Main execution
if __name__ == "__main__":
    try:
        bot = FreedomAITrader()
        
        print("\n" + "="*70)
        print("🤖 FREEDOM AI TRADER - COMPLETE TRADING FREEDOM")
        print("="*70)
        print("1. 🎯 REAL TRADING (Live Binance)")
        print("2. 📝 PAPER TRADING (Virtual)")
        print("3. ⚙️  SET AGGRESSIVENESS")
        print("4. ❌ EXIT")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            if bot.binance:
                print(f"\n🚀 STARTING REAL TRADING WITH FREEDOM AI...")
                bot.start_trading()
            else:
                print(f"\n❌ Binance connection failed.")
                
        elif choice == "2":
            print(f"\n📝 Paper trading coming soon...")
            # paper_bot = FreedomAIPaperTrader(bot)
            # paper_bot.start_paper_trading()
            
        elif choice == "3":
            print(f"\n⚙️  SET AGGRESSIVENESS LEVEL:")
            print("1. CONSERVATIVE (Safe, fewer trades)")
            print("2. BALANCED (Recommended)")
            print("3. AGGRESSIVE (More trades, higher risk)") 
            print("4. VIKING (Maximum aggression)")
            
            agg_choice = input("\nSelect aggressiveness (1-4): ").strip()
            aggressiveness_map = {
                "1": "CONSERVATIVE",
                "2": "BALANCED", 
                "3": "AGGRESSIVE",
                "4": "VIKING"
            }
            
            selected = aggressiveness_map.get(agg_choice, "BALANCED")
            bot.set_aggressiveness(selected)
            print(f"\n✅ Aggressiveness set to: {selected}")
            print("Restart the program to begin trading with new settings.")
            
        elif choice == "4":
            print(f"\n👋 Exiting...")
            
        else:
            print(f"\n❌ Invalid choice. Exiting...")
            
    except KeyboardInterrupt:
        print(f"\n🛑 Program stopped by user")
    except Exception as e:
        print(f"\n❌ Main execution error: {e}")
        import traceback
        traceback.print_exc()
