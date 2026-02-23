import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import mplfinance as mpf
import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

st.set_page_config(page_title="Forex Analyzer", layout="wide")
st.title("AI Forex Market Analyzer")
st.markdown("Real-time analysis powered by AI")

ALL_PAIRS = {
    "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD", "USDJPY=X": "USD/JPY",
    "AUDUSD=X": "AUD/USD", "USDCAD=X": "USD/CAD", "NZDUSD=X": "NZD/USD",
    "USDCHF=X": "USD/CHF", "EURGBP=X": "EUR/GBP", "EURJPY=X": "EUR/JPY",
    "GBPJPY=X": "GBP/JPY", "AUDJPY=X": "AUD/JPY", "GBPAUD=X": "GBP/AUD",
    "EURCAD=X": "EUR/CAD", "EURCHF=X": "EUR/CHF", "GBPCAD=X": "GBP/CAD",
    "GBPCHF=X": "GBP/CHF", "AUDCAD=X": "AUD/CAD", "AUDCHF=X": "AUD/CHF",
    "AUDNZD=X": "AUD/NZD", "CADJPY=X": "CAD/JPY", "CHFJPY=X": "CHF/JPY",
    "NZDJPY=X": "NZD/JPY", "XAUUSD=X": "XAU/USD (Gold)", "GC=F": "Gold Futures",
    "SI=F": "Silver", "CL=F": "Crude Oil", "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum", "BNB-USD": "BNB", "SOL-USD": "Solana",
}

TV_SYMBOL_MAP = {
    "EURUSD=X": "FX:EURUSD", "GBPUSD=X": "FX:GBPUSD", "USDJPY=X": "FX:USDJPY",
    "AUDUSD=X": "FX:AUDUSD", "USDCAD=X": "FX:USDCAD", "NZDUSD=X": "FX:NZDUSD",
    "USDCHF=X": "FX:USDCHF", "EURGBP=X": "FX:EURGBP", "EURJPY=X": "FX:EURJPY",
    "GBPJPY=X": "FX:GBPJPY", "AUDJPY=X": "FX:AUDJPY", "GBPAUD=X": "FX:GBPAUD",
    "EURCAD=X": "FX:EURCAD", "EURCHF=X": "FX:EURCHF", "GBPCAD=X": "FX:GBPCAD",
    "GBPCHF=X": "FX:GBPCHF", "AUDCAD=X": "FX:AUDCAD", "AUDCHF=X": "FX:AUDCHF",
    "AUDNZD=X": "FX:AUDNZD", "CADJPY=X": "FX:CADJPY", "CHFJPY=X": "FX:CHFJPY",
    "NZDJPY=X": "FX:NZDJPY", "XAUUSD=X": "TVC:GOLD", "GC=F": "COMEX:GC1!",
    "SI=F": "COMEX:SI1!", "CL=F": "NYMEX:CL1!", "BTC-USD": "BINANCE:BTCUSDT",
    "ETH-USD": "BINANCE:ETHUSDT", "BNB-USD": "BINANCE:BNBUSDT", "SOL-USD": "BINANCE:SOLUSDT",
}

TV_INTERVAL_MAP = {
    "1 Minute": "1", "15 Minutes": "15", "1 Hour": "60",
    "4 Hours": "240", "1 Day": "D", "1 Week": "W", "1 Month": "M",
}

TIMEFRAME_MAP = {
    "1 Minute":  {"interval": "1m",  "periods": ["1d", "5d", "7d"]},
    "15 Minutes": {"interval": "15m", "periods": ["5d", "30d", "60d"]},
    "1 Hour":    {"interval": "1h",  "periods": ["7d", "30d", "60d", "90d"]},
    "4 Hours":   {"interval": "1h",  "periods": ["30d", "60d", "6mo", "1y"]},
    "1 Day":     {"interval": "1d",  "periods": ["1mo", "3mo", "6mo", "1y", "2y"]},
    "1 Week":    {"interval": "1wk", "periods": ["6mo", "1y", "2y", "5y"]},
    "1 Month":   {"interval": "1mo", "periods": ["1y", "2y", "5y", "10y"]},
}

tab1, tab2, tab3 = st.tabs(["Live Analysis", "Smart Money Concepts", "Backtesting"])

def tradingview_widget(symbol, interval, height=520):
    tv_symbol = TV_SYMBOL_MAP.get(symbol, "FX:EURUSD")
    tv_interval = TV_INTERVAL_MAP.get(interval, "D")
    html = f"""
    <div class="tradingview-widget-container" style="height:{height}px; width:100%;">
      <div id="tradingview_chart" style="height:{height}px; width:100%;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "width": "100%", "height": {height},
        "symbol": "{tv_symbol}", "interval": "{tv_interval}",
        "timezone": "Etc/UTC", "theme": "dark", "style": "1",
        "locale": "en", "toolbar_bg": "#131722",
        "enable_publishing": false, "hide_side_toolbar": false,
        "allow_symbol_change": true,
        "studies": ["RSI@tv-basicstudies", "MACD@tv-basicstudies"],
        "container_id": "tradingview_chart",
        "backgroundColor": "#131722"
      }});
      </script>
    </div>
    """
    components.html(html, height=height + 30)

def compute_indicators(df):
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["BB_Middle"] = df["Close"].rolling(20).mean()
    df["BB_Upper"] = df["BB_Middle"] + 2 * df["Close"].rolling(20).std()
    df["BB_Lower"] = df["BB_Middle"] - 2 * df["Close"].rolling(20).std()
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Histogram"] = df["MACD"] - df["Signal"]
    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["Stoch_K"] = 100 * (df["Close"] - low14) / (high14 - low14)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()
    df["TR"] = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = df["TR"].rolling(14).mean()
    return df

def get_entry_signal(rsi, macd, signal, close, ema200):
    buy_signals = 0
    sell_signals = 0
    if rsi < 40: buy_signals += 1
    elif rsi > 60: sell_signals += 1
    if macd > signal: buy_signals += 1
    elif macd < signal: sell_signals += 1
    if close > ema200: buy_signals += 1
    elif close < ema200: sell_signals += 1
    if buy_signals >= 2 and buy_signals > sell_signals: return "BUY", "green"
    elif sell_signals >= 2 and sell_signals > buy_signals: return "SELL", "red"
    else: return "NEUTRAL", "orange"

def get_sl_tp(close, atr, signal, multiplier):
    sl_distance = atr * multiplier
    tp_distance = sl_distance * 2
    if signal == "BUY": return round(close - sl_distance, 5), round(close + tp_distance, 5)
    elif signal == "SELL": return round(close + sl_distance, 5), round(close - tp_distance, 5)
    return None, None

def get_swing_points(df, window=5):
    swings = []
    for i in range(window, len(df) - window):
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]
        if high == df["High"].iloc[i - window:i + window].max():
            swings.append({"type": "swing_high", "price": high, "idx": i})
        if low == df["Low"].iloc[i - window:i + window].min():
            swings.append({"type": "swing_low", "price": low, "idx": i})
    highs = [s for s in swings if s["type"] == "swing_high"]
    lows = [s for s in swings if s["type"] == "swing_low"]
    labeled = []
    for i, h in enumerate(highs):
        h["type"] = "HH" if (i == 0 or h["price"] > highs[i-1]["price"]) else "LH"
        labeled.append(h)
    for i, l in enumerate(lows):
        l["type"] = "LL" if (i == 0 or l["price"] < lows[i-1]["price"]) else "HL"
        labeled.append(l)
    labeled.sort(key=lambda x: x["idx"])
    return labeled[-16:]

def get_order_blocks(df):
    obs = []
    for i in range(3, len(df) - 1):
        curr = df.iloc[i]
        nxt = df.iloc[i + 1]
        if (curr["Close"] < curr["Open"] and nxt["Close"] > nxt["Open"] and nxt["Close"] > curr["High"]):
            obs.append({"type": "bullish", "top": curr["Open"], "bottom": curr["Close"], "idx": i})
        if (curr["Close"] > curr["Open"] and nxt["Close"] < nxt["Open"] and nxt["Close"] < curr["Low"]):
            obs.append({"type": "bearish", "top": curr["Close"], "bottom": curr["Open"], "idx": i})
    return obs[-4:] if len(obs) > 4 else obs

def get_fvgs(df):
    fvgs = []
    for i in range(1, len(df) - 1):
        prev = df.iloc[i - 1]
        nxt = df.iloc[i + 1]
        if nxt["Low"] > prev["High"]:
            gap = nxt["Low"] - prev["High"]
            if gap > df["ATR"].iloc[i] * 0.3:
                fvgs.append({"type": "bullish", "top": nxt["Low"], "bottom": prev["High"], "idx": i})
        if nxt["High"] < prev["Low"]:
            gap = prev["Low"] - nxt["High"]
            if gap > df["ATR"].iloc[i] * 0.3:
                fvgs.append({"type": "bearish", "top": prev["Low"], "bottom": nxt["High"], "idx": i})
    recent = [f for f in fvgs if f["idx"] > len(df) - 80]
    return recent[-4:] if len(recent) > 4 else recent

def get_displacement(df):
    disps = []
    atr = df["ATR"].mean()
    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i - 1]
        body = abs(curr["Close"] - curr["Open"])
        move = abs(curr["Close"] - prev["Close"])
        if body > atr * 2.0 and move > atr * 1.5:
            direction = "bullish" if curr["Close"] > curr["Open"] else "bearish"
            disps.append({"type": direction, "open": curr["Open"], "close": curr["Close"],
                          "high": curr["High"], "low": curr["Low"], "idx": i})
    recent = [d for d in disps if d["idx"] > len(df) - 80]
    return recent[-4:] if len(recent) > 4 else recent

def get_liquidity_sweeps(df, window=10):
    sweeps = []
    for i in range(window, len(df)):
        curr = df.iloc[i]
        lb = df.iloc[i - window:i]
        prev_high = lb["High"].max()
        prev_low = lb["Low"].min()
        if curr["High"] > prev_high and curr["Close"] < prev_high:
            sweeps.append({"type": "buyside", "price": curr["High"], "idx": i})
        if curr["Low"] < prev_low and curr["Close"] > prev_low:
            sweeps.append({"type": "sellside", "price": curr["Low"], "idx": i})
    recent = [s for s in sweeps if s["idx"] > len(df) - 80]
    return recent[-4:] if len(recent) > 4 else recent

def get_support_resistance(df, window=10):
    levels = []
    for i in range(window, len(df) - window):
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]
        if high == df["High"].iloc[i - window:i + window].max():
            levels.append(("resistance", high, i))
        if low == df["Low"].iloc[i - window:i + window].min():
            levels.append(("support", low, i))
    filtered = []
    for level in levels:
        too_close = False
        for existing in filtered:
            if abs(level[1] - existing[1]) < df["ATR"].iloc[-1] * 0.8:
                too_close = True
                break
        if not too_close:
            filtered.append(level)
    return filtered[-8:]

def get_bos_choch(df):
    events = []
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    for i in range(3, len(df) - 1):
        if closes[i] > highs[i-1] and closes[i-1] < highs[i-2]:
            events.append({"type": "BOS", "direction": "bull", "price": closes[i], "idx": i})
        if closes[i] < lows[i-1] and closes[i-1] > lows[i-2]:
            events.append({"type": "BOS", "direction": "bear", "price": closes[i], "idx": i})
        if (closes[i-2] < closes[i-3] and closes[i-1] < closes[i-2] and
                closes[i] > closes[i-1] and closes[i] > highs[i-2]):
            events.append({"type": "CHoCH", "direction": "bull", "price": closes[i], "idx": i})
        if (closes[i-2] > closes[i-3] and closes[i-1] > closes[i-2] and
                closes[i] < closes[i-1] and closes[i] < lows[i-2]):
            events.append({"type": "CHoCH", "direction": "bear", "price": closes[i], "idx": i})
    recent = [e for e in events if e["idx"] > len(df) - 80]
    return recent[-6:] if len(recent) > 6 else recent

def draw_smc_chart(df_plot, symbol_name, timeframe, order_blocks, fvgs, displacements,
                   liquidity_sweeps, swing_points, sr_levels, bos_choch):
    n = len(df_plot)
    fig = plt.figure(figsize=(22, 14), facecolor="#131722")
    ax = fig.add_axes([0.04, 0.12, 0.90, 0.80], facecolor="#131722")

    # CANDLES
    for i in range(n):
        o = df_plot["Open"].iloc[i]
        h = df_plot["High"].iloc[i]
        l = df_plot["Low"].iloc[i]
        c = df_plot["Close"].iloc[i]
        color = "#26a69a" if c >= o else "#ef5350"
        ax.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=2)
        body_bottom = min(o, c)
        body_height = max(abs(c - o), df_plot["ATR"].iloc[-1] * 0.01)
        rect = mpatches.Rectangle((i - 0.35, body_bottom), 0.7, body_height, color=color, zorder=3)
        ax.add_patch(rect)

    atr_val = float(df_plot["ATR"].iloc[-1])

    # ORDER BLOCKS
    for ob in order_blocks:
        x = ob["idx"]
        if x >= n: continue
        width = n - x + 2
        height = ob["top"] - ob["bottom"]
        color = "#26a69a" if ob["type"] == "bullish" else "#ef5350"
        label = "Bull OB" if ob["type"] == "bullish" else "Bear OB"
        rect = mpatches.FancyBboxPatch((x - 0.5, ob["bottom"]), width, height,
            boxstyle="round,pad=0", linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.18, zorder=1)
        ax.add_patch(rect)
        ax.plot([x - 0.5, x - 0.5 + width], [ob["top"], ob["top"]], color=color, linewidth=0.9, linestyle="--", alpha=0.8)
        ax.plot([x - 0.5, x - 0.5 + width], [ob["bottom"], ob["bottom"]], color=color, linewidth=0.9, linestyle="--", alpha=0.8)
        ax.text(x + 0.5, ob["top"] + height * 0.1, label, fontsize=8, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#131722", edgecolor=color, alpha=0.9))

    # FAIR VALUE GAPS
    for fvg in fvgs:
        x = fvg["idx"]
        if x >= n: continue
        width = n - x + 2
        height = fvg["top"] - fvg["bottom"]
        color = "#2196F3" if fvg["type"] == "bullish" else "#FF9800"
        label = "Bull FVG" if fvg["type"] == "bullish" else "Bear FVG"
        rect = mpatches.FancyBboxPatch((x - 0.5, fvg["bottom"]), width, height,
            boxstyle="round,pad=0", linewidth=1, edgecolor=color, facecolor=color, alpha=0.15, zorder=1, linestyle="--")
        ax.add_patch(rect)
        ax.plot([x - 0.5, x - 0.5 + width], [fvg["top"], fvg["top"]], color=color, linewidth=0.6, linestyle=":", alpha=0.9)
        ax.plot([x - 0.5, x - 0.5 + width], [fvg["bottom"], fvg["bottom"]], color=color, linewidth=0.6, linestyle=":", alpha=0.9)
        mid = (fvg["top"] + fvg["bottom"]) / 2
        ax.text(x + 0.5, mid, label, fontsize=7.5, color=color, fontstyle="italic",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#131722", edgecolor=color, alpha=0.85))

    # DISPLACEMENT BLOCKS
    for disp in displacements:
        x = disp["idx"]
        if x >= n: continue
        body_bottom = min(disp["open"], disp["close"])
        body_height = abs(disp["close"] - disp["open"])
        color_face = "#B2DFDB" if disp["type"] == "bullish" else "#FFCDD2"
        color_edge = "#26a69a" if disp["type"] == "bullish" else "#ef5350"
        rect = mpatches.FancyBboxPatch((x - 0.42, body_bottom), 0.84, body_height,
            boxstyle="round,pad=0", linewidth=2.5, edgecolor=color_edge, facecolor=color_face, alpha=0.55, zorder=4)
        ax.add_patch(rect)
        label_y = disp["high"] + atr_val * 0.6 if disp["type"] == "bullish" else disp["low"] - atr_val * 0.6
        ax.text(x, label_y, "DISP", fontsize=8, color=color_edge, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#131722", edgecolor=color_edge, alpha=0.95))

    # SUPPORT AND RESISTANCE
    for level_type, level_price, level_idx in sr_levels:
        color = "#26a69a" if level_type == "support" else "#ef5350"
        label = "S" if level_type == "support" else "R"
        ax.axhline(level_price, color=color, linestyle="--", linewidth=1.0, alpha=0.65, zorder=1)
        ax.text(n + 0.5, level_price, label + " " + str(round(level_price, 4)),
                fontsize=7.5, color=color, va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#131722", edgecolor=color, alpha=0.9))

    # SWING POINTS — HH, HL, LH, LL
    swing_colors = {"HH": "#26a69a", "HL": "#80cbc4", "LH": "#ef9a9a", "LL": "#ef5350"}
    swing_marker = {"HH": "^", "HL": "^", "LH": "v", "LL": "v"}
    swing_va = {"HH": "bottom", "HL": "bottom", "LH": "top", "LL": "top"}
    swing_offset = {"HH": 1.8, "HL": 1.8, "LH": -1.8, "LL": -1.8}

    for sp in swing_points:
        x = sp["idx"]
        if x >= n: continue
        stype = sp["type"]
        if stype not in swing_colors: continue
        color = swing_colors[stype]
        offset = atr_val * abs(swing_offset[stype])
        y_label = sp["price"] + offset if swing_offset[stype] > 0 else sp["price"] - offset
        ax.plot(x, sp["price"], swing_marker[stype], color=color, markersize=8, zorder=6, markeredgecolor="#131722", markeredgewidth=0.5)
        ax.text(x, y_label, stype, fontsize=8.5, color=color, fontweight="bold", ha="center", va=swing_va[stype],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#131722", edgecolor=color, alpha=0.92))

    # Connect swing structure
    sorted_swings = sorted(swing_points, key=lambda x: x["idx"])
    valid = [s for s in sorted_swings if s["idx"] < n]
    if len(valid) > 1:
        xs = [s["idx"] for s in valid]
        ys = [s["price"] for s in valid]
        ax.plot(xs, ys, color="#546e7a", linewidth=0.8, linestyle=":", alpha=0.7, zorder=1)

    # LIQUIDITY SWEEPS
    for sweep in liquidity_sweeps:
        x = sweep["idx"]
        if x >= n: continue
        color = "#FFD700"
        label = "BSL ⚡" if sweep["type"] == "buyside" else "SSL ⚡"
        ax.axhline(sweep["price"], color=color, linestyle=":", linewidth=1.0, alpha=0.75)
        ax.plot(x, sweep["price"], "D", color=color, markersize=7, zorder=6)
        ax.text(x, sweep["price"] + atr_val * 0.4, label, fontsize=7.5, color=color, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#131722", edgecolor=color, alpha=0.9))

    # BOS / CHOCH
    for event in bos_choch:
        x = event["idx"]
        if x >= n: continue
        color = "#26a69a" if event["direction"] == "bull" else "#ef5350"
        label = event["type"]
        y = event["price"]
        ax.annotate("", xy=(x + 3, y), xytext=(x - 1, y),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))
        offset = atr_val * 0.5 if event["direction"] == "bull" else -atr_val * 0.5
        ax.text(x + 1, y + offset, label, fontsize=8.5, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#131722", edgecolor=color, alpha=0.92))

    # AXES STYLING
    step = max(1, n // 14)
    xticks = list(range(0, n, step))
    xlabels = [str(df_plot.index[i])[:10] for i in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=30, fontsize=7.5, color="#787b86")
    ax.yaxis.set_tick_params(labelcolor="#787b86", labelsize=8)
    ax.yaxis.tick_right()
    ax.set_xlim(-1, n + 6)
    price_range = df_plot["High"].max() - df_plot["Low"].min()
    ax.set_ylim(df_plot["Low"].min() - price_range * 0.06, df_plot["High"].max() + price_range * 0.10)
    for spine in ax.spines.values():
        spine.set_color("#2a2e39")
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(axis="y", color="#2a2e39", linewidth=0.5, alpha=0.9)
    ax.grid(axis="x", color="#2a2e39", linewidth=0.3, alpha=0.5)

    ax.set_title(symbol_name + "   |   " + timeframe + "   —   Smart Money Concepts",
                 fontsize=13, color="#d1d4dc", fontweight="bold", pad=12, loc="left")

    # LEGEND
    legend_elements = [
        mpatches.Patch(facecolor="#26a69a", alpha=0.4, edgecolor="#26a69a", label="Bull OB"),
        mpatches.Patch(facecolor="#ef5350", alpha=0.4, edgecolor="#ef5350", label="Bear OB"),
        mpatches.Patch(facecolor="#2196F3", alpha=0.3, edgecolor="#2196F3", label="Bull FVG"),
        mpatches.Patch(facecolor="#FF9800", alpha=0.3, edgecolor="#FF9800", label="Bear FVG"),
        mpatches.Patch(facecolor="#B2DFDB", alpha=0.5, edgecolor="#26a69a", label="Bull Disp"),
        mpatches.Patch(facecolor="#FFCDD2", alpha=0.5, edgecolor="#ef5350", label="Bear Disp"),
        Line2D([0], [0], color="#26a69a", linewidth=1.2, linestyle="--", label="Support"),
        Line2D([0], [0], color="#ef5350", linewidth=1.2, linestyle="--", label="Resistance"),
        Line2D([0], [0], marker="^", color="#26a69a", linewidth=0, markersize=8, label="HH / HL"),
        Line2D([0], [0], marker="v", color="#ef5350", linewidth=0, markersize=8, label="LH / LL"),
        Line2D([0], [0], color="#FFD700", linewidth=1.2, linestyle=":", label="Liq Sweep"),
        Line2D([0], [0], color="#26a69a", linewidth=2, label="BOS Bull"),
        Line2D([0], [0], color="#ef5350", linewidth=2, label="BOS/CHoCH Bear"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8,
              facecolor="#1e222d", edgecolor="#2a2e39", labelcolor="#d1d4dc",
              ncol=5, framealpha=0.97, handlelength=1.5)

    plt.tight_layout()
    return fig

# TAB 1 - LIVE ANALYSIS
with tab1:
    pairs = st.multiselect("Select instruments:", list(ALL_PAIRS.keys()), default=["EURUSD=X"], format_func=lambda x: ALL_PAIRS.get(x, x))
    timeframe = st.selectbox("Timeframe:", list(TIMEFRAME_MAP.keys()), index=4, key="tab1_tf")
    tf_config = TIMEFRAME_MAP[timeframe]
    period = st.selectbox("Period:", tf_config["periods"], index=0, key="tab1_period")
    atr_multiplier = st.slider("ATR Multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

    if st.button("Run Analysis"):
        for symbol in pairs:
            with st.spinner("Analyzing " + ALL_PAIRS.get(symbol, symbol) + "..."):
                try:
                    df = yf.download(symbol, period=period, interval=tf_config["interval"], progress=False)
                    if df.empty:
                        st.error("No data for " + symbol)
                        continue
                    df.columns = df.columns.droplevel(1)
                    df = compute_indicators(df)
                    latest = df.tail(1)
                    close = round(float(latest["Close"].iloc[0]), 5)
                    ma20 = round(float(latest["MA20"].iloc[0]), 5)
                    ma50 = round(float(latest["MA50"].iloc[0]), 5)
                    ema200 = round(float(latest["EMA200"].iloc[0]), 5)
                    rsi = round(float(latest["RSI"].iloc[0]), 2)
                    bb_upper = round(float(latest["BB_Upper"].iloc[0]), 5)
                    bb_lower = round(float(latest["BB_Lower"].iloc[0]), 5)
                    macd = round(float(latest["MACD"].iloc[0]), 5)
                    signal = round(float(latest["Signal"].iloc[0]), 5)
                    stoch_k = round(float(latest["Stoch_K"].iloc[0]), 2)
                    stoch_d = round(float(latest["Stoch_D"].iloc[0]), 2)
                    atr = round(float(latest["ATR"].iloc[0]), 5)
                    entry_signal, signal_color = get_entry_signal(rsi, macd, signal, close, ema200)
                    stop_loss, take_profit = get_sl_tp(close, atr, entry_signal, atr_multiplier)
                    sr_levels = get_support_resistance(df)

                    prompt = (
                        "You are a professional forex analyst. Analyze " + ALL_PAIRS.get(symbol, symbol) +
                        " on the " + timeframe + " timeframe:\n"
                        "- Price: " + str(close) + " | RSI: " + str(rsi) + " | ATR: " + str(atr) + "\n"
                        "- EMA200: " + str(ema200) + " | MACD: " + str(macd) + " | Signal: " + str(signal) + "\n"
                        "- Entry: " + entry_signal + " | SL: " + str(stop_loss) + " | TP: " + str(take_profit) + "\n"
                        "5 sentence professional analysis including timeframe context, entry justification, SL/TP reasoning, risks."
                    )
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": prompt}]
                    )

                    st.subheader(ALL_PAIRS.get(symbol, symbol) + " — " + timeframe)
                    st.markdown(
                        "<div style='background-color:" + signal_color + "; padding:10px; border-radius:8px; text-align:center;'>"
                        "<h2 style='color:white; margin:0;'>ENTRY SIGNAL: " + entry_signal + "</h2>"
                        "</div>", unsafe_allow_html=True
                    )
                    st.markdown("")
                    if stop_loss and take_profit:
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("Price", close)
                        col2.metric("Stop Loss", stop_loss, delta=round(stop_loss - close, 5))
                        col3.metric("Take Profit", take_profit, delta=round(take_profit - close, 5))
                        col4.metric("ATR", atr)
                        col5.metric("RSI", rsi)
                    else:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Price", close)
                        col2.metric("ATR", atr)
                        col3.metric("RSI", rsi)
                        st.warning("No clear entry signal.")

                    df_plot = df.tail(80)
                    add_plots = [
                        mpf.make_addplot(df_plot["BB_Upper"], color="red", linestyle="--", width=0.8),
                        mpf.make_addplot(df_plot["BB_Lower"], color="green", linestyle="--", width=0.8),
                        mpf.make_addplot(df_plot["EMA200"], color="purple", width=1),
                        mpf.make_addplot(df_plot["MA20"], color="orange", width=0.8),
                    ]
                    hlines = []
                    hline_colors = []
                    for level_type, level_price, _ in sr_levels:
                        hlines.append(level_price)
                        hline_colors.append("blue" if level_type == "support" else "magenta")
                    if stop_loss: hlines.append(stop_loss); hline_colors.append("red")
                    if take_profit: hlines.append(take_profit); hline_colors.append("lime")
                    fig_candle, _ = mpf.plot(df_plot, type="candle", style="nightclouds",
                        title=ALL_PAIRS.get(symbol, symbol) + " — " + timeframe,
                        addplot=add_plots,
                        hlines=dict(hlines=hlines, colors=hline_colors, linestyle="--", linewidths=0.8),
                        volume=False, returnfig=True, figsize=(14, 6))
                    st.pyplot(fig_candle)

                    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 6))
                    axes2[0].plot(df.index, df["MACD"], label="MACD", color="blue")
                    axes2[0].plot(df.index, df["Signal"], label="Signal", color="orange")
                    axes2[0].bar(df.index, df["Histogram"], color="grey", alpha=0.5)
                    axes2[0].set_title("MACD"); axes2[0].legend(fontsize=8)
                    axes2[1].plot(df.index, df["Stoch_K"], label="Stoch K", color="blue")
                    axes2[1].plot(df.index, df["Stoch_D"], label="Stoch D", color="orange")
                    axes2[1].axhline(80, color="red", linestyle="--", alpha=0.5)
                    axes2[1].axhline(20, color="green", linestyle="--", alpha=0.5)
                    axes2[1].set_title("Stochastic Oscillator"); axes2[1].legend(fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    st.info(response.choices[0].message.content)
                    st.divider()
                except Exception as e:
                    st.error("Could not load data for " + symbol + ". Try different settings.")

# TAB 2 - SMART MONEY CONCEPTS
with tab2:
    st.header("Smart Money Concepts")

    smc_symbol = st.selectbox("Select instrument:", list(ALL_PAIRS.keys()), format_func=lambda x: ALL_PAIRS.get(x, x), key="smc")
    smc_timeframe = st.selectbox("Timeframe:", list(TIMEFRAME_MAP.keys()), index=4, key="smc_tf")
    smc_tf_config = TIMEFRAME_MAP[smc_timeframe]
    smc_period = st.selectbox("Period:", smc_tf_config["periods"], index=0, key="smc_period")

    col_checks = st.columns(5)
    show_ob = col_checks[0].checkbox("Order Blocks", value=True)
    show_fvg = col_checks[1].checkbox("Fair Value Gaps", value=True)
    show_disp = col_checks[2].checkbox("Displacement", value=True)
    show_liq = col_checks[3].checkbox("Liquidity Sweeps", value=True)
    show_bos = col_checks[4].checkbox("BOS / CHoCH", value=True)

    st.subheader("TradingView Live Chart")
    tradingview_widget(smc_symbol, smc_timeframe, height=520)

    if st.button("Run SMC Analysis"):
        with st.spinner("Running SMC Analysis..."):
            df = yf.download(smc_symbol, period=smc_period, interval=smc_tf_config["interval"], progress=False)
            df.columns = df.columns.droplevel(1)
            df = compute_indicators(df)
            df = df.dropna()
            df_plot = df.tail(80).copy()
            df_plot = df_plot.reset_index(drop=False)

            # Compute all on full df, using positional idx within df
            all_obs = get_order_blocks(df) if show_ob else []
            all_fvgs = get_fvgs(df) if show_fvg else []
            all_disps = get_displacement(df) if show_disp else []
            all_sweeps = get_liquidity_sweeps(df) if show_liq else []
            all_swings = get_swing_points(df)
            all_sr = get_support_resistance(df)
            all_bos = get_bos_choch(df) if show_bos else []

            offset = len(df) - 80

            def rebase(items):
                result = []
                for item in items:
                    new_item = dict(item)
                    new_item["idx"] = item["idx"] - offset
                    if new_item["idx"] >= 0:
                        result.append(new_item)
                return result

            obs = rebase(all_obs)
            fvgs = rebase(all_fvgs)
            disps = rebase(all_disps)
            sweeps = rebase(all_sweeps)
            swings = rebase(all_swings)
            bos_events = rebase(all_bos)

            fig_smc = draw_smc_chart(
                df_plot.set_index(df_plot.columns[0]) if not isinstance(df_plot.index, pd.DatetimeIndex) else df_plot,
                ALL_PAIRS.get(smc_symbol, smc_symbol),
                smc_timeframe,
                obs, fvgs, disps, sweeps, swings, all_sr, bos_events
            )
            st.subheader("SMC Analysis Chart")
            st.pyplot(fig_smc)

            # SUMMARY
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("**Order Blocks**")
                for ob in all_obs:
                    emoji = "🟢" if ob["type"] == "bullish" else "🔴"
                    st.write(emoji + " " + ob["type"].upper() + " OB @ " + str(round(ob["top"], 5)))
            with col2:
                st.markdown("**Fair Value Gaps**")
                for fvg in all_fvgs:
                    emoji = "🔵" if fvg["type"] == "bullish" else "🟠"
                    st.write(emoji + " FVG " + str(round(fvg["bottom"], 5)) + " — " + str(round(fvg["top"], 5)))
            with col3:
                st.markdown("**Liquidity Sweeps**")
                for sweep in all_sweeps:
                    label = "BSL" if sweep["type"] == "buyside" else "SSL"
                    st.write("🟡 " + label + " @ " + str(round(sweep["price"], 5)))
            with col4:
                st.markdown("**Displacement**")
                for disp in all_disps:
                    emoji = "🟢" if disp["type"] == "bullish" else "🔴"
                    st.write(emoji + " " + disp["type"].upper() + " @ " + str(round(disp["close"], 5)))

            ob_text = ", ".join([ob["type"] + " OB at " + str(round(ob["top"], 5)) for ob in all_obs]) or "None"
            fvg_text = ", ".join([f["type"] + " FVG " + str(round(f["bottom"], 5)) + "-" + str(round(f["top"], 5)) for f in all_fvgs]) or "None"
            disp_text = ", ".join([d["type"] + " disp at " + str(round(d["close"], 5)) for d in all_disps]) or "None"
            liq_text = ", ".join([s["type"].replace("_", " ") + " at " + str(round(s["price"], 5)) for s in all_sweeps]) or "None"
            bos_text = ", ".join([e["type"] + " " + e["direction"] + " at " + str(round(e["price"], 5)) for e in all_bos[-3:]]) or "None"
            swing_text = ", ".join([s["type"] + " at " + str(round(s["price"], 5)) for s in all_swings[-6:]]) or "None"

            smc_prompt = (
                "You are an expert Smart Money Concepts analyst. Analyze " +
                ALL_PAIRS.get(smc_symbol, smc_symbol) + " on the " + smc_timeframe + " timeframe:\n"
                "- Current Price: " + str(round(float(df["Close"].iloc[-1]), 5)) + "\n"
                "- Order Blocks: " + ob_text + "\n"
                "- Fair Value Gaps: " + fvg_text + "\n"
                "- Displacement: " + disp_text + "\n"
                "- Liquidity Sweeps: " + liq_text + "\n"
                "- BOS/CHoCH: " + bos_text + "\n"
                "- Swing Structure (HH/HL/LH/LL): " + swing_text + "\n"
                "Give a detailed 6-7 sentence SMC analysis covering: "
                "1) Market structure from HH/HL/LH/LL, "
                "2) What liquidity sweeps reveal about stop hunting, "
                "3) Whether displacement confirms institutional intent, "
                "4) Key order blocks for the next move, "
                "5) FVG fill probability and direction, "
                "6) Overall SMC bias and most likely next move."
            )
            smc_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": smc_prompt}]
            )
            st.subheader("AI SMC Analysis")
            st.info(smc_response.choices[0].message.content)

# TAB 3 - BACKTESTING
with tab3:
    st.header("Backtest Your Strategy")
    bt_symbol = st.selectbox("Select instrument:", list(ALL_PAIRS.keys()), format_func=lambda x: ALL_PAIRS.get(x, x), key="bt_symbol")
    bt_timeframe = st.selectbox("Timeframe:", list(TIMEFRAME_MAP.keys()), index=4, key="bt_tf")
    bt_tf_config = TIMEFRAME_MAP[bt_timeframe]
    bt_period = st.selectbox("Period:", bt_tf_config["periods"], index=0, key="bt_period")
    bt_multiplier = st.slider("ATR Multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1, key="bt")

    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            df = yf.download(bt_symbol, period=bt_period, interval=bt_tf_config["interval"], progress=False)
            df.columns = df.columns.droplevel(1)
            df = compute_indicators(df)
            df = df.dropna()
            trades = []
            in_trade = False
            entry_price = 0
            trade_type = ""
            sl = 0
            tp = 0
            for i in range(len(df)):
                row = df.iloc[i]
                if not in_trade:
                    try:
                        sig, _ = get_entry_signal(float(row["RSI"]), float(row["MACD"]),
                                                   float(row["Signal"]), float(row["Close"]), float(row["EMA200"]))
                    except: continue
                    if sig in ["BUY", "SELL"]:
                        entry_price = float(row["Close"])
                        atr_val = float(row["ATR"])
                        sl_dist = atr_val * bt_multiplier
                        tp_dist = sl_dist * 2
                        if sig == "BUY": sl = entry_price - sl_dist; tp = entry_price + tp_dist
                        else: sl = entry_price + sl_dist; tp = entry_price - tp_dist
                        trade_type = sig; in_trade = True; entry_date = df.index[i]
                else:
                    current_price = float(row["Close"])
                    if trade_type == "BUY":
                        if current_price <= sl:
                            trades.append({"date": entry_date, "type": trade_type, "entry": entry_price, "exit": sl, "pnl": sl - entry_price, "result": "LOSS"}); in_trade = False
                        elif current_price >= tp:
                            trades.append({"date": entry_date, "type": trade_type, "entry": entry_price, "exit": tp, "pnl": tp - entry_price, "result": "WIN"}); in_trade = False
                    elif trade_type == "SELL":
                        if current_price >= sl:
                            trades.append({"date": entry_date, "type": trade_type, "entry": entry_price, "exit": sl, "pnl": entry_price - sl, "result": "LOSS"}); in_trade = False
                        elif current_price <= tp:
                            trades.append({"date": entry_date, "type": trade_type, "entry": entry_price, "exit": tp, "pnl": entry_price - tp, "result": "WIN"}); in_trade = False
            if len(trades) == 0:
                st.warning("No completed trades found. Try a longer period.")
            else:
                trades_df = pd.DataFrame(trades)
                total_trades = len(trades_df)
                wins = len(trades_df[trades_df["result"] == "WIN"])
                losses = len(trades_df[trades_df["result"] == "LOSS"])
                win_rate = round((wins / total_trades) * 100, 2)
                total_pnl = round(trades_df["pnl"].sum(), 5)
                st.subheader("Results — " + ALL_PAIRS.get(bt_symbol, bt_symbol) + " — " + bt_timeframe)
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Total Trades", total_trades); col2.metric("Wins", wins)
                col3.metric("Losses", losses); col4.metric("Win Rate", str(win_rate) + "%"); col5.metric("Total PnL", total_pnl)
                trades_df["cumulative_pnl"] = trades_df["pnl"].cumsum()
                fig, axes = plt.subplots(2, 1, figsize=(14, 8))
                colors = ["green" if r == "WIN" else "red" for r in trades_df["result"]]
                axes[0].bar(range(len(trades_df)), trades_df["pnl"], color=colors)
                axes[0].axhline(0, color="white", linewidth=0.8); axes[0].set_title("Individual Trade PnL")
                axes[1].plot(range(len(trades_df)), trades_df["cumulative_pnl"], color="cyan", linewidth=2)
                axes[1].fill_between(range(len(trades_df)), trades_df["cumulative_pnl"], alpha=0.2, color="cyan")
                axes[1].axhline(0, color="white", linewidth=0.8); axes[1].set_title("Cumulative PnL Over Time")
                plt.tight_layout(); st.pyplot(fig)
                st.subheader("Trade Log"); st.dataframe(trades_df)
