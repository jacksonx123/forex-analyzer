import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

def get_support_resistance(df, window=10):
    levels = []
    for i in range(window, len(df) - window):
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]
        if high == df["High"].iloc[i - window:i + window].max():
            levels.append(("resistance", high, df.index[i]))
        if low == df["Low"].iloc[i - window:i + window].min():
            levels.append(("support", low, df.index[i]))
    filtered = []
    for level in levels:
        too_close = False
        for existing in filtered:
            if abs(level[1] - existing[1]) < df["ATR"].iloc[-1] * 0.5:
                too_close = True
                break
        if not too_close:
            filtered.append(level)
    return filtered

def get_order_blocks(df, lookback=5):
    order_blocks = []
    for i in range(lookback, len(df) - 1):
        current = df.iloc[i]
        next_candle = df.iloc[i + 1]
        if (current["Close"] < current["Open"] and
                next_candle["Close"] > next_candle["Open"] and
                next_candle["Close"] > current["High"]):
            order_blocks.append({"type": "bullish", "top": current["Open"], "bottom": current["Close"], "date": df.index[i]})
        if (current["Close"] > current["Open"] and
                next_candle["Close"] < next_candle["Open"] and
                next_candle["Close"] < current["Low"]):
            order_blocks.append({"type": "bearish", "top": current["Close"], "bottom": current["Open"], "date": df.index[i]})
    return order_blocks[-5:] if len(order_blocks) > 5 else order_blocks

def get_fair_value_gaps(df):
    fvgs = []
    for i in range(1, len(df) - 1):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        nxt = df.iloc[i + 1]
        # Bullish FVG: gap between prev candle high and next candle low
        if nxt["Low"] > prev["High"]:
            fvgs.append({
                "type": "bullish",
                "top": nxt["Low"],
                "bottom": prev["High"],
                "date": df.index[i],
                "filled": df["Close"].iloc[i + 1:].min() <= prev["High"] if i + 1 < len(df) else False
            })
        # Bearish FVG: gap between next candle high and prev candle low
        if nxt["High"] < prev["Low"]:
            fvgs.append({
                "type": "bearish",
                "top": prev["Low"],
                "bottom": nxt["High"],
                "date": df.index[i],
                "filled": df["Close"].iloc[i + 1:].max() >= prev["Low"] if i + 1 < len(df) else False
            })
    # Return only recent unfilled FVGs
    unfilled = [f for f in fvgs if not f["filled"]]
    return unfilled[-5:] if len(unfilled) > 5 else unfilled

def get_displacement(df):
    displacements = []
    atr = df["ATR"].mean()
    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i - 1]
        body_size = abs(curr["Close"] - curr["Open"])
        move = abs(curr["Close"] - prev["Close"])
        # Displacement = large candle body (2x ATR) with strong directional move
        if body_size > atr * 2 and move > atr * 1.5:
            direction = "bullish" if curr["Close"] > curr["Open"] else "bearish"
            displacements.append({
                "type": direction,
                "price": curr["Close"],
                "high": curr["High"],
                "low": curr["Low"],
                "date": df.index[i]
            })
    return displacements[-5:] if len(displacements) > 5 else displacements

def get_liquidity_sweeps(df, window=10):
    sweeps = []
    for i in range(window, len(df)):
        curr = df.iloc[i]
        lookback = df.iloc[i - window:i]
        prev_high = lookback["High"].max()
        prev_low = lookback["Low"].min()
        # Liquidity sweep above highs (buyside liquidity taken)
        if curr["High"] > prev_high and curr["Close"] < prev_high:
            sweeps.append({
                "type": "buyside_sweep",
                "price": curr["High"],
                "date": df.index[i]
            })
        # Liquidity sweep below lows (sellside liquidity taken)
        if curr["Low"] < prev_low and curr["Close"] > prev_low:
            sweeps.append({
                "type": "sellside_sweep",
                "price": curr["Low"],
                "date": df.index[i]
            })
    return sweeps[-6:] if len(sweeps) > 6 else sweeps

def get_bos_choch(df):
    events = []
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    dates = df.index
    for i in range(3, len(df) - 1):
        if closes[i] > highs[i - 1] and closes[i - 1] < highs[i - 2]:
            events.append({"type": "BOS_BULL", "price": closes[i], "date": dates[i]})
        if closes[i] < lows[i - 1] and closes[i - 1] > lows[i - 2]:
            events.append({"type": "BOS_BEAR", "price": closes[i], "date": dates[i]})
        if (closes[i - 2] < closes[i - 3] and closes[i - 1] < closes[i - 2] and
                closes[i] > closes[i - 1] and closes[i] > highs[i - 2]):
            events.append({"type": "CHOCH_BULL", "price": closes[i], "date": dates[i]})
        if (closes[i - 2] > closes[i - 3] and closes[i - 1] > closes[i - 2] and
                closes[i] < closes[i - 1] and closes[i] < lows[i - 2]):
            events.append({"type": "CHOCH_BEAR", "price": closes[i], "date": dates[i]})
    return events[-10:] if len(events) > 10 else events

# TAB 1 - LIVE ANALYSIS
with tab1:
    pairs = st.multiselect(
        "Select instruments to analyze:",
        list(ALL_PAIRS.keys()),
        default=["EURUSD=X"],
        format_func=lambda x: ALL_PAIRS.get(x, x)
    )
    timeframe = st.selectbox("Select timeframe:", list(TIMEFRAME_MAP.keys()), index=4)
    tf_config = TIMEFRAME_MAP[timeframe]
    period = st.selectbox("Select period:", tf_config["periods"], index=0)
    atr_multiplier = st.slider("ATR Multiplier for Stop Loss", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
    if timeframe in ["1 Minute", "15 Minutes"]:
        st.warning("Short timeframes only support limited history.")

    if st.button("Run Analysis"):
        for symbol in pairs:
            with st.spinner("Analyzing " + ALL_PAIRS.get(symbol, symbol) + " on " + timeframe + "..."):
                try:
                    df = yf.download(symbol, period=period, interval=tf_config["interval"], progress=False)
                    if df.empty:
                        st.error("No data for " + symbol + ". Try different timeframe/period.")
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
                        "- Price: " + str(close) + " | MA20: " + str(ma20) + " | MA50: " + str(ma50) + "\n"
                        "- EMA200: " + str(ema200) + " | RSI: " + str(rsi) + " | ATR: " + str(atr) + "\n"
                        "- BB Upper: " + str(bb_upper) + " | BB Lower: " + str(bb_lower) + "\n"
                        "- MACD: " + str(macd) + " | Signal: " + str(signal) + "\n"
                        "- Stoch K: " + str(stoch_k) + " | Stoch D: " + str(stoch_d) + "\n"
                        "- Entry Signal: " + entry_signal + " | SL: " + str(stop_loss) + " | TP: " + str(take_profit) + "\n"
                        "5 sentence professional analysis. Include timeframe context, entry justification, SL/TP reasoning, and risks."
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
                    if stop_loss:
                        hlines.append(stop_loss)
                        hline_colors.append("red")
                    if take_profit:
                        hlines.append(take_profit)
                        hline_colors.append("lime")
                    fig_candle, _ = mpf.plot(
                        df_plot, type="candle", style="nightclouds",
                        title=ALL_PAIRS.get(symbol, symbol) + " — " + timeframe,
                        addplot=add_plots,
                        hlines=dict(hlines=hlines, colors=hline_colors, linestyle="--", linewidths=0.8),
                        volume=False, returnfig=True, figsize=(14, 6)
                    )
                    st.pyplot(fig_candle)
                    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 6))
                    axes2[0].plot(df.index, df["MACD"], label="MACD", color="blue")
                    axes2[0].plot(df.index, df["Signal"], label="Signal", color="orange")
                    axes2[0].bar(df.index, df["Histogram"], color="grey", alpha=0.5)
                    axes2[0].set_title("MACD")
                    axes2[0].legend(fontsize=8)
                    axes2[1].plot(df.index, df["Stoch_K"], label="Stoch K", color="blue")
                    axes2[1].plot(df.index, df["Stoch_D"], label="Stoch D", color="orange")
                    axes2[1].axhline(80, color="red", linestyle="--", alpha=0.5)
                    axes2[1].axhline(20, color="green", linestyle="--", alpha=0.5)
                    axes2[1].set_title("Stochastic Oscillator")
                    axes2[1].legend(fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    st.info(response.choices[0].message.content)
                    st.divider()
                except Exception as e:
                    st.error("Could not load data for " + symbol + ". Try different settings.")

# TAB 2 - SMART MONEY CONCEPTS
with tab2:
    st.header("Smart Money Concepts")
    st.markdown("Order Blocks | Fair Value Gaps | Displacement | Liquidity Sweeps | BOS | CHoCH")

    smc_symbol = st.selectbox("Select instrument:", list(ALL_PAIRS.keys()), format_func=lambda x: ALL_PAIRS.get(x, x), key="smc")
    smc_timeframe = st.selectbox("Select timeframe:", list(TIMEFRAME_MAP.keys()), index=4, key="smc_tf")
    smc_tf_config = TIMEFRAME_MAP[smc_timeframe]
    smc_period = st.selectbox("Select period:", smc_tf_config["periods"], index=0, key="smc_period")

    show_ob = st.checkbox("Show Order Blocks", value=True)
    show_fvg = st.checkbox("Show Fair Value Gaps", value=True)
    show_disp = st.checkbox("Show Displacement", value=True)
    show_liq = st.checkbox("Show Liquidity Sweeps", value=True)
    show_bos = st.checkbox("Show BOS / CHoCH", value=True)

    if st.button("Run SMC Analysis"):
        with st.spinner("Analyzing Smart Money Concepts..."):
            df = yf.download(smc_symbol, period=smc_period, interval=smc_tf_config["interval"], progress=False)
            df.columns = df.columns.droplevel(1)
            df = compute_indicators(df)
            df = df.dropna()

            order_blocks = get_order_blocks(df) if show_ob else []
            fvgs = get_fair_value_gaps(df) if show_fvg else []
            displacements = get_displacement(df) if show_disp else []
            liquidity_sweeps = get_liquidity_sweeps(df) if show_liq else []
            bos_choch = get_bos_choch(df) if show_bos else []
            sr_levels = get_support_resistance(df)

            df_plot = df.tail(80)

            fig_smc, ax = mpf.plot(
                df_plot, type="candle", style="nightclouds",
                title=ALL_PAIRS.get(smc_symbol, smc_symbol) + " — " + smc_timeframe + " SMC",
                volume=False, returnfig=True, figsize=(16, 10)
            )
            ax_main = fig_smc.axes[0]

            # ORDER BLOCKS
            for ob in order_blocks:
                if ob["date"] in df_plot.index:
                    idx = df_plot.index.get_loc(ob["date"])
                    color = "lime" if ob["type"] == "bullish" else "tomato"
                    rect = mpatches.FancyBboxPatch(
                        (idx - 0.5, ob["bottom"]),
                        len(df_plot) - idx,
                        ob["top"] - ob["bottom"],
                        boxstyle="round,pad=0",
                        linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.2
                    )
                    ax_main.add_patch(rect)
                    ax_main.text(idx, ob["top"], ob["type"][0].upper() + " OB", fontsize=6, color=color, fontweight="bold")

            # FAIR VALUE GAPS
            for fvg in fvgs:
                if fvg["date"] in df_plot.index:
                    idx = df_plot.index.get_loc(fvg["date"])
                    color = "deepskyblue" if fvg["type"] == "bullish" else "orange"
                    rect = mpatches.FancyBboxPatch(
                        (idx - 0.5, fvg["bottom"]),
                        len(df_plot) - idx,
                        fvg["top"] - fvg["bottom"],
                        boxstyle="round,pad=0",
                        linewidth=1, edgecolor=color, facecolor=color, alpha=0.15,
                        linestyle="dashed"
                    )
                    ax_main.add_patch(rect)
                    ax_main.text(idx, fvg["top"], fvg["type"][0].upper() + " FVG", fontsize=6, color=color)

            # DISPLACEMENT
            for disp in displacements:
                if disp["date"] in df_plot.index:
                    idx = df_plot.index.get_loc(disp["date"])
                    color = "lime" if disp["type"] == "bullish" else "red"
                    ax_main.annotate(
                        "DISP",
                        xy=(idx, disp["price"]),
                        fontsize=7, color=color, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.3)
                    )

            # LIQUIDITY SWEEPS
            for sweep in liquidity_sweeps:
                if sweep["date"] in df_plot.index:
                    idx = df_plot.index.get_loc(sweep["date"])
                    color = "yellow"
                    marker = "v" if sweep["type"] == "buyside_sweep" else "^"
                    label = "BSL Swept" if sweep["type"] == "buyside_sweep" else "SSL Swept"
                    ax_main.annotate(
                        label,
                        xy=(idx, sweep["price"]),
                        fontsize=6, color=color, fontweight="bold",
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.8)
                    )
                    ax_main.axhline(sweep["price"], color=color, linestyle=":", linewidth=0.8, alpha=0.6)

            # BOS / CHOCH
            for event in bos_choch:
                if event["date"] in df_plot.index:
                    idx = df_plot.index.get_loc(event["date"])
                    color = "lime" if "BULL" in event["type"] else "red"
                    label = event["type"].replace("_BULL", "").replace("_BEAR", "")
                    ax_main.annotate(label, xy=(idx, event["price"]), fontsize=7, color=color, fontweight="bold")

            # SUPPORT / RESISTANCE
            for level_type, level_price, _ in sr_levels:
                color = "cyan" if level_type == "support" else "magenta"
                ax_main.axhline(level_price, color=color, linestyle="--", linewidth=0.8, alpha=0.6)

            st.pyplot(fig_smc)

            # LEGEND
            st.markdown("""
            **Chart Legend:**
            🟢 **Green zones** = Bullish Order Blocks &nbsp;|&nbsp;
            🔴 **Red zones** = Bearish Order Blocks &nbsp;|&nbsp;
            🔵 **Blue dashed zones** = Bullish FVG &nbsp;|&nbsp;
            🟠 **Orange dashed zones** = Bearish FVG &nbsp;|&nbsp;
            🟡 **Yellow lines** = Liquidity Sweeps (BSL/SSL) &nbsp;|&nbsp;
            **DISP** = Displacement candles &nbsp;|&nbsp;
            **BOS** = Break of Structure &nbsp;|&nbsp;
            **CHoCH** = Change of Character
            """)

            # SUMMARY COLUMNS
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown("**Order Blocks**")
                for ob in order_blocks:
                    emoji = "🟢" if ob["type"] == "bullish" else "🔴"
                    st.write(emoji + " " + ob["type"].upper() + " OB @ " + str(round(ob["top"], 5)))

            with col2:
                st.markdown("**Fair Value Gaps**")
                for fvg in fvgs:
                    emoji = "🔵" if fvg["type"] == "bullish" else "🟠"
                    st.write(emoji + " " + fvg["type"].upper() + " FVG " + str(round(fvg["bottom"], 5)) + " — " + str(round(fvg["top"], 5)))

            with col3:
                st.markdown("**Liquidity Sweeps**")
                for sweep in liquidity_sweeps:
                    emoji = "🟡"
                    label = "BSL" if sweep["type"] == "buyside_sweep" else "SSL"
                    st.write(emoji + " " + label + " swept @ " + str(round(sweep["price"], 5)))

            with col4:
                st.markdown("**Displacement**")
                for disp in displacements:
                    emoji = "🟢" if disp["type"] == "bullish" else "🔴"
                    st.write(emoji + " " + disp["type"].upper() + " DISP @ " + str(round(disp["price"], 5)))

            # AI SMC ANALYSIS
            ob_text = ", ".join([ob["type"] + " OB at " + str(round(ob["top"], 5)) for ob in order_blocks]) or "None detected"
            fvg_text = ", ".join([f["type"] + " FVG " + str(round(f["bottom"], 5)) + "-" + str(round(f["top"], 5)) for f in fvgs]) or "None detected"
            disp_text = ", ".join([d["type"] + " displacement at " + str(round(d["price"], 5)) for d in displacements]) or "None detected"
            liq_text = ", ".join([s["type"].replace("_", " ") + " at " + str(round(s["price"], 5)) for s in liquidity_sweeps]) or "None detected"
            bos_text = ", ".join([e["type"] + " at " + str(round(e["price"], 5)) for e in bos_choch[-3:]]) or "None detected"

            smc_prompt = (
                "You are an expert Smart Money Concepts forex analyst. Analyze " +
                ALL_PAIRS.get(smc_symbol, smc_symbol) + " on the " + smc_timeframe + " timeframe:\n"
                "- Current Price: " + str(round(float(df["Close"].iloc[-1]), 5)) + "\n"
                "- Order Blocks: " + ob_text + "\n"
                "- Fair Value Gaps: " + fvg_text + "\n"
                "- Displacement: " + disp_text + "\n"
                "- Liquidity Sweeps: " + liq_text + "\n"
                "- BOS/CHoCH: " + bos_text + "\n"
                "Give a detailed 6-7 sentence SMC analysis. Explain: "
                "1) What the liquidity sweeps tell us about where price has been hunting stops, "
                "2) Whether displacement confirms institutional intent, "
                "3) Which order blocks are most significant for the next move, "
                "4) Whether FVGs need to be filled before continuation, "
                "5) The overall SMC bias and most likely next move."
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
    bt_symbol = st.selectbox("Select instrument:", list(ALL_PAIRS.keys()), format_func=lambda x: ALL_PAIRS.get(x, x))
    bt_timeframe = st.selectbox("Select timeframe:", list(TIMEFRAME_MAP.keys()), index=4, key="bt_tf")
    bt_tf_config = TIMEFRAME_MAP[bt_timeframe]
    bt_period = st.selectbox("Select backtest period:", bt_tf_config["periods"], index=0)
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
                        sig, _ = get_entry_signal(
                            float(row["RSI"]), float(row["MACD"]),
                            float(row["Signal"]), float(row["Close"]), float(row["EMA200"])
                        )
                    except:
                        continue
                    if sig in ["BUY", "SELL"]:
                        entry_price = float(row["Close"])
                        atr_val = float(row["ATR"])
                        sl_dist = atr_val * bt_multiplier
                        tp_dist = sl_dist * 2
                        if sig == "BUY":
                            sl = entry_price - sl_dist
                            tp = entry_price + tp_dist
                        else:
                            sl = entry_price + sl_dist
                            tp = entry_price - tp_dist
                        trade_type = sig
                        in_trade = True
                        entry_date = df.index[i]
                else:
                    current_price = float(row["Close"])
                    if trade_type == "BUY":
                        if current_price <= sl:
                            trades.append({"date": entry_date, "type": trade_type, "entry": entry_price, "exit": sl, "pnl": sl - entry_price, "result": "LOSS"})
                            in_trade = False
                        elif current_price >= tp:
                            trades.append({"date": entry_date, "type": trade_type, "entry": entry_price, "exit": tp, "pnl": tp - entry_price, "result": "WIN"})
                            in_trade = False
                    elif trade_type == "SELL":
                        if current_price >= sl:
                            trades.append({"date": entry_date, "type": trade_type, "entry": entry_price, "exit": sl, "pnl": entry_price - sl, "result": "LOSS"})
                            in_trade = False
                        elif current_price <= tp:
                            trades.append({"date": entry_date, "type": trade_type, "entry": entry_price, "exit": tp, "pnl": entry_price - tp, "result": "WIN"})
                            in_trade = False
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
                col1.metric("Total Trades", total_trades)
                col2.metric("Wins", wins)
                col3.metric("Losses", losses)
                col4.metric("Win Rate", str(win_rate) + "%")
                col5.metric("Total PnL", total_pnl)
                trades_df["cumulative_pnl"] = trades_df["pnl"].cumsum()
                fig, axes = plt.subplots(2, 1, figsize=(14, 8))
                colors = ["green" if r == "WIN" else "red" for r in trades_df["result"]]
                axes[0].bar(range(len(trades_df)), trades_df["pnl"], color=colors)
                axes[0].axhline(0, color="white", linewidth=0.8)
                axes[0].set_title("Individual Trade PnL")
                axes[1].plot(range(len(trades_df)), trades_df["cumulative_pnl"], color="cyan", linewidth=2)
                axes[1].fill_between(range(len(trades_df)), trades_df["cumulative_pnl"], alpha=0.2, color="cyan")
                axes[1].axhline(0, color="white", linewidth=0.8)
                axes[1].set_title("Cumulative PnL Over Time")
                plt.tight_layout()
                st.pyplot(fig)
                st.subheader("Trade Log")
                st.dataframe(trades_df)
