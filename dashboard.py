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

    if rsi < 40:
        buy_signals += 1
    elif rsi > 60:
        sell_signals += 1

    if macd > signal:
        buy_signals += 1
    elif macd < signal:
        sell_signals += 1

    if close > ema200:
        buy_signals += 1
    elif close < ema200:
        sell_signals += 1

    if buy_signals >= 2 and buy_signals > sell_signals:
        return "BUY", "green"
    elif sell_signals >= 2 and sell_signals > buy_signals:
        return "SELL", "red"
    else:
        return "NEUTRAL", "orange"

def get_sl_tp(close, atr, signal, multiplier):
    sl_distance = atr * multiplier
    tp_distance = sl_distance * 2

    if signal == "BUY":
        stop_loss = round(close - sl_distance, 5)
        take_profit = round(close + tp_distance, 5)
    elif signal == "SELL":
        stop_loss = round(close + sl_distance, 5)
        take_profit = round(close - tp_distance, 5)
    else:
        stop_loss = None
        take_profit = None

    return stop_loss, take_profit

def get_support_resistance(df, window=10):
    levels = []
    for i in range(window, len(df) - window):
        high = df["High"].iloc[i]
        low = df["Low"].iloc[i]

        if high == df["High"].iloc[i - window:i + window].max():
            levels.append(("resistance", high, df.index[i]))

        if low == df["Low"].iloc[i - window:i + window].min():
            levels.append(("support", low, df.index[i]))

    # Remove duplicate levels that are too close
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

        # Bullish order block - bearish candle before strong bullish move
        if (current["Close"] < current["Open"] and
                next_candle["Close"] > next_candle["Open"] and
                next_candle["Close"] > current["High"]):
            order_blocks.append({
                "type": "bullish",
                "top": current["Open"],
                "bottom": current["Close"],
                "date": df.index[i]
            })

        # Bearish order block - bullish candle before strong bearish move
        if (current["Close"] > current["Open"] and
                next_candle["Close"] < next_candle["Open"] and
                next_candle["Close"] < current["Low"]):
            order_blocks.append({
                "type": "bearish",
                "top": current["Close"],
                "bottom": current["Open"],
                "date": df.index[i]
            })

    return order_blocks[-5:] if len(order_blocks) > 5 else order_blocks

def get_bos_choch(df):
    events = []
    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    dates = df.index

    for i in range(3, len(df) - 1):
        # Break of Structure - bullish
        if closes[i] > highs[i - 1] and closes[i - 1] < highs[i - 2]:
            events.append({"type": "BOS_BULL", "price": closes[i], "date": dates[i]})

        # Break of Structure - bearish
        if closes[i] < lows[i - 1] and closes[i - 1] > lows[i - 2]:
            events.append({"type": "BOS_BEAR", "price": closes[i], "date": dates[i]})

        # Change of Character - bullish (downtrend broken)
        if (closes[i - 2] < closes[i - 3] and
                closes[i - 1] < closes[i - 2] and
                closes[i] > closes[i - 1] and
                closes[i] > highs[i - 2]):
            events.append({"type": "CHOCH_BULL", "price": closes[i], "date": dates[i]})

        # Change of Character - bearish (uptrend broken)
        if (closes[i - 2] > closes[i - 3] and
                closes[i - 1] > closes[i - 2] and
                closes[i] < closes[i - 1] and
                closes[i] < lows[i - 2]):
            events.append({"type": "CHOCH_BEAR", "price": closes[i], "date": dates[i]})

    return events[-10:] if len(events) > 10 else events

# TAB 1 - LIVE ANALYSIS
with tab1:
    pairs = st.multiselect(
        "Select currency pairs to analyze:",
        ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "USDCHF=X"],
        default=["EURUSD=X"]
    )

    period = st.selectbox("Select time period:", ["1mo", "3mo", "6mo", "1y"], index=2)
    atr_multiplier = st.slider("ATR Multiplier for Stop Loss", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

    if st.button("Run Analysis"):
        for symbol in pairs:
            with st.spinner("Analyzing " + symbol + "..."):
                df = yf.download(symbol, period=period, interval="1d", progress=False)
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
                    "You are a professional forex analyst. Analyze " + symbol + " based on these indicators:\n"
                    "- Current Price: " + str(close) + "\n"
                    "- 20-day MA: " + str(ma20) + "\n"
                    "- 50-day MA: " + str(ma50) + "\n"
                    "- 200 EMA: " + str(ema200) + "\n"
                    "- RSI (14): " + str(rsi) + "\n"
                    "- Bollinger Upper: " + str(bb_upper) + "\n"
                    "- Bollinger Lower: " + str(bb_lower) + "\n"
                    "- MACD: " + str(macd) + "\n"
                    "- Signal Line: " + str(signal) + "\n"
                    "- Stochastic K: " + str(stoch_k) + "\n"
                    "- Stochastic D: " + str(stoch_d) + "\n"
                    "- ATR: " + str(atr) + "\n"
                    "- Entry Signal: " + entry_signal + "\n"
                    "- Stop Loss: " + str(stop_loss) + "\n"
                    "- Take Profit: " + str(take_profit) + "\n"
                    "Give a professional analysis in 5 sentences. Explain the entry signal, justify the stop loss and take profit levels, and mention key risks."
                )

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}]
                )

                st.subheader(symbol)
                st.markdown(
                    "<div style='background-color:" + signal_color + "; padding:10px; border-radius:8px; text-align:center;'>"
                    "<h2 style='color:white; margin:0;'>ENTRY SIGNAL: " + entry_signal + "</h2>"
                    "</div>",
                    unsafe_allow_html=True
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
                    st.warning("No clear entry signal — no stop loss or take profit calculated.")

                # Candlestick chart with S/R
                df_plot = df.tail(60)
                add_plots = [
                    mpf.make_addplot(df_plot["BB_Upper"], color="red", linestyle="--", width=0.8),
                    mpf.make_addplot(df_plot["BB_Lower"], color="green", linestyle="--", width=0.8),
                    mpf.make_addplot(df_plot["EMA200"], color="purple", width=1),
                    mpf.make_addplot(df_plot["MA20"], color="orange", width=0.8),
                ]

                hlines = []
                hline_colors = []
                for level_type, level_price, level_date in sr_levels:
                    hlines.append(level_price)
                    hline_colors.append("blue" if level_type == "support" else "magenta")

                if stop_loss:
                    hlines.append(stop_loss)
                    hline_colors.append("red")
                if take_profit:
                    hlines.append(take_profit)
                    hline_colors.append("lime")

                fig_candle, axes_candle = mpf.plot(
                    df_plot,
                    type="candle",
                    style="nightclouds",
                    title=symbol + " Candlestick Chart",
                    addplot=add_plots,
                    hlines=dict(hlines=hlines, colors=hline_colors, linestyle="--", linewidths=0.8),
                    volume=False,
                    returnfig=True,
                    figsize=(14, 6)
                )
                st.pyplot(fig_candle)

                # MACD and Stochastic
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

# TAB 2 - SMART MONEY CONCEPTS
with tab2:
    st.header("Smart Money Concepts")
    st.markdown("Order Blocks, Break of Structure (BOS), and Change of Character (CHoCH)")

    smc_symbol = st.selectbox("Select pair:", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"], key="smc")
    smc_period = st.selectbox("Select period:", ["1mo", "3mo", "6mo"], index=1, key="smc_period")

    if st.button("Run SMC Analysis"):
        with st.spinner("Analyzing Smart Money Concepts..."):
            df = yf.download(smc_symbol, period=smc_period, interval="1d", progress=False)
            df.columns = df.columns.droplevel(1)
            df = compute_indicators(df)
            df = df.dropna()

            order_blocks = get_order_blocks(df)
            bos_choch = get_bos_choch(df)
            sr_levels = get_support_resistance(df)

            df_plot = df.tail(80)

            # Build order block patches
            fig_smc, ax = mpf.plot(
                df_plot,
                type="candle",
                style="nightclouds",
                title=smc_symbol + " Smart Money Concepts",
                volume=False,
                returnfig=True,
                figsize=(16, 8)
            )

            ax_main = fig_smc.axes[0]

            # Draw order blocks
            for ob in order_blocks:
                if ob["date"] in df_plot.index:
                    idx = df_plot.index.get_loc(ob["date"])
                    color = "lime" if ob["type"] == "bullish" else "tomato"
                    rect = mpatches.FancyBboxPatch(
                        (idx - 0.5, ob["bottom"]),
                        len(df_plot) - idx,
                        ob["top"] - ob["bottom"],
                        boxstyle="round,pad=0",
                        linewidth=1,
                        edgecolor=color,
                        facecolor=color,
                        alpha=0.2
                    )
                    ax_main.add_patch(rect)
                    ax_main.text(idx, ob["top"], ob["type"].upper() + " OB", fontsize=6, color=color)

            # Draw S/R levels
            for level_type, level_price, level_date in sr_levels:
                color = "cyan" if level_type == "support" else "magenta"
                ax_main.axhline(level_price, color=color, linestyle="--", linewidth=0.8, alpha=0.7)

            # Mark BOS and CHoCH
            for event in bos_choch:
                if event["date"] in df_plot.index:
                    idx = df_plot.index.get_loc(event["date"])
                    if "BULL" in event["type"]:
                        color = "lime"
                        marker = "^"
                    else:
                        color = "red"
                        marker = "v"

                    label = event["type"].replace("_BULL", "").replace("_BEAR", "")
                    ax_main.annotate(
                        label,
                        xy=(idx, event["price"]),
                        fontsize=7,
                        color=color,
                        fontweight="bold"
                    )

            st.pyplot(fig_smc)

            # Summary
            st.subheader("SMC Summary")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Order Blocks**")
                for ob in order_blocks:
                    emoji = "🟢" if ob["type"] == "bullish" else "🔴"
                    st.write(emoji + " " + ob["type"].upper() + " OB at " + str(round(ob["top"], 5)))

            with col2:
                st.markdown("**Support Levels**")
                for level_type, level_price, _ in sr_levels:
                    if level_type == "support":
                        st.write("🔵 Support at " + str(round(level_price, 5)))

            with col3:
                st.markdown("**Resistance Levels**")
                for level_type, level_price, _ in sr_levels:
                    if level_type == "resistance":
                        st.write("🟣 Resistance at " + str(round(level_price, 5)))

            # AI SMC interpretation
            ob_text = ", ".join([ob["type"] + " OB at " + str(round(ob["top"], 5)) for ob in order_blocks])
            bos_text = ", ".join([e["type"] + " at " + str(round(e["price"], 5)) for e in bos_choch[-3:]])

            smc_prompt = (
                "You are a Smart Money Concepts forex analyst. Analyze " + smc_symbol + " using SMC:\n"
                "- Recent Order Blocks: " + ob_text + "\n"
                "- Recent BOS/CHoCH events: " + bos_text + "\n"
                "- Current Price: " + str(round(float(df["Close"].iloc[-1]), 5)) + "\n"
                "Explain what these SMC signals mean for the next likely price movement in 4-5 sentences. "
                "Mention whether price is likely to seek liquidity, respect order blocks, or continue the current structure."
            )

            smc_response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": smc_prompt}]
            )

            st.info(smc_response.choices[0].message.content)

# TAB 3 - BACKTESTING
with tab3:
    st.header("Backtest Your Strategy")
    st.markdown("Tests whether the BUY/SELL signals would have been profitable historically.")

    bt_symbol = st.selectbox("Select pair to backtest:", ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"])
    bt_period = st.selectbox("Select backtest period:", ["6mo", "1y", "2y", "5y"], index=1)
    bt_multiplier = st.slider("ATR Multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1, key="bt")

    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            df = yf.download(bt_symbol, period=bt_period, interval="1d", progress=False)
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
                            float(row["RSI"]),
                            float(row["MACD"]),
                            float(row["Signal"]),
                            float(row["Close"]),
                            float(row["EMA200"])
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
                            pnl = sl - entry_price
                            trades.append({"date": entry_date, "type": trade_type, "entry": entry_price, "exit": sl, "pnl": pnl, "result": "LOSS"})
                            in_trade = False
                        elif current_price >= tp:
                            pnl = tp - entry_price
                            trades.append({"date": entry_date, "type": trade_type, "entry": entry_price, "exit": tp, "pnl": pnl, "result": "WIN"})
                            in_trade = False

                    elif trade_type == "SELL":
                        if current_price >= sl:
                            pnl = entry_price - sl
                            trades.append({"date": entry_date, "type": trade_type, "entry": entry_price, "exit": sl, "pnl": pnl, "result": "LOSS"})
                            in_trade = False
                        elif current_price <= tp:
                            pnl = entry_price - tp
                            trades.append({"date": entry_date, "type": trade_type, "entry": entry_price, "exit": tp, "pnl": pnl, "result": "WIN"})
                            in_trade = False

            if len(trades) == 0:
                st.warning("No completed trades found in this period. Try a longer period.")
            else:
                trades_df = pd.DataFrame(trades)
                total_trades = len(trades_df)
                wins = len(trades_df[trades_df["result"] == "WIN"])
                losses = len(trades_df[trades_df["result"] == "LOSS"])
                win_rate = round((wins / total_trades) * 100, 2)
                total_pnl = round(trades_df["pnl"].sum(), 5)

                st.subheader("Backtest Results — " + bt_symbol)

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
                axes[0].set_xlabel("Trade Number")
                axes[0].set_ylabel("PnL")

                axes[1].plot(range(len(trades_df)), trades_df["cumulative_pnl"], color="cyan", linewidth=2)
                axes[1].fill_between(range(len(trades_df)), trades_df["cumulative_pnl"], alpha=0.2, color="cyan")
                axes[1].axhline(0, color="white", linewidth=0.8)
                axes[1].set_title("Cumulative PnL Over Time")
                axes[1].set_xlabel("Trade Number")
                axes[1].set_ylabel("Cumulative PnL")

                plt.tight_layout()
                st.pyplot(fig)

                st.subheader("Trade Log")
                st.dataframe(trades_df)
