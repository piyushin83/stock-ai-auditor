import streamlit as st
import pandas as pd
from prophet import Prophet
import pandas_datareader.data as web
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import datetime

# 1. UI SETUP & CUSTOM REFINED CSS
st.set_page_config(page_title="Master AI Terminal V3", layout="wide")

st.markdown("""
<style>
    /* Metric Value - Reduced font size for cleaner look */
    [data-testid="stMetricValue"] { 
        font-size: 28px !important; 
        font-weight: 700 !important; 
    }
    /* Metric Label - Reduced font size */
    [data-testid="stMetricLabel"] p { 
        font-size: 14px !important; 
        color: #555;
    }
    /* Stop Loss Highlight Box */
    .stop-loss-box {
        background-color: #fff1f1;
        padding: 12px;
        border-radius: 8px;
        border-left: 6px solid #ff4b4b;
        margin-bottom: 20px;
    }
    .stop-loss-text {
        font-size: 18px;
        font-weight: bold;
        color: #d32f2f;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Strategic AI Investment Architect")

# 2. SIDEBAR CONFIGURATION
st.sidebar.header("⚙️ Configuration")
stock_symbol = st.sidebar.text_input("Ticker", value="NVDA").upper()
total_capital = st.sidebar.number_input("Capital ($)", value=1000)

# 3. DATA ENGINE (STOOQ + FINVIZ BYPASS)
def get_full_audit(ticker):
    try:
        # Fetch 3 Years of History
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=1095) 
        df = web.DataReader(f"{ticker}.US", 'stooq', start, end)
        
        if df is None or df.empty:
            return None, None
            
        df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'}).sort_values('ds')
        
        # Fetch Stats via Finviz
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        f_res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(f_res.text, 'html.parser')
        
        def fvz(label):
            td = soup.find('td', string=label)
            return td.find_next_sibling('td').text.strip('%').replace(',', '') if td else "0"

        health = {
            "ROE": float(fvz("ROE")) / 100 if fvz("ROE") != "-" else 0,
            "Debt/Eq": float(fvz("Debt/Eq")) if fvz("Debt/Eq") != "-" else 0,
            "Current Ratio": fvz("Current Ratio"),
            "Profit Margin": fvz("Profit Margin") + "%"
        }
        return df, health
    except:
        return None, None

# 4. EXECUTION
if st.sidebar.button("🚀 Run Full Audit"):
    with st.spinner(f"📡 Auditing {stock_symbol}..."):
        df, health = get_full_audit(stock_symbol)
        
        if df is not None:
            # --- AI PREDICTION (180 DAYS) ---
            m = Prophet(daily_seasonality=False, yearly_seasonality=True).fit(df)
            future = m.make_future_dataframe(periods=180)
            forecast = m.predict(future)
            
            cur_p = df['y'].iloc[-1]
            # Calculate 180-day ROI from forecast
            roi_180 = ((forecast['yhat'].iloc[-1] - cur_p) / cur_p) * 100
            
            # Conviction Logic (3 Point Score)
            points = (1 if health['ROE'] > 0.15 else 0) + (1 if roi_180 > 10 else 0) + (1 if health['Debt/Eq'] < 1.1 else 0)

            # --- DYNAMIC STRATEGY MAPPING ---
            if points == 3:
                mode, label, pct = "Aggressive", "🌟 STRONG BUY", 0.15
                risk_lvl = "Low-to-Moderate"
                sl_price = cur_p * 0.90  # 10% Stop Loss
                sl_msg = f"STOP LOSS: Exit immediately if price drops below ${sl_price:.2f} (10% Buffer)."
                ui_box = st.success
            elif points >= 1:
                mode, label, pct = "Defensive", "🟡 ACCUMULATE / HOLD", 0.05
                risk_lvl = "Moderate"
                sl_price = cur_p * 0.85  # 15% Stop Loss
                sl_msg = f"STOP LOSS: Exit immediately if price drops below ${sl_price:.2f} (15% Buffer)."
                ui_box = st.warning
            else:
                mode, label, pct = "Preservation", "🛑 AVOID / SELL", 0.0
                risk_lvl = "High Risk"
                sl_msg = "STOP LOSS: No entry recommended. Liquidate existing positions."
                ui_box = st.error

            imm_buy = total_capital * pct
            staging_amt = total_capital - imm_buy

            # --- DASHBOARD RENDERING ---
            ui_box(f"### Strategy Selected: {mode}: {label}")
            
            # Stop Loss Statement
            st.markdown(f'<div class="stop-loss-box"><p class="stop-loss-text">{sl_msg}</p></div>', unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Action 1: Buy", f"${imm_buy:.2f}")
            c2.metric("Company Risk", risk_lvl)
            c3.metric("180-Day AI ROI", f"{roi_180:+.1f}%")
            c4.metric("Current Price", f"${cur_p:.2f}")

            st.markdown("---")
            
            # PHASED INSTRUCTIONS
            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("🚀 PHASE 1: IMMEDIATE")
                if imm_buy > 0:
                    st.write(f"**Action:** BUY - Invest **${imm_buy:.2f}** today.")
                    st.write(f"Acquire approx **{imm_buy/cur_p:.2f} shares** at current market price.")
                else:
                    st.write("**Action:** HOLD/SELL - Do not deploy new capital.")

            with col_r:
                st.subheader("⏳ PHASE 2: STAGING")
                if points == 3:
                    st.write(f"**Plan:** Deploy **${staging_amt:.2f}** over the next 60 days. Double monthly buys if price dips 5%.")
                elif points >= 1:
                    st.write(f"**Plan:** Park **${staging_amt:.2f}** in SGOV ETF. Accumulate only on significant market pullbacks.")
                else:
                    st.write(f"**Plan:** Maintain **${total_capital:.2f}** in cash reserves (T-Bills/MMF).")

            st.markdown("---")

            # HEALTH & RISK AUDIT
            col_h, col_ri = st.columns(2)
            with col_h:
                st.subheader("🏥 Company Health Detail")
                st.table(pd.DataFrame({
                    "Metric": ["Efficiency (ROE)", "Solvency (Debt/Eq)", "Liquidity Ratio", "Profit Margin"],
                    "Status": [f"{health['ROE']*100:.1f}%", health['Debt/Eq'], health['Current Ratio'], health['Profit Margin']],
                    "Rating": ["✅ Prime" if health['ROE'] > 0.15 else "⚠️ Low",
                               "✅ Safe" if health['Debt/Eq'] < 1.1 else "⚠️ Leveraged", "✅ Liquid", "✅ Stable"]
                }))

            with col_ri:
                st.subheader("⚖️ Risk Analysis")
                st.write(f"**Audit Conviction Score:** {int((points/3)*100)}%")
                st.progress(points / 3)
                st.markdown(f"""
                - **Leverage Risk:** {'Low' if health['Debt/Eq'] < 1.2 else 'High'}
                - **Trend Strength:** {'Strong' if roi_180 > 15 else 'Moderate/Weak'}
                - **Staging Priority:** {'Aggressive' if points == 3 else 'Patient'}
                """)

            # 180-DAY GRAPH
            st.markdown("---")
            st.subheader("🤖 180-Day AI Price Projection")
            fig = m.plot(forecast)
            plt.title(f"{stock_symbol} - 180 Day AI Trend Analysis")
            st.pyplot(fig)
            
            st.warning("⚠️ AI ADVISORY: Forecasts are mathematical probabilities. Human judgment is required before investing.")
        else:
            st.error("❌ Data Fetch Error: Please verify ticker or ensure requirements.txt includes pandas_datareader.")
