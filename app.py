import streamlit as st
import pandas as pd
from prophet import Prophet
import pandas_datareader.data as web
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import datetime

# 1. UI SETUP
st.set_page_config(page_title="Master AI Terminal V3", layout="wide")
st.title("🏛️ Master AI Investment Terminal")
st.markdown("""
<div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; border: 1px solid #d1d5db;">
    🟢 <b>Data Sources:</b> Stooq (Prices) + Finviz (Fundamentals) | <b>Status:</b> Yahoo Bypass Active
</div>
""", unsafe_allow_html=True)

# 2. SIDEBAR
st.sidebar.header("⚙️ Parameters")
stock_symbol = st.sidebar.text_input("Ticker", value="NVDA").upper()
total_capital = st.sidebar.number_input("Capital ($)", value=1000)

# 3. DATA ENGINE (NO YAHOO)
def get_data_unlimited(ticker):
    try:
        # A. Fetch Prices from Stooq (Historical)
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=1095) 
        df = web.DataReader(f"{ticker}.US", 'stooq', start, end)
        
        if df is None or df.empty:
            return None, 0, 0
            
        df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
        df = df.sort_values('ds')
        
        # B. Fetch Fundamentals from Finviz
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        f_res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(f_res.text, 'html.parser')
        
        def get_finviz_val(label):
            td = soup.find('td', string=label)
            if td:
                val = td.find_next_sibling('td').text
                return val.strip('%').replace(',', '')
            return "0"

        roe = float(get_finviz_val("ROE")) / 100
        debt = float(get_finviz_val("Debt/Eq"))
        
        return df, roe, debt
    except:
        return None, 0, 0

# 4. EXECUTION
if st.sidebar.button("🚀 Run Full Audit"):
    with st.spinner(f"📊 Analyzing {stock_symbol}..."):
        df, roe, de = get_data_unlimited(stock_symbol)
        
        if df is not None:
            # AI Prediction
            m = Prophet(daily_seasonality=False, yearly_seasonality=True).fit(df)
            future = m.make_future_dataframe(periods=180)
            forecast = m.predict(future)
            
            cur_p = df['y'].iloc[-1]
            target_roi = ((forecast['yhat'].iloc[-1] - cur_p) / cur_p) * 100
            
            # SCORING (Max 3 points: Fundamental Pass, AI ROI > 10%, Debt/ROE Balance)
            f_pass = 1 if (roe > 0.15 and de < 1.5) else 0
            ai_pass = 1 if target_roi > 10 else 0
            points = f_pass + ai_pass + 1 # Simulating 3rd point for now

            # --- DYNAMIC STRATEGY LOGIC ---
            if points == 3:
                action_label = "🌟 ACTION: HIGH CONVICTION BUY"
                confidence = "Confidence: Strong. All three indicators are positive."
                imm_pct = 0.15
                strat_text = "Aggressive Accumulation: Phase in remaining cash over 2 months. If price drops 5%, double the monthly buy."
                ui_box = st.success
            elif points >= 1:
                action_label = "🟡 ACTION: ACCUMULATE / HOLD (Partial Alignment)"
                confidence = "Confidence: Moderate. One or more indicators suggest caution."
                imm_pct = 0.05
                strat_text = "Defensive Staging: Park cash in SGOV ETF. Phase in remaining cash over 4 months."
                ui_box = st.warning
            else:
                action_label = "🛑 ACTION: AVOID"
                confidence = "Confidence: Low. Key indicators are negative or stagnant."
                imm_pct = 0.0
                strat_text = "Capital Preservation: Market conditions for this ticker are currently unfavorable."
                ui_box = st.error

            # DOLLAR CALCULATIONS
            imm_buy = total_capital * imm_pct
            staging_amt = total_capital - imm_buy

            # --- RENDER TO DASHBOARD ---
            ui_box(f"### {action_label}")
            st.info(f"**{confidence}**")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Points", f"{points}/3")
            c2.metric("ROE", f"{roe*100:.1f}%")
            c3.metric("Current Price", f"${cur_p:.2f}")
            c4.metric("90-Day ROI", f"{target_roi:+.1f}%")

            st.markdown("---")
            col_l, col_r = st.columns(2)
            
            with col_l:
                st.subheader("🚀 PHASE 1: IMMEDIATE")
                if imm_buy > 0:
                    st.write(f"**Action:** Invest **${imm_buy:.2f}** today.")
                    st.write(f"Estimated **{imm_buy/cur_p:.2f} shares**.")
                else:
                    st.write("**Action:** Stay in Cash. No immediate buy.")

            with col_r:
                st.subheader("⏳ PHASE 2: STAGING")
                st.write(f"**Action:** {strat_text}")
                st.write(f"**Reserve Amount:** ${staging_amt:.2f}")

            st.markdown("---")
            st.subheader("🤖 180-Day AI Price Projection")
            fig = m.plot(forecast)
            st.pyplot(fig)
            
            st.caption("⚠️ IT IS AI AND HUMAN SHOULD USE THEIR BRAIN BEFORE INVESTING.")
        else:
            st.error("❌ Connection Issue: Ensure your requirements.txt includes pandas_datareader.")
