import streamlit as st
import pandas as pd
from prophet import Prophet
import pandas_datareader.data as web
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import datetime
import time

# 1. UI SETUP
st.set_page_config(page_title="Master AI Terminal V3", layout="wide")
st.title("🏛️ Master AI Investment Terminal")
st.markdown("""
<div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; border: 1px solid #d1d5db;">
    🟢 <b>Data Sources:</b> Stooq (Historical Prices) + Finviz (Financial Stats) | <b>Status:</b> Unlimited Bypass
</div>
""", unsafe_allow_html=True)

# 2. SIDEBAR
st.sidebar.header("⚙️ Parameters")
stock_symbol = st.sidebar.text_input("Ticker", value="NVDA").upper()
total_capital = st.sidebar.number_input("Capital ($)", value=1000)

# 3. DATA ENGINE (NO YAHOO USED)
def get_data_unlimited(ticker):
    try:
        # A. Fetch Prices from Stooq (No Limit / No Key)
        # We use pandas_datareader to pull directly from Stooq's CSV export
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=1095) # 3 years
        df = web.DataReader(f"{ticker}.US", 'stooq', start, end)
        
        if df.empty:
            return None, 0, 0
            
        df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
        df = df.sort_values('ds')
        
        # B. Fetch Fundamentals from Finviz
        # Finviz is much less aggressive at blocking than Yahoo
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        f_res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(f_res.text, 'html.parser')
        
        # Function to find metrics in Finviz's data table
        def get_finviz_val(label):
            td = soup.find('td', string=label)
            if td:
                val = td.find_next_sibling('td').text
                return val.strip('%').replace(',', '')
            return "0"

        roe_str = get_finviz_val("ROE")
        debt_str = get_finviz_val("Debt/Eq")
        
        roe = float(roe_str) / 100 if roe_str != "-" else 0.0
        debt = float(debt_str) if debt_str != "-" else 0.0
        
        return df, roe, debt
    except Exception as e:
        st.error(f"Engine Error: {e}")
        return None, 0, 0

# 4. EXECUTION
if st.sidebar.button("🚀 Run Full Audit"):
    with st.spinner(f"📊 Analyzing {stock_symbol} via Stooq & Finviz..."):
        df, roe, de = get_data_unlimited(stock_symbol)
        
        if df is not None and not df.empty:
            # AI Prediction using Prophet
            m = Prophet(daily_seasonality=False, yearly_seasonality=True).fit(df)
            future = m.make_future_dataframe(periods=90)
            forecast = m.predict(future)
            
            cur_p = df['y'].iloc[-1]
            target_roi = ((forecast['yhat'].iloc[-1] - cur_p) / cur_p) * 100
            
            # POINT SYSTEM (2 INDICATORS: FINANCIALS + AI)
            f_score = 1 if (roe > 0.15 and de < 1.5) else 0
            ai_score = 1 if target_roi > 10 else 0
            points = f_score + ai_score 
            
            # SIGNAL LOGIC
            if points == 2:
                label, imm_pct = "🌟 HIGH CONVICTION BUY", 0.15
                strat = "Aggressive: Phase in remaining cash over 2 months. If price drops 5%, double monthly buy."
            elif points == 1:
                label, imm_pct = "🟡 ACCUMULATE / HOLD", 0.05
                strat = "Defensive: Phase in remaining cash over 4 months."
            else:
                label, imm_pct = "🛑 AVOID", 0.0
                strat = "Capital Preservation: AI and Fundamentals are not aligned. Stay in cash."

            imm_buy = total_capital * imm_pct

            # DASHBOARD UI
            if points == 2: st.success(f"### {label}")
            elif points == 1: st.warning(f"### {label}")
            else: st.error(f"### {label}")
            
            st.info(f"ROE: {roe*100:.1f}% | Debt/Eq: {de:.2f} | 90d Forecast: {target_roi:+.1f}%")

            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", f"${cur_p:.2f}")
            c2.metric("Immediate Action", f"${imm_buy:.2f}")
            c3.metric("Shares to Buy", f"{imm_buy/cur_p:.2f}" if imm_buy > 0 else "0.00")

            st.markdown("---")
            st.subheader("🎯 Investment Strategy")
            st.write(f"**Plan:** {strat}")
            
            st.markdown("---")
            st.subheader("🤖 90-Day AI Forecast")
            fig = m.plot(forecast)
            st.pyplot(fig)
            
            st.caption("⚠️ THIS IS AI. A HUMAN SHOULD USE THEIR BRAIN BEFORE INVESTING REAL MONEY.")
        else:
            st.error("❌ Data Fetch Failed. Ensure the Ticker is correct (e.g., AAPL, NVDA, TSLA).")
