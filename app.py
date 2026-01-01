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
    🟢 <b>Data Sources:</b> Stooq (Prices) + Finviz (Fundamentals) | <b>Status:</b> Unlimited Bypass Enabled
</div>
""", unsafe_allow_html=True)

# 2. SIDEBAR
st.sidebar.header("⚙️ Parameters")
stock_symbol = st.sidebar.text_input("Ticker", value="NVDA").upper()
total_capital = st.sidebar.number_input("Capital ($)", value=1000)

# 3. DATA ENGINE (NO YAHOO)
def get_data_unlimited(ticker):
    try:
        # A. Fetch Prices from Stooq
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=1095) 
        # Stooq is a reliable alternative to Yahoo for price history
        df = web.DataReader(f"{ticker}.US", 'stooq', start, end)
        
        if df is None or df.empty:
            return None, 0, 0
            
        df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
        df = df.sort_values('ds')
        
        # B. Fetch Fundamentals from Finviz
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        f_res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(f_res.text, 'html.parser')
        
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
        return None, 0, 0

# 4. EXECUTION
if st.sidebar.button("🚀 Run Full Audit"):
    with st.spinner(f"📊 Analyzing {stock_symbol}..."):
        df, roe, de = get_data_unlimited(stock_symbol)
        
        if df is not None:
            # AI Prediction
            m = Prophet(daily_seasonality=False, yearly_seasonality=True)
            m.fit(df)
            future = m.make_future_dataframe(periods=90)
            forecast = m.predict(future)
            
            cur_p = df['y'].iloc[-1]
            target_roi = ((forecast['yhat'].iloc[-1] - cur_p) / cur_p) * 100
            
            # SCORING Logic
            f_score = 1 if (roe > 0.15 and de < 1.5) else 0
            ai_score = 1 if target_roi > 10 else 0
            points = f_score + ai_score # Max points 2 in this source version
            
            # STRATEGY MAPPING
            if points == 2:
                label, imm_pct = "🌟 HIGH CONVICTION BUY", 0.15
                strat = "Aggressive: Invest 15% now. Phase in rest over 2 months. If price drops 5%, double the monthly buy."
            elif points == 1:
                label, imm_pct = "🟡 ACCUMULATE / HOLD", 0.05
                strat = "Defensive: Invest 5% now. Park rest in SGOV ETF. Phase in over 4 months."
            else:
                label, imm_pct = "🛑 AVOID", 0.0
                strat = "Capital Preservation: AI ROI or Fundamentals are weak. Keep capital in cash."

            imm_buy = total_capital * imm_pct

            # DASHBOARD UI
            if points == 2: st.success(f"### {label}")
            elif points == 1: st.warning(f"### {label}")
            else: st.error(f"### {label}")
            
            st.info(f"Financial Score: {'PASS' if f_score else 'FAIL'} (ROE: {roe*100:.1f}%) | 90d AI ROI: {target_roi:+.1f}%")

            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", f"${cur_p:.2f}")
            c2.metric("Action 1 (Buy)", f"${imm_buy:.2f}")
            c3.metric("Shares", f"{imm_buy/cur_p:.2f}" if imm_buy > 0 else "0.0")

            st.markdown("---")
            st.subheader("🎯 Strategy Detail")
            st.write(f"**Strategy:** {strat}")
            
            st.markdown("---")
            st.subheader("🤖 AI Price Trajectory")
            fig = m.plot(forecast)
            st.pyplot(fig)
            
            st.caption("⚠️ THIS IS AI. A HUMAN SHOULD USE THEIR BRAIN BEFORE INVESTING REAL MONEY.")
        else:
            st.error("❌ Data Fetch Failed. Check the ticker or try again in 1 minute.")
