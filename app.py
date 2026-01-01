import streamlit as st
import pandas as pd
from prophet import Prophet
import requests
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time

# 1. INITIALIZE ENGINES
@st.cache_resource
def load_essentials():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = load_essentials()

# 2. UI SETUP
st.set_page_config(page_title="Master AI Terminal V2", layout="wide")
st.title("🏛️ Master AI Investment Terminal")
st.caption("Now Powered by Alpha Vantage Intelligence")

st.sidebar.header("🔑 Authentication")
api_key = st.sidebar.text_input("Alpha Vantage API Key", type="password")
st.sidebar.info("Get a free key at alphavantage.co")

st.sidebar.header("⚙️ Parameters")
stock_symbol = st.sidebar.text_input("Ticker", value="NVDA").upper()
total_capital = st.sidebar.number_input("Capital ($)", value=1000)

# 3. ALPHA VANTAGE DATA ENGINE
def fetch_alpha_data(ticker, key):
    # Fetch Prices (Daily Adjusted)
    price_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={ticker}&apikey={key}&outputsize=full'
    # Fetch Fundamentals
    fund_url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={key}'
    
    try:
        p_res = requests.get(price_url).json()
        f_res = requests.get(fund_url).json()
        
        # Parse Prices
        time_series = p_res['Time Series (Daily)']
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={'4. close': 'Close'}).sort_index()
        df['Date'] = df.index
        df['Close'] = df['Close'].astype(float)
        
        # Parse Fundamentals
        roe = float(f_res.get('ReturnOnEquityTTM', 0.12))
        debt = float(f_res.get('DebtToEquityRatio', 0.5))
        
        return df, roe, debt
    except Exception as e:
        st.error(f"Data Error: {str(e)}")
        return None, 0, 0

# 4. EXECUTION
if st.sidebar.button("🔍 Run Full Audit"):
    if not api_key:
        st.warning("Please enter your API Key first.")
    else:
        with st.spinner(f"📡 Accessing Alpha Vantage for {stock_symbol}..."):
            df, roe, de = fetch_alpha_data(stock_symbol, api_key)
            
            if df is not None:
                # Prophet Forecasting
                df_p = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
                df_p['ds'] = df_p['ds'].dt.tz_localize(None)
                m = Prophet().fit(df_p.tail(1000)) # Last ~3 years
                future = m.make_future_dataframe(periods=90)
                forecast = m.predict(future)
                
                cur_p = df['Close'].iloc[-1]
                target_roi = ((forecast['yhat'].iloc[-1] - cur_p) / cur_p) * 100
                
                # Scoring Logic
                f_score = 1 if (roe > 0.15 and de < 1.5) else 0
                ai_score = 1 if target_roi > 10 else 0
                points = f_score + ai_score # Sentiment can be added via Alpha Vantage News API
                
                # ALLOCATION MAPPING
                if points >= 2:
                    label, imm_pct = "🌟 HIGH CONVICTION BUY", 0.15
                    strat = "Aggressive: Phase in remaining cash over 2 months."
                elif points == 1:
                    label, imm_pct = "🟡 ACCUMULATE / HOLD", 0.05
                    strat = "Defensive: Phase in remaining cash over 4 months."
                else:
                    label, imm_pct = "🛑 AVOID", 0.0
                    strat = "Capital Preservation: Stay in cash."

                imm_buy = total_capital * imm_pct

                # DASHBOARD
                st.subheader(f"Strategy Report: {stock_symbol}")
                if points >= 2: st.success(label)
                elif points == 1: st.warning(label)
                else: st.error(label)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Current Price", f"${cur_p:.2f}")
                c2.metric("Predicted 90d ROI", f"{target_roi:+.1f}%")
                c3.metric("Immediate Buy", f"${imm_buy:.2f}")

                st.markdown("---")
                st.write(f"**Action Plan:** {strat}")
                st.pyplot(m.plot(forecast))
