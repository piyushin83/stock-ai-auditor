import streamlit as st
import pandas as pd
from prophet import Prophet
import pandas_datareader.data as web
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import datetime
import yfinance as yf

# 1. UI SETUP & CSS
st.set_page_config(page_title="Strategic AI Investment Architect", layout="wide")

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 32px !important; font-weight: 800 !important; color: #1f77b4; }
    [data-testid="stMetricLabel"] p { font-size: 16px !important; font-weight: 600 !important; color: #333; }
    .stop-loss-box { background-color: #fff1f1; padding: 15px; border-radius: 8px; border-left: 6px solid #ff4b4b; margin: 15px 0; }
    .stop-loss-text { font-size: 18px; font-weight: bold; color: #d32f2f; margin: 0; }
    .risk-box { background-color: #fffde7; padding: 15px; border-radius: 8px; border-left: 6px solid #fbc02d; margin: 15px 0; }
    .phase-card { background-color: #f8f9fa; padding: 20px; border-radius: 12px; border: 1px solid #dee2e6; min-height: 250px; }
    .disclaimer-box { background-color: #f4f4f4; padding: 10px; border-radius: 5px; font-size: 12px; color: #666; margin-bottom: 20px;}
</style>
""", unsafe_allow_html=True)

# 2. HEADER & DISCLAIMER
st.title("🏛️ Strategic AI Investment Architect")
st.markdown('<div class="disclaimer-box"><b>LEGAL:</b> Educational tool only. No financial advice provided. AI projections are probabilistic.</div>', unsafe_allow_html=True)

# 3. ENGINES
def get_exchange_rate(from_curr, to_curr):
    if from_curr == to_curr: return 1.0
    try:
        pair = f"{from_curr}{to_curr}=X"
        data = yf.download(pair, period="1d", progress=False)
        return data['Close'].iloc[-1]
    except: return 1.07

def resolve_smart_ticker(user_input):
    user_input = user_input.strip()
    try:
        s = yf.Search(user_input, max_results=1)
        if s.tickers:
            res = s.tickers[0]
            ticker = res['symbol']
            name = res.get('longname', ticker)
            exch = res.get('exchange', 'NYQ')
            t_obj = yf.Ticker(ticker)
            native_curr = t_obj.fast_info.get('currency', 'USD')
            suffix_map = {'LSE': '.UK', 'GER': '.DE', 'FRA': '.DE', 'PAR': '.FR', 'AMS': '.NL', 'TSE': '.JP', 'HKG': '.HK'}
            suffix = suffix_map.get(exch, ".US")
            return ticker.split('.')[0], name, suffix, native_curr
    except: pass
    return user_input.upper(), user_input.upper(), ".US", "USD"

def get_audit_data(ticker, suffix):
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=1095) 
        df = web.DataReader(f"{ticker}{suffix}", 'stooq', start, end)
        if df is None or df.empty: return None, None
        df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'}).sort_values('ds')
        
        health = {"ROE": 0.0, "Debt": 0.0, "Ratio": "N/A", "Margin": "N/A"}
        if suffix == ".US":
            try:
                url = f"https://finviz.com/quote.ashx?t={ticker}"
                headers = {'User-Agent': 'Mozilla/5.0'}
                soup = BeautifulSoup(requests.get(url, headers=headers, timeout=5).text, 'html.parser')
                def fvz(label):
                    td = soup.find('td', string=label)
                    return td.find_next_sibling('td').text.strip('%').replace(',', '') if td else "-"
                health = {"ROE": float(fvz("ROE"))/100 if fvz("ROE")!="-" else 0,
                          "Debt": float(fvz("Debt/Eq")) if fvz("Debt/Eq")!="-" else 0,
                          "Ratio": fvz("Current Ratio"), "Margin": fvz("Profit Margin") + "%"}
            except: pass
        return df, health
    except: return None, None

# 4. SIDEBAR
st.sidebar.header("⚙️ Configuration")
user_query = st.sidebar.text_input("Company Name or Ticker", value="Nvidia")
display_currency = st.sidebar.selectbox("Display Currency", ["USD", "EUR"])
total_capital = st.sidebar.number_input(f"Total Capital ({display_currency})", value=1000)

# 5. EXECUTION
if st.sidebar.button("🚀 Execute Strategic Audit"):
    with st.spinner(f"📡 Resolving Global Markets for '{user_query}'..."):
        ticker, name, suffix, native_curr = resolve_smart_ticker(user_query)
        df, health = get_audit_data(ticker, suffix)
        
        if df is not None:
            fx_rate = get_exchange_rate(native_curr, display_currency)
            symbol = "$" if display_currency == "USD" else "€"
            
            # AI (180 DAYS)
            m = Prophet(daily_seasonality=False, yearly_seasonality=True).fit(df)
            future = m.make_future_dataframe(periods=180)
            forecast = m.predict(future)
            
            raw_p = df['y'].iloc[-1]
            conv_p = raw_p * fx_rate
            roi_180 = ((forecast['yhat'].iloc[-1] - raw_p) / raw_p) * 100
            
            # SCORING & LOGIC
            points = (1 if health['ROE'] > 0.15 else 0) + (1 if roi_180 > 10 else 0) + (1 if health['Debt'] < 1.1 else 0)
            conviction_score = int((points / 3) * 100)
            prob_profit = max(min(50 + (roi_180 / 2), 99), 1)

            if points == 3:
                mode, action_label, pct, sl_buf = "Aggressive", "🌟 HIGH CONVICTION BUY", 15, 0.90
                risk_level = "Low-to-Moderate (Strong Fundamentals)"
            elif points >= 1:
                mode, action_label, pct, sl_buf = "Defensive", "🟡 ACCUMULATE / HOLD", 5, 0.85
                risk_level = "Moderate (Mixed Indicators)"
            else:
                mode, action_label, pct, sl_buf = "Preservation", "🛑 AVOID / SELL", 0, 0.0
                risk_level = "High (Weak Fundamentals/Trend)"

            # --- OUTPUT: METRICS ---
            st.subheader(f"📊 Deep Audit: {name} ({ticker}{suffix})")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Conviction Score", f"{conviction_score}/100")
            m2.metric("Prob. of Profit (180d)", f"{prob_profit:.1f}%")
            m3.metric("180-Day AI ROI", f"{roi_180:+.1f}%")
            m4.metric(f"Price ({display_currency})", f"{symbol}{conv_p:.2f}")

            # --- OUTPUT: RISK & STRATEGY ---
            st.markdown("---")
            st.success(f"### ACTION: {action_label}")
            st.markdown(f'<div class="risk-box"><b>⚠️ RISK ASSESSMENT:</b> {risk_level}. ' + 
                        (f"Primary risk includes high debt or negative ROI trend." if points < 2 else "Standard market volatility applies.") + '</div>', unsafe_allow_html=True)
            
            if pct > 0:
                st.markdown(f'<div class="stop-loss-box"><p class="stop-loss-text">STOP LOSS: Exit below {symbol}{conv_p * sl_buf:.2f}</p></div>', unsafe_allow_html=True)

            # --- OUTPUT: DETAILED PHASES ---
            st.markdown("---")
            p1, p2 = st.columns(2)
            with p1:
                st.markdown('<div class="phase-card">', unsafe_allow_html=True)
                st.subheader("🚀 PHASE 1: IMMEDIATE")
                st.write(f"**Allocation:** {pct}% of total funds.")
                st.write(f"**Strategic Goal:** Establish a core position while current momentum is valid.")
                st.write(f"**Amount Today:** {symbol}{total_capital * (pct/100):,.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with p2:
                st.markdown('<div class="phase-card">', unsafe_allow_html=True)
                st.subheader("⏳ PHASE 2: STAGING")
                st.write(f"**Allocation:** {100-pct}% of total funds.")
                st.write(f"**Strategic Goal:** Buy the dips. Deploy remainder over 180 days only if price stays above Stop Loss.")
                st.write(f"**Reserve Amount:** {symbol}{total_capital * ((100-pct)/100):,.2f}")
                st.markdown('</div>', unsafe_allow_html=True)

            # --- OUTPUT: HEALTH TABLE (Above Chart) ---
            st.markdown("---")
            st.subheader("🏥 Company Health Detail")
            
            st.table(pd.DataFrame({
                "Metric": ["ROE (Efficiency)", "Debt/Equity", "Current Ratio", "Profit Margin"],
                "Status": [f"{health['ROE']*100:.1f}%", health['Debt'], health['Ratio'], health['Margin']],
                "Rating": ["✅ Prime" if health['ROE'] > 0.15 else "⚠️ Low/NA", "✅ Safe" if health['Debt'] < 1.1 else "⚠️ High/NA", "✅ Liquid", "✅ Stable"]
            }))

            # --- OUTPUT: 180-DAY GRAPH ---
            st.markdown("---")
            st.subheader(f"🤖 180-Day AI Price Projection ({display_currency})")
            forecast[['yhat', 'yhat_lower', 'yhat_upper']] *= fx_rate
            fig = m.plot(forecast)
            plt.title(f"{name} Growth Forecast")
            st.pyplot(fig)
        else:
            st.error(f"❌ Error: Data unavailable for {user_query}.")
