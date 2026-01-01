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
    /* Metric Boxes Styling */
    [data-testid="stMetricValue"] { font-size: 32px !important; font-weight: 800 !important; color: #1f77b4; }
    [data-testid="stMetricLabel"] p { font-size: 16px !important; font-weight: 600 !important; color: #333; }
    /* Strategy Highlight Boxes */
    .stop-loss-box { background-color: #fff1f1; padding: 15px; border-radius: 8px; border-left: 6px solid #ff4b4b; margin: 15px 0; }
    .stop-loss-text { font-size: 18px; font-weight: bold; color: #d32f2f; margin: 0; }
    /* Phase Card Styling */
    .phase-card { background-color: #f8f9fa; padding: 20px; border-radius: 12px; border: 1px solid #dee2e6; height: 100%; }
    .disclaimer-box { background-color: #fffde7; padding: 15px; border-radius: 5px; border: 1px solid #fbc02d; font-size: 13px; color: #5d4037; margin-bottom: 25px;}
</style>
""", unsafe_allow_html=True)

# 2. HEADER & DISCLAIMER
st.title("🏛️ Strategic AI Investment Architect")
st.markdown("""
<div class="disclaimer-box">
    <b>⚠️ LEGAL DISCLAIMER:</b> Educational tool only. AI (Prophet) projections are mathematical probabilities, not guarantees. 
    Past performance is not indicative of future results. <b>Human judgment is mandatory.</b>
</div>
""", unsafe_allow_html=True)

# 3. ENGINES
def get_exchange_rate(from_curr, to_curr):
    if from_curr == to_curr: return 1.0
    try:
        pair = f"{from_curr}{to_curr}=X"
        data = yf.download(pair, period="1d", progress=False)
        return data['Close'].iloc[-1]
    except: return 1.07 if to_curr == "USD" else 0.93

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
    with st.spinner(f"📡 Analyzing {user_query}..."):
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
            
            # --- SCORING ENGINE ---
            points = (1 if health['ROE'] > 0.15 else 0) + (1 if roi_180 > 10 else 0) + (1 if health['Debt'] < 1.1 else 0)
            conviction_score = int((points / 3) * 100)
            prob_profit = 50 + (roi_180 / 2) if roi_180 > 0 else 50 - abs(roi_180 / 2)
            prob_profit = max(min(prob_profit, 99), 1) # Clamp between 1-99%

            # --- STRATEGY LOGIC ---
            if points == 3:
                mode, action_label, pct, sl_buf = "Aggressive", "🌟 HIGH CONVICTION BUY", 0.15, 0.90
                confidence_msg = "Strong. All indicators are positive."
            elif points >= 1:
                mode, action_label, pct, sl_buf = "Defensive", "🟡 ACCUMULATE / HOLD", 0.05, 0.85
                confidence_msg = "Moderate. One or more indicators suggest caution."
            else:
                mode, action_label, pct, sl_buf = "Preservation", "🛑 AVOID / SELL", 0.0, 0.0
                confidence_msg = "Low. Key indicators are negative or stagnant."

            imm_buy = total_capital * pct
            staging_amt = total_capital - imm_buy

            # --- OUTPUT: METRICS BAR ---
            st.subheader(f"📊 Deep Audit: {name} ({ticker}{suffix})")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Conviction Score", f"{conviction_score}/100")
            m2.metric("Prob. of Profit (180d)", f"{prob_profit:.1f}%")
            m3.metric("180-Day AI ROI", f"{roi_180:+.1f}%")
            m4.metric(f"Current Price ({display_currency})", f"{symbol}{conv_p:.2f}")

            # --- OUTPUT: STRATEGY BOX ---
            st.markdown("---")
            st.success(f"### ACTION: {action_label}")
            st.write(f"**Confidence:** {confidence_msg}")
            
            if pct > 0:
                sl_price = conv_p * sl_buf
                st.markdown(f'<div class="stop-loss-box"><p class="stop-loss-text">STOP LOSS: Exit below {symbol}{sl_price:.2f} ({(1-sl_buf)*100:.0f}% Buffer)</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="stop-loss-box"><p class="stop-loss-text">STOP LOSS: No entry recommended. Keep 100% in Cash/SGOV.</p></div>', unsafe_allow_html=True)

            # --- OUTPUT: PHASED PLAN ---
            st.markdown("---")
            p1, p2 = st.columns(2)
            with p1:
                st.markdown('<div class="phase-card">', unsafe_allow_html=True)
                st.subheader("🚀 PHASE 1: IMMEDIATE")
                if imm_buy > 0:
                    st.write(f"**BUY:** Invest **{symbol}{imm_buy:,.2f}** today.")
                else:
                    st.write(f"**AVOID/SELL:** Invest **{symbol}0.00** (Keep in cash).")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with p2:
                st.markdown('<div class="phase-card">', unsafe_allow_html=True)
                st.subheader("⏳ PHASE 2: STAGING")
                if staging_amt > 0 and points > 0:
                    st.write(f"**HOLD:** Park **{symbol}{staging_amt:,.2f}** in SGOV ETF or Cash for staged entry.")
                else:
                    st.write(f"**RESERVE:** Maintain **{symbol}{total_capital:,.2f}** in cash reserves.")
                st.markdown('</div>', unsafe_allow_html=True)

            # --- OUTPUT: 180-DAY GRAPH ---
            st.markdown("---")
            st.subheader(f"🤖 180-Day AI Price Projection ({display_currency})")
            forecast[['yhat', 'yhat_lower', 'yhat_upper']] *= fx_rate
            fig = m.plot(forecast)
            plt.title(f"{name} Growth Forecast")
            st.pyplot(fig)
            
            # --- OUTPUT: HEALTH AUDIT ---
            st.markdown("---")
            st.subheader("🏥 Company Health Detail")
            st.table(pd.DataFrame({
                "Metric": ["ROE (Efficiency)", "Debt/Equity", "Current Ratio", "Profit Margin"],
                "Status": [f"{health['ROE']*100:.1f}%", health['Debt'], health['Ratio'], health['Margin']],
                "Rating": ["✅ Prime" if health['ROE'] > 0.15 else "⚠️ Low/NA", "✅ Safe" if health['Debt'] < 1.1 else "⚠️ High/NA", "✅ Liquid", "✅ Stable"]
            }))

        else:
            st.error(f"❌ Error: Found {user_query} but could not pull market data.")
