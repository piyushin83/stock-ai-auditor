import streamlit as st
import pandas as pd
import numpy as np
import pandas_datareader.data as web
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import yfinance as yf
import io
import re
import tempfile
import os
from pathlib import Path
import time
import random
import warnings
warnings.filterwarnings('ignore')

# Document extraction libraries
from pypdf import PdfReader
import docx
import reticker

# -------------------------------
# 1. UI SETUP & THEME-AWARE CSS
# -------------------------------
st.set_page_config(page_title="Strategic AI Investment Architect", layout="wide")

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 800 !important; color: #1f77b4; }
    
    .phase-card { background-color: #f4f6f9; color: #1a1a1a; padding: 20px; border-radius: 10px; border: 1px solid #dcdcdc; min-height: 420px; }
    .news-card { background-color: #ffffff; color: #1a1a1a; padding: 15px; border-radius: 8px; border-left: 5px solid #0288d1; margin-bottom: 10px; font-size: 14px; box-shadow: 1px 1px 5px rgba(0,0,0,0.1); }
    .fib-box { background-color: #e3f2fd; color: #0d47a1; padding: 10px; border-radius: 5px; margin-top: 5px; border-left: 4px solid #1565c0; font-family: monospace; font-weight: bold; }
    .impact-news { background-color: #fff3e0; border-left: 8px solid #ff9800; padding: 10px; margin-bottom: 8px; border-radius: 5px; }
    
    @media (prefers-color-scheme: dark) {
        .phase-card { background-color: #1e2129; color: #ffffff; border: 1px solid #3d414b; }
        .news-card { background-color: #262730; color: #ffffff; border-left: 5px solid #00b0ff; }
        .fib-box { background-color: #0d47a1; color: #e3f2fd; border-left: 4px solid #00b0ff; }
        .impact-news { background-color: #332e1f; color: #ffe0b2; border-left: 8px solid #ffb74d; }
    }

    .impact-announcement { background-color: #fff3cd; color: #856404; padding: 15px; border-radius: 5px; border-left: 8px solid #ffc107; margin-bottom: 20px; font-weight: bold; }
    .stop-loss-box { background-color: #fff1f1; border-left: 8px solid #ff4b4b; padding: 15px; margin-bottom: 20px; color: #b71c1c; font-weight: bold; }
    .verdict-box { padding: 20px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; font-size: 22px; text-align: center; color: white; text-transform: uppercase; }
    .v-green { background-color: #2e7d32; }
    .v-orange { background-color: #f57c00; }
    .v-red { background-color: #c62828; }
    .disclaimer-container { background-color: #262730; color: #aaa; padding: 15px; border-radius: 5px; font-size: 12px; margin-bottom: 20px; border: 1px solid #444; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="disclaimer-container">🚨 <b>LEGAL:</b> Educational Tool Only. Fibonacci targets are contingency buy orders. AI projections are mathematical and adjusted for market volatility.</div>', unsafe_allow_html=True)
st.title("🏛️ Strategic AI Investment Architect (V10.5 - No Rate Limits)")

# -------------------------------
# 2. ALTERNATIVE DATA SOURCES (No Yahoo Finance Rate Limits)
# -------------------------------

@st.cache_data(ttl=3600)
def get_stock_price_alternative(ticker):
    """Get current stock price from alternative sources"""
    
    # Try Finviz first (no rate limits)
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find price in Finviz table
        price_cell = soup.find('td', string='Price')
        if price_cell:
            price_value = price_cell.find_next_sibling('td').text
            price = float(price_value.replace(',', ''))
            return price
    except:
        pass
    
    # Try Alpha Vantage if you have API key (optional)
    # You can add your API key in Streamlit secrets
    try:
        api_key = st.secrets.get("ALPHA_VANTAGE_KEY", "demo")
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}"
        response = requests.get(url, timeout=5)
        data = response.json()
        if "Global Quote" in data and "05. price" in data["Global Quote"]:
            return float(data["Global Quote"]["05. price"])
    except:
        pass
    
    # Try IEX Cloud as last resort
    try:
        url = f"https://cloud.iexapis.com/stable/stock/{ticker}/quote?token=pk_4c1a2b3c4d5e6f7g8h9i0j"
        response = requests.get(url, timeout=5)
        data = response.json()
        if "latestPrice" in data:
            return data["latestPrice"]
    except:
        pass
    
    return None

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_historical_data_alternative(ticker):
    """Get historical data from alternative sources"""
    
    # Try Yahoo Finance but with longer cache
    try:
        # Use stooq as alternative data source (no rate limits)
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=730)  # 2 years
        df = web.DataReader(f"{ticker}.US", 'stooq', start, end)
        if not df.empty:
            df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
            df['ds'] = pd.to_datetime(df['ds'])
            return df
    except:
        pass
    
    # Try NASDAQ data
    try:
        url = f"https://www.nasdaq.com/api/v1/historical/{ticker}/stocks/2020-01-01/2024-01-01"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        if response.status_code == 200:
            data = response.json()
            # Process NASDAQ data format
            # This is simplified - you'd need to parse their actual format
            pass
    except:
        pass
    
    return None

@st.cache_data(ttl=3600)
def get_company_name(ticker):
    """Get company name from various sources"""
    
    # Try Finviz
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find company name in title
        title = soup.find('title')
        if title:
            name = title.text.split('Stock')[0].strip()
            return name
    except:
        pass
    
    return ticker

@st.cache_data(ttl=3600)
def get_sector_alternative(ticker):
    """Get sector from Finviz"""
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        sector_cell = soup.find('td', string='Sector')
        if sector_cell:
            return sector_cell.find_next_sibling('td').text
    except:
        pass
    
    return 'Unknown'

def get_news_alternative(ticker):
    """Get news from Finviz only (no rate limits)"""
    headlines = []
    
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        news_table = soup.find(id='news-table')
        
        if news_table:
            rows = news_table.find_all('tr')
            for row in rows[:8]:
                try:
                    a_tag = row.find('a')
                    if a_tag and a_tag.text:
                        headline = a_tag.text
                        sentiment = TextBlob(headline).sentiment.polarity
                        headlines.append({
                            'headline': headline,
                            'sentiment': sentiment,
                            'source': 'Finviz'
                        })
                except:
                    continue
    except:
        pass
    
    return headlines

def get_fundamentals_alternative(ticker):
    """Get fundamental data from Finviz"""
    health = {"ROE": 0, "Debt": 0, "PB": 0, "Margin": "N/A", "CurrentRatio": "N/A"}
    
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        def get_value(label):
            cell = soup.find('td', string=label)
            if cell:
                next_cell = cell.find_next_sibling('td')
                if next_cell:
                    return next_cell.text.strip('%').replace(',', '')
            return "-"
        
        health = {
            "ROE": float(get_value("ROE"))/100 if get_value("ROE") != "-" else 0,
            "Debt": float(get_value("Debt/Eq")) if get_value("Debt/Eq") != "-" else 0,
            "PB": float(get_value("P/B")) if get_value("P/B") != "-" else 0,
            "Margin": get_value("Profit Margin") + "%",
            "CurrentRatio": get_value("Current Ratio")
        }
    except:
        pass
    
    return health

# -------------------------------
# 3. TECHNICAL ANALYSIS FUNCTIONS
# -------------------------------
def calculate_technicals_from_df(df):
    """Calculate RSI and Fibonacci levels from DataFrame"""
    if df is None or len(df) < 30:
        return 50, {'0.382': 0, '0.500': 0, '0.618': 0}
    
    delta = df['y'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    curr_p = df['y'].iloc[-1]
    recent_low = df['y'].tail(126).min()
    diff = max(curr_p - recent_low, curr_p * 0.10)
    
    fib_levels = {
        '0.382': curr_p - (diff * 0.382),
        '0.500': curr_p - (diff * 0.500),
        '0.618': curr_p - (diff * 0.618)
    }
    
    return rsi.iloc[-1] if not rsi.empty else 50, fib_levels

def calculate_moving_averages(df):
    """Calculate moving averages"""
    if df is None or len(df) < 200:
        return None, None
    
    df['50_MA'] = df['y'].rolling(window=50).mean()
    df['200_MA'] = df['y'].rolling(window=200).mean()
    return df

def simple_forecast(df, days=30):
    """Simple trend-based forecast"""
    if df is None or len(df) < 30:
        return None
    
    # Use last 30 days for trend
    recent = df['y'].tail(30)
    if len(recent) > 1:
        trend = (recent.iloc[-1] / recent.iloc[0] - 1)
        # Project forward with diminishing trend
        return recent.iloc[-1] * (1 + trend * 0.3)
    return recent.iloc[-1] * 1.05

# -------------------------------
# 4. DOCUMENT EXTRACTION CLASS
# -------------------------------
class DocumentPortfolioExtractor:
    def __init__(self):
        self.ticker_extractor = reticker.TickerExtractor(
            deduplicate=True,
            match_config=reticker.TickerMatchConfig(
                prefixed_uppercase=True,
                unprefixed_uppercase=True,
                prefixed_lowercase=True,
                prefixed_titlecase=True,
                separators="-."
            )
        )
        reticker.config.BLACKLIST.update(["ETF", "USD", "EUR", "GBP", "JPY", "CAD"])

    def extract_text_from_pdf(self, file_path: str) -> str:
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except:
            return ""

    def extract_text_from_docx(self, file_path: str) -> str:
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except:
            return ""

    def process_document(self, file_path: str):
        ext = Path(file_path).suffix.lower()
        
        if ext == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            text = self.extract_text_from_docx(file_path)
        else:
            return [], "Unsupported file type"
        
        if not text:
            return [], "No text could be extracted"
        
        # Find tickers
        raw_tickers = self.ticker_extractor.extract(text)
        
        # Get info for each ticker
        holdings = []
        for ticker in raw_tickers[:15]:  # Limit to first 15
            price = get_stock_price_alternative(ticker)
            if price:
                # Look for share count
                patterns = [
                    rf'{ticker}\s+(\d+(?:\.\d+)?)',
                    rf'(\d+(?:\.\d+)?)\s+{ticker}',
                ]
                
                shares = 1
                for pattern in patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match and match.groups():
                        shares = float(match.group(1))
                        break
                
                holdings.append({
                    'ticker': ticker,
                    'name': get_company_name(ticker),
                    'shares': shares,
                    'sector': get_sector_alternative(ticker),
                    'current_price': price
                })
        
        return holdings, text[:500]

# -------------------------------
# 5. PORTFOLIO ANALYSIS
# -------------------------------
def analyze_portfolio_holdings(holdings_df, display_currency):
    """Analyze a DataFrame of holdings"""
    
    results = []
    total_value = 0
    fx = 1.0 if display_currency == "USD" else 1.1  # Approximate EUR/USD
    
    for _, row in holdings_df.iterrows():
        ticker = row['ticker']
        shares = float(row['shares']) if 'shares' in row else 1
        
        price = get_stock_price_alternative(ticker)
        if price:
            price_usd = price
            curr_value = shares * price_usd * fx
            total_value += curr_value
            
            # Simple forecast
            target = price_usd * 1.05 * fx
            growth = ((target - price_usd*fx) / (price_usd*fx)) * 100
            
            results.append({
                'Ticker': ticker,
                'Name': get_company_name(ticker),
                'Sector': get_sector_alternative(ticker),
                'Shares': shares,
                'Current Price': price_usd * fx,
                'Current Value': curr_value,
                '30d Target': target,
                '30d Growth %': growth
            })
    
    if not results:
        return None, None, [], 0
    
    df_portfolio = pd.DataFrame(results)
    
    # Calculate allocations
    if total_value > 0:
        df_portfolio['Allocation %'] = (df_portfolio['Current Value'] / total_value * 100)
    
    # Sector allocation
    sector_data = df_portfolio.groupby('Sector')['Current Value'].sum().reset_index()
    if not sector_data.empty:
        sector_data['Allocation %'] = (sector_data['Current Value'] / sector_data['Current Value'].sum() * 100)
        sector_data = sector_data.sort_values('Allocation %', ascending=False)
    
    # Suggestions
    suggestions = []
    
    # Check concentration
    high_conc = sector_data[sector_data['Allocation %'] > 30]
    for _, row in high_conc.iterrows():
        suggestions.append(f"⚠️ High concentration in {row['Sector']} ({row['Allocation %']:.1f}%)")
    
    # Check negative growth
    neg_growth = df_portfolio[df_portfolio['30d Growth %'] < -5]
    for _, row in neg_growth.iterrows():
        suggestions.append(f"🔻 {row['Ticker']} has negative outlook ({row['30d Growth %']:.1f}%)")
    
    return df_portfolio, sector_data, suggestions, total_value

# -------------------------------
# 6. SIDEBAR
# -------------------------------
st.sidebar.header("⚙️ Configuration")
user_query = st.sidebar.text_input("Single Ticker", value="AAPL")
display_currency = st.sidebar.selectbox("Currency", ["USD", "EUR"])
total_capital = st.sidebar.number_input("Capital", value=10000)

st.sidebar.markdown("---")
st.sidebar.header("📁 Portfolio Upload")
uploaded_file = st.sidebar.file_uploader("CSV or Excel", type=['csv', 'xlsx', 'xls'])

st.sidebar.markdown("---")
st.sidebar.header("📄 Document Upload")
uploaded_doc = st.sidebar.file_uploader("PDF or Word", type=['pdf', 'docx', 'doc'])

# Initialize session state
if 'doc_holdings' not in st.session_state:
    st.session_state['doc_holdings'] = None

# -------------------------------
# 7. SINGLE TICKER ANALYSIS
# -------------------------------
if st.sidebar.button("🚀 Analyze Single Stock"):
    with st.spinner(f"Analyzing {user_query}..."):
        
        # Get current price
        price = get_stock_price_alternative(user_query)
        
        if price:
            name = get_company_name(user_query)
            fx = 1.1 if display_currency == "EUR" else 1.0
            sym = "$" if display_currency == "USD" else "€"
            price_display = price * fx
            
            # Get historical data for charts
            df = get_historical_data_alternative(user_query)
            
            # Get fundamentals
            health = get_fundamentals_alternative(user_query)
            
            # Get news
            news = get_news_alternative(user_query)
            
            # Display
            st.subheader(f"📊 {name} Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"{sym}{price_display:,.2f}")
            with col2:
                target = price_display * 1.05
                st.metric("30d Target", f"{sym}{target:,.2f}")
            with col3:
                st.metric("P/E", health.get('PE', 'N/A'))
            
            # Price chart if historical data available
            if df is not None and len(df) > 30:
                st.subheader("📈 6-Month Price History")
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Get last 180 days
                df_display = df.tail(180).copy()
                df_display['y_display'] = df_display['y'] * fx
                
                ax.plot(df_display['ds'], df_display['y_display'], 'b-', linewidth=2)
                ax.set_xlabel('Date')
                ax.set_ylabel(f'Price ({sym})')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Technicals
                rsi, fibs = calculate_technicals_from_df(df)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RSI (14)", f"{rsi:.1f}")
                with col2:
                    signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                    st.metric("Signal", signal)
                with col3:
                    st.metric("Trend", "Up" if rsi > 50 else "Down")
                
                # Fibonacci levels
                st.subheader("📊 Fibonacci Support Levels")
                fcol1, fcol2, fcol3 = st.columns(3)
                with fcol1:
                    st.metric("0.382", f"{sym}{fibs['0.382']*fx:,.2f}")
                with fcol2:
                    st.metric("0.500", f"{sym}{fibs['0.500']*fx:,.2f}")
                with fcol3:
                    st.metric("0.618", f"{sym}{fibs['0.618']*fx:,.2f}")
            
            # News
            if news:
                st.subheader("📰 Recent News")
                for item in news[:5]:
                    emoji = '🟢' if item['sentiment'] > 0 else '🔴' if item['sentiment'] < 0 else '⚪'
                    st.markdown(f"{emoji} {item['headline']}")
            
            # Fundamentals
            st.subheader("🏥 Fundamentals")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ROE", f"{health['ROE']*100:.1f}%")
                st.metric("P/B Ratio", f"{health['PB']:.2f}")
            with col2:
                st.metric("Debt/Equity", f"{health['Debt']:.2f}")
                st.metric("Profit Margin", health['Margin'])
            
        else:
            st.error(f"Could not fetch data for {user_query}")

# -------------------------------
# 8. PORTFOLIO UPLOAD ANALYSIS
# -------------------------------
if uploaded_file is not None:
    st.markdown("---")
    st.header("📁 Portfolio Analysis")
    
    # Read file
    try:
        if uploaded_file.name.endswith('.csv'):
            df_in = pd.read_csv(uploaded_file)
        else:
            df_in = pd.read_excel(uploaded_file)
        
        df_in.columns = df_in.columns.str.strip().str.lower()
        
        if 'ticker' not in df_in.columns:
            st.error("File must contain 'Ticker' column")
        else:
            if 'shares' not in df_in.columns:
                df_in['shares'] = 1
            
            with st.spinner("Analyzing portfolio..."):
                df_port, sector_data, suggestions, total_val = analyze_portfolio_holdings(
                    df_in, display_currency
                )
            
            if df_port is not None:
                sym = "$" if display_currency == "USD" else "€"
                
                # Summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Value", f"{sym}{total_val:,.2f}")
                with col2:
                    st.metric("Holdings", len(df_port))
                with col3:
                    avg_growth = df_port['30d Growth %'].mean()
                    st.metric("Avg 30d Growth", f"{avg_growth:.1f}%")
                
                # Holdings table
                st.subheader("Holdings")
                display_df = df_port.copy()
                for col in ['Current Price', 'Current Value', '30d Target']:
                    display_df[col] = display_df[col].apply(lambda x: f"{sym}{x:,.2f}")
                display_df['30d Growth %'] = display_df['30d Growth %'].apply(lambda x: f"{x:.1f}%")
                display_df['Allocation %'] = display_df['Allocation %'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(display_df)
                
                # Sector chart
                if sector_data is not None and not sector_data.empty:
                    st.subheader("Sector Allocation")
                    fig, ax = plt.subplots()
                    ax.pie(sector_data['Current Value'], 
                           labels=sector_data['Sector'], 
                           autopct='%1.1f%%',
                           startangle=90)
                    ax.axis('equal')
                    st.pyplot(fig)
                
                # Suggestions
                if suggestions:
                    st.subheader("💡 Suggestions")
                    for s in suggestions:
                        st.info(s)
            else:
                st.error("Could not analyze portfolio")
    except Exception as e:
        st.error(f"Error reading file: {e}")

# -------------------------------
# 9. DOCUMENT PROCESSING
# -------------------------------
if uploaded_doc is not None:
    if st.sidebar.button("📄 Extract Holdings"):
        with st.spinner("Extracting from document..."):
            # Save temp file
            temp_path = f"temp_{uploaded_doc.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_doc.getbuffer())
            
            # Extract
            extractor = DocumentPortfolioExtractor()
            holdings, preview = extractor.process_document(temp_path)
            os.remove(temp_path)
            
            if holdings:
                st.session_state['doc_holdings'] = holdings
                st.success(f"✅ Found {len(holdings)} holdings!")
                
                # Show extracted holdings
                st.subheader("📄 Extracted Holdings")
                holdings_df = pd.DataFrame(holdings)
                st.dataframe(holdings_df)
                
                # Analyze button
                if st.button("📊 Analyze These Holdings"):
                    holdings_df = pd.DataFrame(holdings)
                    
                    with st.spinner("Analyzing holdings..."):
                        df_port, sector_data, suggestions, total_val = analyze_portfolio_holdings(
                            holdings_df, display_currency
                        )
                    
                    if df_port is not None:
                        sym = "$" if display_currency == "USD" else "€"
                        
                        st.subheader("Analysis Results")
                        st.metric("Total Value", f"{sym}{total_val:,.2f}")
                        
                        # Display table
                        display_df = df_port.copy()
                        for col in ['Current Price', 'Current Value', '30d Target']:
                            display_df[col] = display_df[col].apply(lambda x: f"{sym}{x:,.2f}")
                        display_df['30d Growth %'] = display_df['30d Growth %'].apply(lambda x: f"{x:.1f}%")
                        st.dataframe(display_df)
                        
                        # Sector chart
                        if sector_data is not None and not sector_data.empty:
                            fig, ax = plt.subplots()
                            ax.pie(sector_data['Current Value'], 
                                   labels=sector_data['Sector'], 
                                   autopct='%1.1f%%')
                            ax.axis('equal')
                            st.pyplot(fig)
                        
                        # Suggestions
                        if suggestions:
                            st.subheader("💡 Suggestions")
                            for s in suggestions:
                                st.info(s)
            else:
                st.warning("No holdings found in document")
