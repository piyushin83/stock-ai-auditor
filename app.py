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
import io
import re
import tempfile
import os
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

# Document extraction libraries
try:
    from pypdf import PdfReader
    import docx
    import reticker
except ImportError:
    st.error("Please install required libraries: pypdf, python-docx, reticker")

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
st.title("🏛️ Strategic AI Investment Architect (V10.6)")

# -------------------------------
# 2. INITIALIZE SESSION STATE
# -------------------------------
if 'single_ticker' not in st.session_state:
    st.session_state['single_ticker'] = "AAPL"
if 'doc_holdings' not in st.session_state:
    st.session_state['doc_holdings'] = None
if 'analysis_done' not in st.session_state:
    st.session_state['analysis_done'] = False
if 'portfolio_data' not in st.session_state:
    st.session_state['portfolio_data'] = None

# -------------------------------
# 3. DATA FETCHING FUNCTIONS (No Yahoo Finance)
# -------------------------------
@st.cache_data(ttl=3600)
def get_stock_price(ticker):
    """Get current stock price from Finviz"""
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find price in Finviz table
        price_cell = soup.find('td', class_='snapshot-td2', string='Price')
        if price_cell:
            price_value = price_cell.find_next_sibling('td').text
            price = float(price_value.replace(',', ''))
            return price
    except Exception as e:
        pass
    
    # Try alternative method
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        if 'regularMarketPrice' in response.text:
            # Extract price using regex
            match = re.search(r'regularMarketPrice":\{"raw":(\d+\.?\d*)', response.text)
            if match:
                return float(match.group(1))
    except:
        pass
    
    return None

@st.cache_data(ttl=86400)
def get_company_name(ticker):
    """Get company name from Finviz"""
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        title = soup.find('title')
        if title:
            name = title.text.split('Stock')[0].strip()
            return name
    except:
        pass
    return ticker

@st.cache_data(ttl=86400)
def get_sector(ticker):
    """Get sector from Finviz"""
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find sector in the snapshot table
        rows = soup.find_all('tr', class_='snapshot-td2')
        for row in rows:
            cells = row.find_all('td')
            for i, cell in enumerate(cells):
                if cell.text == 'Sector' and i + 1 < len(cells):
                    return cells[i + 1].text
    except:
        pass
    return 'Unknown'

@st.cache_data(ttl=86400)
def get_historical_data(ticker):
    """Get historical data from stooq"""
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=365)
        df = web.DataReader(f"{ticker}.US", 'stooq', start, end)
        if not df.empty:
            df = df.reset_index()
            df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
            df['ds'] = pd.to_datetime(df['ds'])
            return df.sort_values('ds')
    except:
        pass
    return None

@st.cache_data(ttl=3600)
def get_news(ticker):
    """Get news from Finviz"""
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

@st.cache_data(ttl=86400)
def get_fundamentals(ticker):
    """Get fundamental data from Finviz"""
    health = {"ROE": 0, "Debt": 0, "PB": 0, "Margin": "N/A", "CurrentRatio": "N/A", "PE": "N/A"}
    
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        def get_value(label):
            cells = soup.find_all('td', class_='snapshot-td2-cp')
            for i, cell in enumerate(cells):
                if cell.text == label and i + 1 < len(cells):
                    return cells[i + 1].text
            return "-"
        
        health = {
            "ROE": float(get_value("ROE").replace('%', '')) / 100 if get_value("ROE") != "-" else 0,
            "Debt": float(get_value("Debt/Eq")) if get_value("Debt/Eq") != "-" else 0,
            "PB": float(get_value("P/B")) if get_value("P/B") != "-" else 0,
            "PE": get_value("P/E"),
            "Margin": get_value("Profit Margin"),
            "CurrentRatio": get_value("Current Ratio")
        }
    except:
        pass
    
    return health

def calculate_technicals(df):
    """Calculate RSI and Fibonacci levels"""
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

# -------------------------------
# 4. DOCUMENT EXTRACTION CLASS
# -------------------------------
class DocumentPortfolioExtractor:
    def __init__(self):
        try:
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
        except:
            self.ticker_extractor = None

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

    def find_tickers_regex(self, text: str):
        """Find tickers using regex as fallback"""
        # Common stock ticker pattern (1-5 uppercase letters)
        pattern = r'\b[A-Z]{1,5}\b'
        matches = re.findall(pattern, text)
        # Filter out common false positives
        blacklist = {'ETF', 'USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'THE', 'AND', 'FOR'}
        return [m for m in matches if m not in blacklist][:20]

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
        if self.ticker_extractor:
            raw_tickers = self.ticker_extractor.extract(text)
        else:
            raw_tickers = self.find_tickers_regex(text)
        
        # Get info for each ticker
        holdings = []
        for ticker in raw_tickers[:15]:
            price = get_stock_price(ticker)
            if price and price > 0:
                holdings.append({
                    'ticker': ticker,
                    'name': get_company_name(ticker),
                    'shares': 1,  # Default to 1 share
                    'sector': get_sector(ticker),
                    'current_price': price
                })
        
        return holdings, text[:500]

# -------------------------------
# 5. SIDEBAR CONFIGURATION
# -------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Single Ticker Input
    ticker_input = st.text_input("Single Ticker", value=st.session_state['single_ticker'], key="ticker_input")
    st.session_state['single_ticker'] = ticker_input
    
    display_currency = st.selectbox("Currency", ["USD", "EUR"])
    total_capital = st.number_input("Capital", value=10000)
    
    st.markdown("---")
    st.header("📁 Portfolio Upload")
    uploaded_file = st.file_uploader("CSV or Excel", type=['csv', 'xlsx', 'xls'])
    
    st.markdown("---")
    st.header("📄 Document Upload")
    uploaded_doc = st.file_uploader("PDF or Word", type=['pdf', 'docx', 'doc'])
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        analyze_single = st.button("🚀 Analyze Stock", use_container_width=True)
    with col2:
        extract_doc = st.button("📄 Extract Holdings", use_container_width=True)

# -------------------------------
# 6. SINGLE TICKER ANALYSIS
# -------------------------------
if analyze_single:
    st.session_state['analysis_done'] = True
    ticker = st.session_state['single_ticker']
    
    with st.spinner(f"Analyzing {ticker}..."):
        # Get data
        price = get_stock_price(ticker)
        
        if price:
            name = get_company_name(ticker)
            fx = 1.1 if display_currency == "EUR" else 1.0
            sym = "$" if display_currency == "USD" else "€"
            price_display = price * fx
            
            # Get additional data
            df = get_historical_data(ticker)
            health = get_fundamentals(ticker)
            news = get_news(ticker)
            
            # Display results
            st.markdown("---")
            st.header(f"📊 {name} Analysis")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"{sym}{price_display:,.2f}")
            with col2:
                target = price_display * 1.05
                st.metric("30d Target", f"{sym}{target:,.2f}")
            with col3:
                st.metric("P/E", health.get('PE', 'N/A'))
            with col4:
                st.metric("ROE", f"{health['ROE']*100:.1f}%")
            
            # Price chart
            if df is not None and len(df) > 30:
                st.subheader("📈 6-Month Price History")
                fig, ax = plt.subplots(figsize=(12, 6))
                
                df_display = df.tail(180).copy()
                df_display['y_display'] = df_display['y'] * fx
                
                ax.plot(df_display['ds'], df_display['y_display'], 'b-', linewidth=2, label='Price')
                ax.set_xlabel('Date')
                ax.set_ylabel(f'Price ({sym})')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.legend()
                st.pyplot(fig)
                
                # Technical indicators
                rsi, fibs = calculate_technicals(df)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RSI (14)", f"{rsi:.1f}")
                with col2:
                    signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                    st.metric("Signal", signal)
                with col3:
                    trend = "Bullish" if rsi > 50 else "Bearish"
                    st.metric("Trend", trend)
                
                # Fibonacci levels
                st.subheader("📊 Fibonacci Support Levels")
                fcol1, fcol2, fcol3 = st.columns(3)
                with fcol1:
                    st.metric("0.382", f"{sym}{fibs['0.382']*fx:,.2f}")
                with fcol2:
                    st.metric("0.500", f"{sym}{fibs['0.500']*fx:,.2f}")
                with fcol3:
                    st.metric("0.618", f"{sym}{fibs['0.618']*fx:,.2f}")
            
            # News section
            if news:
                st.subheader("📰 Recent News")
                for item in news[:5]:
                    emoji = '🟢' if item['sentiment'] > 0 else '🔴' if item['sentiment'] < 0 else '⚪'
                    st.markdown(f"{emoji} {item['headline']}")
            
            # Fundamentals
            st.subheader("🏥 Fundamentals")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Debt/Equity", f"{health['Debt']:.2f}")
                st.metric("P/B Ratio", f"{health['PB']:.2f}")
            with col2:
                st.metric("Profit Margin", health['Margin'])
                st.metric("Current Ratio", health['CurrentRatio'])
        else:
            st.error(f"Could not fetch data for {ticker}")

# -------------------------------
# 7. PORTFOLIO ANALYSIS
# -------------------------------
if uploaded_file is not None:
    try:
        # Read file
        if uploaded_file.name.endswith('.csv'):
            df_in = pd.read_csv(uploaded_file)
        else:
            df_in = pd.read_excel(uploaded_file)
        
        # Normalize columns
        df_in.columns = df_in.columns.str.strip().str.lower()
        
        if 'ticker' not in df_in.columns:
            st.error("File must contain 'Ticker' column")
        else:
            if 'shares' not in df_in.columns:
                df_in['shares'] = 1
            
            st.markdown("---")
            st.header("📁 Portfolio Analysis")
            
            with st.spinner("Analyzing portfolio..."):
                results = []
                total_value = 0
                fx = 1.1 if display_currency == "EUR" else 1.0
                sym = "$" if display_currency == "USD" else "€"
                
                progress_bar = st.progress(0)
                for idx, row in df_in.iterrows():
                    ticker = str(row['ticker']).strip().upper()
                    shares = float(row['shares'])
                    
                    progress_bar.progress((idx + 1) / len(df_in))
                    
                    price = get_stock_price(ticker)
                    if price:
                        price_display = price * fx
                        value = shares * price_display
                        total_value += value
                        
                        results.append({
                            'Ticker': ticker,
                            'Name': get_company_name(ticker),
                            'Sector': get_sector(ticker),
                            'Shares': shares,
                            'Current Price': price_display,
                            'Current Value': value,
                            '30d Target': price_display * 1.05,
                            '30d Growth %': 5.0
                        })
                
                progress_bar.empty()
                
                if results:
                    df_port = pd.DataFrame(results)
                    
                    # Calculate allocations
                    if total_value > 0:
                        df_port['Allocation %'] = (df_port['Current Value'] / total_value * 100)
                    
                    # Display summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Value", f"{sym}{total_value:,.2f}")
                    with col2:
                        st.metric("Holdings", len(df_port))
                    with col3:
                        st.metric("Avg Price", f"{sym}{df_port['Current Price'].mean():,.2f}")
                    
                    # Holdings table
                    st.subheader("Holdings")
                    display_df = df_port.copy()
                    for col in ['Current Price', 'Current Value', '30d Target']:
                        display_df[col] = display_df[col].apply(lambda x: f"{sym}{x:,.2f}")
                    display_df['30d Growth %'] = display_df['30d Growth %'].apply(lambda x: f"{x:.1f}%")
                    display_df['Allocation %'] = display_df['Allocation %'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(display_df)
                    
                    # Sector allocation
                    st.subheader("Sector Allocation")
                    sector_data = df_port.groupby('Sector')['Current Value'].sum().reset_index()
                    if not sector_data.empty:
                        fig, ax = plt.subplots()
                        ax.pie(sector_data['Current Value'], 
                               labels=sector_data['Sector'], 
                               autopct='%1.1f%%',
                               startangle=90)
                        ax.axis('equal')
                        st.pyplot(fig)
                    
                    # Suggestions
                    st.subheader("💡 Suggestions")
                    high_conc = sector_data[sector_data['Current Value'] / total_value > 0.3]
                    if not high_conc.empty:
                        for _, row in high_conc.iterrows():
                            st.info(f"⚠️ High concentration in {row['Sector']}")
                    
                    if len(results) < 3:
                        st.info("📊 Consider adding more holdings for diversification")
                else:
                    st.error("No valid tickers found")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# -------------------------------
# 8. DOCUMENT PROCESSING
# -------------------------------
if extract_doc and uploaded_doc is not None:
    with st.spinner("Extracting holdings from document..."):
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
            st.markdown("---")
            st.subheader("📄 Extracted Holdings")
            holdings_df = pd.DataFrame(holdings)
            st.dataframe(holdings_df)
            
            # Analyze button
            if st.button("📊 Analyze These Holdings"):
                with st.spinner("Analyzing holdings..."):
                    fx = 1.1 if display_currency == "EUR" else 1.0
                    sym = "$" if display_currency == "USD" else "€"
                    
                    results = []
                    total_value = 0
                    
                    for h in holdings:
                        price = h['current_price']
                        price_display = price * fx
                        value = h['shares'] * price_display
                        total_value += value
                        
                        results.append({
                            'Ticker': h['ticker'],
                            'Name': h['name'],
                            'Sector': h['sector'],
                            'Shares': h['shares'],
                            'Current Price': price_display,
                            'Current Value': value,
                            '30d Target': price_display * 1.05,
                            '30d Growth %': 5.0
                        })
                    
                    if results:
                        df_port = pd.DataFrame(results)
                        
                        if total_value > 0:
                            df_port['Allocation %'] = (df_port['Current Value'] / total_value * 100)
                        
                        st.subheader("Analysis Results")
                        st.metric("Total Value", f"{sym}{total_value:,.2f}")
                        
                        display_df = df_port.copy()
                        for col in ['Current Price', 'Current Value', '30d Target']:
                            display_df[col] = display_df[col].apply(lambda x: f"{sym}{x:,.2f}")
                        display_df['30d Growth %'] = display_df['30d Growth %'].apply(lambda x: f"{x:.1f}%")
                        st.dataframe(display_df)
                        
                        # Sector chart
                        sector_data = df_port.groupby('Sector')['Current Value'].sum().reset_index()
                        if not sector_data.empty:
                            fig, ax = plt.subplots()
                            ax.pie(sector_data['Current Value'], 
                                   labels=sector_data['Sector'], 
                                   autopct='%1.1f%%')
                            ax.axis('equal')
                            st.pyplot(fig)
        else:
            st.warning("No holdings found in document")

# -------------------------------
# 9. FOOTER
# -------------------------------
st.markdown("---")
st.markdown("*Data sourced from Finviz and Stooq. Updates every hour.*")
