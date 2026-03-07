"""
app.py  —  Strategic AI Investment Architect (V11.5 — SECURED)
══════════════════════════════════════════════════════════════════
Security layers applied:
  1. Password gate        — auth.py blocks all access until login
  2. Secrets management   — all keys read from st.secrets / .env, never hardcoded
  3. UI hardening         — source-view menu, footer, toolbar hidden from browser
  4. Temp file cleanup    — uploaded files deleted immediately after processing
  5. Error sanitisation   — tracebacks hidden from end users in production
══════════════════════════════════════════════════════════════════
"""

# ── LAYER 1: Password gate — must be first import & call ─────────────────────
from auth import require_auth
require_auth()
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import pandas_datareader.data as web
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import yfinance as yf
import time
import random
import io
import os
from pathlib import Path
import re
import traceback
import warnings
warnings.filterwarnings('ignore')

# ── LAYER 3: Hide Streamlit UI elements that expose internals ─────────────────
# This CSS hides: top-right hamburger menu, "Made with Streamlit" footer,
# the toolbar with Rerun/Settings buttons, and the "View app source" link.
_HIDE_ST_STYLE = """
<style>
    #MainMenu                        { visibility: hidden; }
    footer                           { visibility: hidden; }
    header                           { visibility: hidden; }
    [data-testid="stToolbar"]        { visibility: hidden; }
    [data-testid="stDecoration"]     { display: none; }
    [data-testid="stStatusWidget"]   { visibility: hidden; }
    .viewerBadge_container__1QSob   { display: none !important; }
    .stDeployButton                  { display: none !important; }
</style>
"""

# -------------------------------
# 0. CHECK DEPENDENCIES
# -------------------------------
missing_libs = []
try:
    from pypdf import PdfReader
    import docx
    import reticker
    PDF_DOCX_AVAILABLE = True
except ImportError as e:
    PDF_DOCX_AVAILABLE = False
    missing_libs.append("pypdf, python-docx, reticker")

try:
    import easyocr
    import torch
    OCR_AVAILABLE = True
    ocr_reader = None
except ImportError as e:
    OCR_AVAILABLE = False
    missing_libs.append("easyocr, torch")

if missing_libs:
    st.sidebar.warning(f"Missing libraries: {', '.join(missing_libs)}. Some file types may not work. Install with: pip install " + " ".join(missing_libs))

# 1. UI SETUP & THEME-AWARE CSS
st.set_page_config(page_title="Strategic AI Investment Architect", layout="wide")

# Inject security CSS + app CSS together
st.markdown(_HIDE_ST_STYLE + """
<style>
    [data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 800 !important; color: #1f77b4; }
    .phase-card { background-color: #f4f6f9; color: #1a1a1a; padding: 20px; border-radius: 10px; border: 1px solid #dcdcdc; min-height: 420px; }
    .news-card { background-color: #ffffff; color: #1a1a1a; padding: 15px; border-radius: 8px; border-left: 5px solid #0288d1; margin-bottom: 10px; font-size: 14px; box-shadow: 1px 1px 5px rgba(0,0,0,0.1); }
    .fib-box { background-color: #e3f2fd; color: #0d47a1; padding: 10px; border-radius: 5px; margin-top: 5px; border-left: 4px solid #1565c0; font-family: monospace; font-weight: bold; }
    .impact-news { background-color: #fff3e0; border-left: 8px solid #ff9800; padding: 10px; margin-bottom: 8px; border-radius: 5px; }
    .roadmap-card { background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #1565c0 100%); color: #fff; padding: 20px; border-radius: 12px; margin-bottom: 16px; box-shadow: 0 4px 15px rgba(21,101,192,0.3); }
    .roadmap-phase { background: rgba(255,255,255,0.12); border-radius: 8px; padding: 14px; margin-bottom: 10px; border-left: 4px solid #64b5f6; }
    .roadmap-phase h5 { margin: 0 0 8px 0; color: #90caf9; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
    .roadmap-ticker { display: inline-block; background: rgba(100,181,246,0.2); border: 1px solid #64b5f6; border-radius: 4px; padding: 2px 8px; margin: 2px; font-family: monospace; font-size: 12px; font-weight: bold; }
    .risk-badge-low { background: #1b5e20; color: #a5d6a7; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; }
    .risk-badge-med { background: #e65100; color: #ffcc80; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; }
    .risk-badge-high { background: #b71c1c; color: #ef9a9a; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; }
    .sector-gap { background: #fff8e1; border-left: 5px solid #ffc107; padding: 12px; border-radius: 6px; margin-bottom: 8px; color: #1a1a1a; }
    @media (prefers-color-scheme: dark) {
        .phase-card { background-color: #1e2129; color: #ffffff; border: 1px solid #3d414b; }
        .news-card { background-color: #262730; color: #ffffff; border-left: 5px solid #00b0ff; }
        .fib-box { background-color: #0d47a1; color: #e3f2fd; border-left: 4px solid #00b0ff; }
        .impact-news { background-color: #332e1f; color: #ffe0b2; border-left: 8px solid #ffb74d; }
        .sector-gap { background: #332e00; border-left: 5px solid #ffc107; color: #fff8e1; }
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
st.title("🏛️ Strategic AI Investment Architect (V11.5)")

# Add logout button in sidebar
with st.sidebar:
    if st.button("🔒 Logout", key="logout_btn"):
        st.session_state["authenticated"] = False
        st.rerun()

# 3. RATE LIMITER
class RateLimiter:
    def __init__(self, calls_per_second=0.3):
        self.calls_per_second = calls_per_second
        self.last_call_time = 0
    def wait(self):
        current_time = time.time()
        time_since_last = current_time - self.last_call_time
        min_interval = 1.0 / self.calls_per_second
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last + random.uniform(0.1, 0.3)
            time.sleep(sleep_time)
        self.last_call_time = time.time()

rate_limiter = RateLimiter(calls_per_second=0.3)

# 4. HELPER FUNCTIONS
def safe_request(func, *args, **kwargs):
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        return None, str(e) + "\n" + traceback.format_exc()

# ── LAYER 5: Sanitised error display — hide tracebacks from end users ─────────
_DEV_MODE = os.environ.get("DEV_MODE", "false").lower() == "true"

def show_error(msg: str, detail: str = ""):
    """Show user-friendly error. Only show traceback in DEV_MODE=true."""
    st.error(msg)
    if _DEV_MODE and detail:
        with st.expander("🔧 Developer details"):
            st.code(detail)

@st.cache_data(ttl=3600)
def get_exchange_rate(from_curr, to_curr):
    if from_curr == to_curr: return 1.0
    try:
        pair = f"{from_curr}{to_curr}=X"
        rate_limiter.wait()
        data = yf.download(pair, period="1d", progress=False)
        return float(data['Close'].iloc[-1]) if not data.empty else 1.0
    except Exception as e:
        st.warning(f"Exchange rate error: {e}")
        return 1.0

@st.cache_data(ttl=3600)
def resolve_smart_ticker(user_input):
    ticker_str = user_input.strip().upper()
    try:
        rate_limiter.wait()
        t_obj = yf.Ticker(ticker_str)
        last_price = t_obj.fast_info.get('lastPrice')
        if last_price is not None:
            info = t_obj.info
            name = (info.get('longName') or info.get('shortName') or
                    info.get('displayName') or ticker_str)
            name = name.strip() if name else ticker_str
            curr = t_obj.fast_info.get('currency', 'USD') or 'USD'
            return ticker_str, name, ".US", curr, None
        rate_limiter.wait()
        s = yf.Search(ticker_str, max_results=3)
        if s.tickers:
            res = s.tickers[0]
            ticker = res['symbol']
            name = (res.get('longname') or res.get('shortname') or
                    res.get('name') or ticker)
            name = name.strip() if name else ticker
            t_obj_fb = yf.Ticker(ticker)
            native_curr = t_obj_fb.fast_info.get('currency', 'USD') or 'USD'
            return ticker, name, "", native_curr, None
    except Exception as e:
        return ticker_str, ticker_str, ".US", "USD", str(e)
    return ticker_str, ticker_str, ".US", "USD", None

@st.cache_data(ttl=3600)
def get_ticker_sector_online(ticker):
    try:
        rate_limiter.wait()
        info = yf.Ticker(ticker).info
        sector = info.get('sector', '')
        industry = info.get('industry', '')
        if sector and sector not in ('', 'N/A', 'None'):
            return sector, industry
    except Exception:
        pass
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        soup = BeautifulSoup(requests.get(url, headers=headers, timeout=10).text, 'html.parser')
        def fvz(label):
            td = soup.find('td', string=label)
            return td.find_next_sibling('td').text.strip() if td else None
        sector = fvz('Sector') or 'Unknown'
        industry = fvz('Industry') or ''
        return sector, industry
    except Exception:
        pass
    return 'Unknown', ''

@st.cache_data(ttl=1800)
def get_enhanced_news(ticker):
    headlines = []
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        req = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(req.text, 'html.parser')
        news_table = soup.find(id='news-table')
        if news_table:
            rows = news_table.find_all('tr')
            for row in rows[:10]:
                try:
                    a_tag = row.find('a')
                    if a_tag and a_tag.text:
                        headline = a_tag.text
                        date_td = row.find('td', class_='nn-date')
                        date_str = date_td.text if date_td else ""
                        if date_str:
                            try:
                                dt = datetime.datetime.strptime(date_str, "%b-%d-%y %I:%M%p")
                            except:
                                dt = datetime.datetime.now()
                        else:
                            dt = datetime.datetime.now()
                        sentiment = TextBlob(headline).sentiment.polarity
                        headlines.append({'date': dt, 'headline': headline, 'sentiment': sentiment, 'source': 'Finviz'})
                except:
                    continue
    except Exception as e:
        st.warning(f"Finviz news error: {e}")
    try:
        rate_limiter.wait()
        ticker_obj = yf.Ticker(ticker)
        yf_news = ticker_obj.news
        for item in yf_news[:10]:
            try:
                title = item.get('title', '')
                if not title: continue
                ts = item.get('providerPublishTime')
                if ts: dt = datetime.datetime.fromtimestamp(ts)
                else: dt = datetime.datetime.now()
                sentiment = TextBlob(title).sentiment.polarity
                headlines.append({'date': dt, 'headline': title, 'sentiment': sentiment, 'source': 'Yahoo'})
            except:
                continue
    except Exception as e:
        st.warning(f"Yahoo news error: {e}")
    unique = []
    seen = set()
    for h in headlines:
        if h['headline'] not in seen:
            seen.add(h['headline'])
            unique.append(h)
    unique.sort(key=lambda x: x['date'], reverse=True)
    return unique

def filter_impactful_news(news_list, threshold=0.3, keywords=None):
    if keywords is None:
        keywords = ['earnings', 'dividend', 'fed', 'revenue', 'lawsuit', 'sec',
                    'merger', 'acquisition', 'growth', 'crash', 'ai', 'claude',
                    'agent', 'product', 'launch', 'guidance', 'forecast', 'upgrade',
                    'downgrade', 'target', 'investor', 'conference']
    impactful = []
    for item in news_list:
        headline_lower = item['headline'].lower()
        if abs(item['sentiment']) > threshold or any(kw in headline_lower for kw in keywords):
            impactful.append(item)
    return impactful

def calculate_technicals(df):
    delta = df['y'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    curr_p = df['y'].iloc[-1]
    recent_low = df['y'].tail(126).min()
    diff = max(curr_p - recent_low, curr_p * 0.10)
    fib_levels = {'0.382': curr_p - (diff * 0.382), '0.500': curr_p - (diff * 0.500), '0.618': curr_p - (diff * 0.618)}
    return df['rsi'].iloc[-1], fib_levels

@st.cache_data(ttl=3600)
def get_fundamental_health(ticker, suffix):
    try:
        rate_limiter.wait()
        t_obj = yf.Ticker(ticker)
        df = t_obj.history(period="5y")
        if df.empty:
            end = datetime.datetime.now(); start = end - datetime.timedelta(days=1825)
            df = web.DataReader(f"{ticker}{suffix}", 'stooq', start, end)
        if df is None or df.empty: return None, None, "No historical data"
        df = df.reset_index().rename(columns={'Date': 'ds', 'Close': 'y', 'Volume': 'vol'}).sort_values('ds')
        df['ds'] = df['ds'].dt.tz_localize(None)
        health = {"ROE": 0, "Debt": 0, "PB": 0, "Margin": "N/A", "CurrentRatio": "N/A"}
        try:
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            soup = BeautifulSoup(requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text, 'html.parser')
            def fvz(label):
                td = soup.find('td', string=label)
                return td.find_next_sibling('td').text.strip('%').replace(',', '') if td else "-"
            health = {
                "ROE": float(fvz("ROE"))/100 if fvz("ROE")!="-" else 0,
                "Debt": float(fvz("Debt/Eq")) if fvz("Debt/Eq")!="-" else 0,
                "PB": float(fvz("P/B")) if fvz("P/B")!="-" else 0,
                "Margin": fvz("Profit Margin") + "%",
                "CurrentRatio": fvz("Current Ratio")
            }
        except Exception as e:
            pass
        return df, health, None
    except Exception as e:
        return None, None, str(e)

def volume_trend_message(df_vol):
    if df_vol.empty or len(df_vol) < 20: return "Insufficient volume data."
    df_vol = df_vol.copy()
    df_vol['price_change'] = df_vol['y'].diff()
    df_vol['vol_change'] = df_vol['vol'].diff()
    up_days = df_vol[df_vol['price_change'] > 0]
    down_days = df_vol[df_vol['price_change'] < 0]
    up_vol_up = (up_days['vol_change'] > 0).sum()
    down_vol_up = (down_days['vol_change'] > 0).sum()
    total_up = len(up_days)
    total_down = len(down_days)
    if total_up == 0 or total_down == 0: return "Neutral volume pattern."
    up_confirmation = up_vol_up / total_up
    down_confirmation = down_vol_up / total_down
    if up_confirmation > 0.6 and down_confirmation < 0.4:
        return "✅ Volume confirms uptrend – bullish."
    elif up_confirmation < 0.4 and down_confirmation > 0.6:
        return "⚠️ Volume divergence – bearish."
    elif up_confirmation > 0.6 and down_confirmation > 0.6:
        return "⚖️ Mixed volume – indecision."
    else:
        return "➡️ Neutral volume trend."


def compute_7day_target(df, forecast, all_news, is_death_cross, fx):
    try:
        last_ds = df['ds'].iloc[-1]
        future_rows = forecast[forecast['ds'] > last_ds].head(7)
        if len(future_rows) >= 7:
            base_raw = float(future_rows['yhat'].iloc[-1])
        else:
            base_raw = float(df['y'].iloc[-1]) * 1.005
    except Exception:
        base_raw = float(df['y'].iloc[-1]) * 1.005
    now = datetime.datetime.now()
    s_sum, w_sum = 0.0, 0.0
    for item in all_news:
        age = max((now - item['date']).total_seconds() / 86400, 0.01)
        if age > 7:
            continue
        w = 1.0 / age
        s_sum += item['sentiment'] * w
        w_sum += w
    avg_sentiment = s_sum / w_sum if w_sum > 0 else 0.0
    news_adj = avg_sentiment * 0.03
    momentum_adj = float(df['y'].pct_change().tail(5).mean()) * 0.5
    cross_penalty = -0.02 if is_death_cross else 0.0
    total_adj = max(0.95, min(1.05, 1.0 + news_adj + momentum_adj + cross_penalty))
    return base_raw * total_adj * fx


# 5. DOCUMENT/IMAGE EXTRACTION CLASS
class PortfolioExtractor:
    def __init__(self):
        self.ticker_extractor = None
        if PDF_DOCX_AVAILABLE:
            try:
                self.ticker_extractor = reticker.TickerExtractor(
                    deduplicate=True,
                    match_config=reticker.TickerMatchConfig(
                        prefixed_uppercase=True, unprefixed_uppercase=True,
                        prefixed_lowercase=True, prefixed_titlecase=True, separators="-."
                    )
                )
            except:
                pass
        self.ocr_reader = None
        self.ocr_available = OCR_AVAILABLE

    def _get_ocr_reader(self):
        if self.ocr_reader is None and self.ocr_available:
            try:
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            except Exception as e:
                st.warning(f"OCR model load failed: {e}. Will use regex only.")
                self.ocr_available = False
        return self.ocr_reader

    def extract_text_from_pdf(self, file_path):
        try:
            reader = PdfReader(file_path)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            return text, None
        except Exception as e:
            return "", str(e)

    def extract_text_from_docx(self, file_path):
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs]), None
        except Exception as e:
            return "", str(e)

    def extract_text_from_image(self, file_path):
        if not self.ocr_available:
            return "", "OCR not available"
        reader = self._get_ocr_reader()
        if reader is None:
            return "", "OCR reader not initialized"
        try:
            result = reader.readtext(file_path, detail=0, paragraph=True)
            return "\n".join(result), None
        except Exception as e:
            return "", str(e)

    def find_tickers_regex(self, text):
        pattern = r'\b[A-Z]{1,5}\b'
        matches = re.findall(pattern, text)
        blacklist = {'ETF', 'USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'THE', 'AND', 'FOR', 'INC', 'LTD', 'LLC', 'NYSE', 'NASDAQ'}
        return list(set([m for m in matches if m not in blacklist]))

    def extract_quantities(self, text, ticker):
        patterns = [
            rf'{ticker}\s+(\d+(?:\.\d+)?)',
            rf'(\d+(?:\.\d+)?)\s+{ticker}',
            rf'\${ticker}\s+(\d+(?:\.\d+)?)',
            rf'(\d+(?:\.\d+)?)\s+shares?\s+of\s+{ticker}',
            rf'{ticker}\s+shares?\s+(\d+(?:\.\d+)?)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.groups():
                return float(match.group(1))
        return 1.0

    def process_file(self, file_path):
        ext = Path(file_path).suffix.lower()
        text = ""
        error = None
        if ext == '.pdf':
            text, error = self.extract_text_from_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            text, error = self.extract_text_from_docx(file_path)
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            text, error = self.extract_text_from_image(file_path)
        else:
            return [], "Unsupported file type", ""
        if error:
            return [], f"Extraction error: {error}", ""
        if not text.strip():
            return [], "No text could be extracted.", ""
        preview = text[:1000] + "..." if len(text) > 1000 else text
        if self.ticker_extractor:
            raw_tickers = self.ticker_extractor.extract(text)
        else:
            raw_tickers = self.find_tickers_regex(text)
        raw_tickers = list(set(raw_tickers))
        holdings = []
        validation_errors = []
        for ticker in raw_tickers[:15]:
            try:
                rate_limiter.wait()
                t_obj = yf.Ticker(ticker)
                info = t_obj.info
                if info.get('regularMarketPrice'):
                    shares = self.extract_quantities(text, ticker)
                    holdings.append({
                        'ticker': ticker, 'name': info.get('longName', ticker),
                        'shares': shares, 'sector': info.get('sector', 'Unknown'),
                        'current_price': info.get('regularMarketPrice')
                    })
                else:
                    validation_errors.append(f"{ticker}: no market price")
            except Exception as e:
                validation_errors.append(f"{ticker}: {str(e)}")
        return holdings, preview, f"Found {len(raw_tickers)} potential tickers, {len(holdings)} validated. Errors: {validation_errors}"


# 6. PORTFOLIO ANALYSIS
def analyze_ticker_basic(ticker, display_currency):
    try:
        rate_limiter.wait()
        t_obj = yf.Ticker(ticker)
        info = t_obj.info
        price = info.get('regularMarketPrice')
        if not price:
            return None, "No market price"
        currency = info.get('currency', 'USD')
        fx = get_exchange_rate(currency, display_currency)
        price_display = price * fx
        name = info.get('longName', ticker)
        if not name:
            name = ticker
        sector = info.get('sector', '')
        industry = info.get('industry', '')
        if not sector or sector in ('', 'N/A', 'None'):
            sector, industry = get_ticker_sector_online(ticker)
        return {
            'name': name, 'sector': sector if sector else 'Unknown',
            'industry': industry if industry else '', 'price': price_display, 'currency': currency
        }, None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=3600)
def batch_score_candidates(candidates_tuple):
    tickers_list = list(candidates_tuple)
    results = {}
    try:
        batch = yf.Tickers(" ".join(tickers_list))
        for ticker in tickers_list:
            try:
                info = batch.tickers[ticker].info
                cur    = info.get('currentPrice') or info.get('regularMarketPrice') or 0
                target = info.get('targetMeanPrice') or 0
                rec    = info.get('recommendationMean') or 3.0
                eps_g  = info.get('earningsGrowth')  or 0
                rev_g  = info.get('revenueGrowth')   or 0
                lo52   = info.get('fiftyTwoWeekLow')  or cur * 0.7
                hi52   = info.get('fiftyTwoWeekHigh') or cur * 1.3
                upside       = ((target - cur) / cur * 100) if cur > 0 and target > 0 else 0
                consensus_sc = max(0, (5 - rec) / 4 * 100)
                growth_sc    = min(100, max(0, (eps_g + rev_g) * 100))
                span         = hi52 - lo52
                momentum_sc  = ((cur - lo52) / span * 100) if span > 0 else 50
                upside_sc    = min(100, max(0, upside * 3))
                score = (upside_sc * 0.35 + consensus_sc * 0.30 +
                         growth_sc * 0.20 + momentum_sc  * 0.15)
                if   rec <= 1.5: label = "Strong Buy"
                elif rec <= 2.5: label = "Buy"
                elif rec <= 3.5: label = "Hold"
                else:            label = "Underperform"
                name    = info.get('shortName') or info.get('longName') or ticker
                summary = (f"Analyst: {label} | Upside: {upside:+.1f}% | "
                           f"EPS Growth: {eps_g*100:.1f}% | Rev Growth: {rev_g*100:.1f}%")
                results[ticker] = {
                    'ticker': ticker, 'name': name, 'score': round(score, 1),
                    'upside_pct': round(upside, 1), 'analyst_label': label,
                    'summary': summary, 'valid': cur > 0
                }
            except Exception:
                results[ticker] = {'ticker': ticker, 'name': ticker, 'score': 0,
                                   'upside_pct': 0, 'analyst_label': 'N/A',
                                   'summary': 'Data unavailable', 'valid': False}
    except Exception:
        for ticker in tickers_list:
            results[ticker] = {'ticker': ticker, 'name': ticker, 'score': 0,
                               'upside_pct': 0, 'analyst_label': 'N/A',
                               'summary': 'Data unavailable', 'valid': False}
    return results


def get_best_candidates(candidate_pool, owned_tickers, top_n=3):
    to_score = tuple(t for t in candidate_pool if t.upper() not in owned_tickers)
    if not to_score:
        return []
    scored_map = batch_score_candidates(to_score)
    scored = [v for v in scored_map.values() if v['valid']]
    scored.sort(key=lambda x: x['score'], reverse=True)
    return scored[:top_n]


def format_candidates_html(candidates):
    if not candidates:
        return "<i style='opacity:.7'>No qualifying candidates in live scan.</i>"
    parts = []
    for c in candidates:
        colour = "#a5d6a7" if c['upside_pct'] > 0 else "#ef9a9a"
        parts.append(
            f"<span class='roadmap-ticker' title='{c['summary']}'>{c['ticker']}</span>"
            f"<span style='font-size:11px;color:{colour};margin-right:10px;'>"
            f"&nbsp;{c['analyst_label']} · {c['upside_pct']:+.1f}%</span>"
        )
    return "".join(parts)


def generate_investment_roadmap(df_results, sector_data, total_value, display_currency):
    sym = "$" if display_currency == "USD" else "€"
    SECTOR_BENCHMARKS = {
        'Technology':             {'target': 28, 'etf': 'XLK',  'risk': 'High',
            'candidates': ['MSFT','GOOGL','META','ORCL','NVDA','AVGO','AMD','CRM','ADBE','NOW']},
        'Healthcare':             {'target': 13, 'etf': 'XLV',  'risk': 'Low',
            'candidates': ['UNH','LLY','ABBV','ISRG','TMO','DHR','BSX','ELV','HCA','VRTX']},
        'Financial Services':     {'target': 13, 'etf': 'XLF',  'risk': 'Medium',
            'candidates': ['JPM','V','MA','BRK-B','GS','MS','AXP','SCHW','CME','ICE']},
        'Consumer Cyclical':      {'target': 10, 'etf': 'XLY',  'risk': 'High',
            'candidates': ['AMZN','TSLA','NKE','MCD','BKNG','ABNB','CMG','TJX','LULU','ROST']},
        'Industrials':            {'target': 8,  'etf': 'XLI',  'risk': 'Medium',
            'candidates': ['CAT','DE','RTX','GE','ETN','EMR','UBER','FDX','LMT','NOC']},
        'Communication Services': {'target': 8,  'etf': 'XLC',  'risk': 'Medium',
            'candidates': ['GOOGL','META','NFLX','SPOT','DIS','CHTR','T','VZ','EA','TTWO']},
        'Consumer Defensive':     {'target': 6,  'etf': 'XLP',  'risk': 'Low',
            'candidates': ['COST','WMT','PG','KO','PEP','MDLZ','CL','KHC','GIS','SYY']},
        'Energy':                 {'target': 4,  'etf': 'XLE',  'risk': 'High',
            'candidates': ['XOM','CVX','SLB','EOG','PXD','MPC','PSX','COP','OXY','HAL']},
        'Utilities':              {'target': 2,  'etf': 'XLU',  'risk': 'Low',
            'candidates': ['NEE','DUK','SO','D','AES','PCG','EXC','XEL','AWK','ED']},
        'Real Estate':            {'target': 2,  'etf': 'XLRE', 'risk': 'Low',
            'candidates': ['PLD','AMT','EQIX','CCI','SPG','O','VICI','WY','AVB','EQR']},
        'Basic Materials':        {'target': 2,  'etf': 'XLB',  'risk': 'Medium',
            'candidates': ['LIN','APD','FCX','NEM','NUE','ALB','DOW','PPG','VMC','MLM']},
    }
    owned_tickers = set(df_results['Ticker'].str.upper().tolist())
    current_sectors = {}
    for _, row in sector_data.iterrows():
        current_sectors[row['Sector']] = row['Current Value'] / total_value * 100
    overweight, underweight, missing = [], [], []
    for sector, info in SECTOR_BENCHMARKS.items():
        current = current_sectors.get(sector, 0)
        gap = current - info['target']
        if gap > 5:
            overweight.append((sector, current, info['target'], gap))
        elif current == 0:
            missing.append((sector, info['target'], info))
        elif gap < -3:
            underweight.append((sector, current, info['target'], abs(gap), info))
    missing.sort(key=lambda x: x[1], reverse=True)
    underweight.sort(key=lambda x: x[3], reverse=True)
    tech_weight   = current_sectors.get('Technology', 0)
    income_weight = sum(current_sectors.get(s, 0) for s in ['Consumer Defensive', 'Utilities', 'Real Estate'])
    concentration = (df_results.nlargest(3, 'Current Value')['Current Value'].sum()
                     / total_value * 100) if total_value > 0 else 0
    relevant_sectors = ([info for _, _, info in missing[:3]] +
                        [info for _, _, _, _, info in underweight[:2]] +
                        [info for _, _, info in missing[3:6]])
    all_candidates = list(dict.fromkeys(
        t for info in relevant_sectors
        for t in info['candidates']
        if t.upper() not in owned_tickers
    ))
    phase3_pools = ['NVDA','MSFT','GOOGL','AMD','AVGO','CRM','ADBE','NOW','SNOW','PLTR',
                    'O','JNJ','PG','KO','VZ','NEE','VICI','ABBV','MO','T']
    all_candidates += [t for t in phase3_pools if t.upper() not in owned_tickers and t not in all_candidates]
    if all_candidates:
        batch_score_candidates(tuple(all_candidates))
    roadmap = []

    # PHASE 1
    phase1_actions = []
    for sector, cur, tgt, gap in overweight[:3]:
        value_to_shed = total_value * (gap / 100)
        target_value  = total_value * (tgt / 100)
        sector_holdings = df_results[df_results['Sector'] == sector].copy()
        sector_holdings = sector_holdings.sort_values('Allocation %', ascending=False)
        trim_lines = []
        remaining_to_shed = value_to_shed
        for _, row in sector_holdings.iterrows():
            if remaining_to_shed <= 0:
                break
            ticker      = row['Ticker']
            shares_held = float(row['Shares'])
            price       = float(row['Current Price'])
            shares_to_sell = min(shares_held, remaining_to_shed / price) if price > 0 else 0
            shares_to_sell = max(0, round(shares_to_sell, 4))
            sell_value     = shares_to_sell * price
            pct_of_pos     = (shares_to_sell / shares_held * 100) if shares_held > 0 else 0
            if shares_to_sell > 0:
                trim_lines.append(
                    f"<span class='roadmap-ticker'>{ticker}</span> "
                    f"sell <b>{shares_to_sell:,.2f} shares</b> "
                    f"≈ {sym}{sell_value:,.0f} "
                    f"<span style='font-size:11px;opacity:.8;'>({pct_of_pos:.0f}% of position)</span>"
                )
            remaining_to_shed -= sell_value
        trim_detail = ("<br>&nbsp;&nbsp;🎯 Suggested trims: " + " &nbsp;·&nbsp; ".join(trim_lines)) if trim_lines else ""
        phase1_actions.append(
            f"<b>Trim {sector}</b>: currently {cur:.1f}% vs {tgt}% benchmark — "
            f"reduce by ~{gap:.0f}% · sell ≈{sym}{value_to_shed:,.0f} to reach target {sym}{target_value:,.0f}"
            f"{trim_detail}"
        )
    if concentration > 50:
        top3 = df_results.nlargest(3, 'Current Value')['Ticker'].tolist()
        phase1_actions.append(
            f"<b>Reduce concentration</b>: top 3 holdings ({', '.join(top3)}) = "
            f"{concentration:.0f}% of portfolio. Target under 40%.")
    if not phase1_actions:
        phase1_actions.append("Portfolio is reasonably balanced — no urgent rebalancing needed.")
    roadmap.append({'phase': 'PHASE 1 — IMMEDIATE (Now → 30 Days)', 'icon': '⚡', 'actions': phase1_actions})

    # PHASE 2
    phase2_actions = []
    for sector, target_pct, info in missing[:3]:
        suggested_alloc = total_value * (target_pct / 100)
        rb_key = 'low' if info['risk']=='Low' else 'med' if info['risk']=='Medium' else 'high'
        risk_badge = f"<span class='risk-badge-{rb_key}'>{info['risk']} Risk</span>"
        best = get_best_candidates(info['candidates'], owned_tickers, top_n=3)
        picks_html = format_candidates_html(best)
        phase2_actions.append(
            f"<b>Add {sector}</b> {risk_badge}: 0% → target {target_pct}% "
            f"(≈{sym}{suggested_alloc:,.0f})<br>"
            f"&nbsp;&nbsp;📡 Live picks: {picks_html} "
            f"or ETF <span class='roadmap-ticker'>{info['etf']}</span>")
    for sector, cur, tgt, gap, info in underweight[:2]:
        suggested_alloc = total_value * (gap / 100)
        rb_key = 'low' if info['risk']=='Low' else 'med' if info['risk']=='Medium' else 'high'
        risk_badge = f"<span class='risk-badge-{rb_key}'>{info['risk']} Risk</span>"
        best = get_best_candidates(info['candidates'], owned_tickers, top_n=3)
        picks_html = format_candidates_html(best)
        phase2_actions.append(
            f"<b>Increase {sector}</b> {risk_badge}: {cur:.1f}% → {tgt}% "
            f"(add ≈{sym}{suggested_alloc:,.0f})<br>"
            f"&nbsp;&nbsp;📡 Live picks: {picks_html}")
    if not phase2_actions:
        phase2_actions.append("Good sector coverage — focus on quality within existing positions.")
    roadmap.append({'phase': 'PHASE 2 — CORE BUILD (1–3 Months)', 'icon': '🏗️', 'actions': phase2_actions})

    # PHASE 3
    phase3_actions = []
    if tech_weight < 15:
        best_tech = get_best_candidates(
            ['NVDA','MSFT','GOOGL','AMD','AVGO','CRM','ADBE','NOW','SNOW','PLTR'],
            owned_tickers, top_n=4)
        phase3_actions.append(
            f"<b>Growth Layer — AI/Tech</b>: Only {tech_weight:.1f}% tech exposure. "
            f"Top live-scored names:<br>&nbsp;&nbsp;📡 {format_candidates_html(best_tech)}")
    if income_weight < 8:
        best_div = get_best_candidates(
            ['O','JNJ','PG','KO','VZ','NEE','VICI','ABBV','MO','T'],
            owned_tickers, top_n=4)
        phase3_actions.append(
            f"<b>Income / Dividend Layer</b>: Only {income_weight:.1f}% in defensive/income sectors. "
            f"Top live-scored payers:<br>&nbsp;&nbsp;📡 {format_candidates_html(best_div)}")
    phase3_actions.append(
        "<b>International Exposure</b>: Consider adding "
        "<span class='roadmap-ticker'>VEA</span> (Developed Markets), "
        "<span class='roadmap-ticker'>VWO</span> (Emerging Markets), or "
        "<span class='roadmap-ticker'>EWJ</span> (Japan) for geographic diversification.")
    phase3_actions.append(
        "<b>Bonds / Alternatives Hedge</b>: For capital protection consider "
        "<span class='roadmap-ticker'>BND</span> (Total Bond), "
        "<span class='roadmap-ticker'>GLD</span> (Gold), or "
        "<span class='roadmap-ticker'>TIP</span> (TIPS/Inflation hedge).")
    roadmap.append({'phase': 'PHASE 3 — GROWTH & INCOME LAYER (3–6 Months)', 'icon': '📈', 'actions': phase3_actions})

    # PHASE 4
    phase4_actions = []
    for sector, target_pct, info in missing[3:6]:
        best = get_best_candidates(info['candidates'], owned_tickers, top_n=2)
        picks_html = format_candidates_html(best)
        phase4_actions.append(
            f"<b>Complete {sector} coverage</b> (target {target_pct}%):<br>"
            f"&nbsp;&nbsp;📡 {picks_html} "
            f"or ETF <span class='roadmap-ticker'>{info['etf']}</span>")
    phase4_actions.append("<b>Rebalance Annually</b>: Review sector weights each January. Trim winners >+5% overweight, add to laggards.")
    phase4_actions.append("<b>Tax-Loss Harvesting</b>: Before year-end, harvest unrealised losses to offset capital gains.")
    phase4_actions.append("<b>Dollar-Cost Averaging</b>: For new positions deploy capital in 3 tranches over 6–8 weeks to reduce timing risk.")
    roadmap.append({'phase': 'PHASE 4 — LONG-TERM OPTIMISATION (6–12 Months)', 'icon': '🎯', 'actions': phase4_actions})

    return roadmap, overweight, underweight, missing


def render_investment_roadmap(roadmap, overweight, underweight, missing, sym):
    st.markdown("---")
    st.subheader("🗺️ Personalised Investment Roadmap")
    st.caption("Phased action plan vs S&P 500 benchmark weights. Ticker picks scored live: analyst consensus, price target upside, EPS/revenue growth & 52-week momentum.")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Sectors Covered", f"{11 - len(missing)}/11", f"-{len(missing)} missing")
    col_b.metric("Overweight Sectors", str(len(overweight)), "Need trimming" if overweight else "✅ OK")
    col_c.metric("Underweight Sectors", str(len(underweight)), "Need building" if underweight else "✅ OK")
    for phase_data in roadmap:
        with st.container():
            st.markdown(f"""
<div class="roadmap-card">
  <h4 style="margin:0 0 12px 0; font-size:16px;">{phase_data['icon']} {phase_data['phase']}</h4>
  {''.join([f'<div class="roadmap-phase"><p style="margin:0; line-height:1.9; font-size:14px;">{action}</p></div>' for action in phase_data['actions']])}
</div>
""", unsafe_allow_html=True)
    if missing or underweight:
        st.markdown("#### 📊 Sector Gap Analysis vs S&P 500 Benchmark")
        gap_rows = []
        for sector, target_pct, info in missing:
            gap_rows.append({'Sector': sector, 'You Have': '0%',
                             'S&P Benchmark': f'{target_pct}%', 'Gap': f'-{target_pct}%',
                             'Risk Level': info['risk'], 'Quick ETF': info['etf']})
        for sector, cur, tgt, gap, info in underweight:
            gap_rows.append({'Sector': sector, 'You Have': f'{cur:.1f}%',
                             'S&P Benchmark': f'{tgt}%', 'Gap': f'-{gap:.1f}%',
                             'Risk Level': info['risk'], 'Quick ETF': info['etf']})
        if gap_rows:
            st.dataframe(pd.DataFrame(gap_rows), use_container_width=True)


def suggest_diversification(current_sectors, total_value):
    suggestions = []
    sector_examples = {
        'Technology': ['AAPL', 'MSFT', 'QQQ'], 'Healthcare': ['JNJ', 'UNH', 'XLV'],
        'Financial Services': ['JPM', 'BAC', 'XLF'], 'Consumer Cyclical': ['AMZN', 'TSLA', 'XLY'],
        'Industrials': ['HON', 'CAT', 'XLI'], 'Energy': ['XOM', 'CVX', 'XLE'],
        'Utilities': ['NEE', 'DUK', 'XLU'], 'Real Estate': ['PLD', 'AMT', 'XLRE'],
        'Communication Services': ['META', 'GOOGL', 'XLC'], 'Consumer Defensive': ['PG', 'KO', 'XLP'],
        'Basic Materials': ['LIN', 'BHP', 'XLB']
    }
    allocated_sectors = set(current_sectors.keys())
    for sector, examples in sector_examples.items():
        if sector not in allocated_sectors:
            suggestions.append(f"💡 No exposure to **{sector}**. Consider adding {', '.join(examples[:3])}")
        elif current_sectors[sector] < 10:
            suggestions.append(f"💡 Low allocation ({current_sectors[sector]:.1f}%) to **{sector}**. Consider increasing via {', '.join(examples[:2])}")
    return suggestions[:5]


def process_uploaded_file(uploaded_file):
    file_name = uploaded_file.name.lower()
    ext = Path(file_name).suffix
    if ext in ['.csv', '.xlsx', '.xls']:
        try:
            if ext == '.csv':
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            orig_cols = list(df.columns)
            df.columns = df.columns.astype(str).str.strip().str.lower()
            COLUMN_MAP = {
                'ticker': ['ticker', 'symbol', 'stock', 'scrip', 'security',
                           'code', 'asset', 'instrument', 'name', 'equity'],
                'shares': ['shares', 'qty', 'quantity', 'units', 'amount',
                           'no of shares', 'no. of shares', 'number of shares',
                           'holding', 'holdings', 'position', 'lots'],
                'purchase price': ['purchase price', 'avg price', 'average price',
                                   'avg cost', 'average cost', 'cost price',
                                   'buy price', 'cost basis', 'price paid',
                                   'cost', 'buy', 'bought at', 'invested price',
                                   'acquisition price'],
            }
            rename_map = {}
            detected = {}
            for canonical, variants in COLUMN_MAP.items():
                for col in df.columns:
                    col_clean = col.strip().lower()
                    if col_clean in variants or any(v in col_clean for v in variants):
                        rename_map[col] = canonical
                        detected[canonical] = col
                        break
            df.rename(columns=rename_map, inplace=True)
            if 'ticker' not in df.columns:
                st.error(f"Could not find a ticker/symbol column. Detected columns: {', '.join(orig_cols)}.")
                return None
            if detected:
                st.success("✅ Auto-detected columns: " + " | ".join(f"**{v}** → {k}" for k, v in detected.items()))
            if 'shares' not in df.columns:
                st.warning("⚠️ No quantity/shares column found — defaulting to 1 share per holding.")
                df['shares'] = 1
            else:
                df['shares'] = pd.to_numeric(df['shares'], errors='coerce').fillna(1)
            if 'purchase price' in df.columns:
                df['purchase price'] = pd.to_numeric(df['purchase price'], errors='coerce')
            df['ticker'] = df['ticker'].astype(str).str.strip().str.upper()
            df = df[df['ticker'].str.len() <= 6]
            df = df[df['ticker'] != 'NAN'].reset_index(drop=True)
            return {'type': 'tabular', 'data': df, 'preview': None}
        except Exception as e:
            show_error(f"Error reading file: {e}", traceback.format_exc())
            return None
    elif ext in ['.pdf', '.docx', '.doc', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        if ext in ['.pdf', '.docx', '.doc'] and not PDF_DOCX_AVAILABLE:
            st.error("PDF/Word support not installed.")
            return None
        extractor = PortfolioExtractor()
        temp_path = f"temp_{uploaded_file.name}"
        try:
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            holdings, preview, message = extractor.process_file(temp_path)
        finally:
            # ── LAYER 4: Always delete temp file, even if extraction fails ────
            if os.path.exists(temp_path):
                os.remove(temp_path)
        with st.expander("📄 Extraction Details"):
            st.text(message)
            st.text("Extracted text preview:")
            st.code(preview)
        if holdings:
            df = pd.DataFrame(holdings)
            return {'type': 'tabular', 'data': df, 'preview': preview}
        else:
            st.warning("No valid stock tickers found in the document/image.")
            return None
    else:
        st.error("Unsupported file type.")
        return None


# 7. SIDEBAR
st.sidebar.header("⚙️ Configuration")
user_query = st.sidebar.text_input("Ticker / Symbol", value="NVDA")
display_currency = st.sidebar.selectbox("Currency", ["USD", "EUR"])
total_capital = st.sidebar.number_input("Capital", value=10000)
run_audit = st.sidebar.button("🚀 Run Deep Audit")
st.sidebar.markdown("---")
st.sidebar.header("📁 Upload Portfolio / Document")
uploaded_file = st.sidebar.file_uploader(
    "CSV, Excel, PDF, Word, or Image",
    type=['csv', 'xlsx', 'xls', 'pdf', 'docx', 'doc', 'png', 'jpg', 'jpeg', 'tiff', 'bmp']
)

# Logout button at bottom of sidebar
st.sidebar.markdown("---")
if st.sidebar.button("🔒 Logout", key="logout_sidebar"):
    st.session_state["authenticated"] = False
    st.rerun()


# 8. SINGLE TICKER DEEP AUDIT
if run_audit:
    with st.spinner(f"Analyzing {user_query}... (may take a moment)"):
        ticker, name, suffix, native_curr, resolve_error = resolve_smart_ticker(user_query)
        if resolve_error:
            st.warning(f"Ticker resolution issue: {resolve_error}")
        display_name = name if (name and name.strip() and name.strip() != ticker) else None
        if not display_name:
            try:
                info = yf.Ticker(ticker).info
                display_name = (info.get('longName') or info.get('shortName') or ticker)
            except:
                display_name = ticker
        df, health, hist_error = get_fundamental_health(ticker, suffix)
        if hist_error:
            st.error(f"Historical data error: {hist_error}")
        if df is not None:
            fx = get_exchange_rate(native_curr, display_currency)
            sym = "$" if display_currency == "USD" else "€"
            cur_p = df['y'].iloc[-1] * fx
            df['50_Day_Moving_Average'] = df['y'].rolling(window=50).mean()
            df['200_Day_Moving_Average'] = df['y'].rolling(window=200).mean()
            forecast = None
            try:
                m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True,
                            changepoint_prior_scale=0.08, changepoint_range=0.9, seasonality_mode='additive')
                m.fit(df[['ds', 'y']])
                future = m.make_future_dataframe(periods=180)
                forecast = m.predict(future)
                hist_min, hist_max = df['y'].min(), df['y'].max()
                forecast['yhat']       = forecast['yhat'].clip(lower=hist_min*0.5, upper=hist_max*2)
                forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=hist_min*0.5, upper=hist_max*2)
                forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=hist_min*0.5, upper=hist_max*2)
                trend_val = forecast[forecast['ds'] == df['ds'].iloc[-1]]['yhat'].values[0] * fx
                deviation = ((cur_p - trend_val) / trend_val) * 100
                fair_low  = trend_val * 0.95
                fair_high = trend_val * 1.05
            except Exception as e:
                st.warning(f"Prophet forecast failed: {e}. Using simple projection.")
                trend_val = cur_p; deviation = 0
                fair_low  = cur_p * 0.95; fair_high = cur_p * 1.05; forecast = None
            is_death_cross = (df['50_Day_Moving_Average'].iloc[-1] < df['200_Day_Moving_Average'].iloc[-1])
            crossover_msg = "Price stability detected."
            cross_point = None
            for i in range(len(df)-60, len(df)):
                prev = i - 1
                if (df['50_Day_Moving_Average'].iloc[prev] < df['200_Day_Moving_Average'].iloc[prev] and
                        df['50_Day_Moving_Average'].iloc[i] >= df['200_Day_Moving_Average'].iloc[i]):
                    cross_point   = (df['ds'].iloc[i], df['50_Day_Moving_Average'].iloc[i], "GOLDEN")
                    crossover_msg = "🚀 GOLDEN CROSS: 50-Day Moving Average crossed ABOVE 200-Day."
                elif (df['50_Day_Moving_Average'].iloc[prev] > df['200_Day_Moving_Average'].iloc[prev] and
                        df['50_Day_Moving_Average'].iloc[i] <= df['200_Day_Moving_Average'].iloc[i]):
                    cross_point   = (df['ds'].iloc[i], df['50_Day_Moving_Average'].iloc[i], "DEATH")
                    crossover_msg = "⚠️ DEATH CROSS: 50-Day Moving Average crossed BELOW 200-Day."
            all_news = get_enhanced_news(ticker)
            if forecast is not None:
                target_p_7 = compute_7day_target(df, forecast, all_news, is_death_cross, fx)
            else:
                mom = float(df['y'].pct_change().tail(5).mean()) * 0.5
                target_p_7 = cur_p * max(0.95, min(1.05, 1.0 + mom + (-0.02 if is_death_cross else 0)))
            ai_roi_7 = ((target_p_7 - cur_p) / cur_p) * 100
            rsi, fibs = calculate_technicals(df)
            impactful_news = filter_impactful_news(all_news)
            headlines_display = [
                f"{'🟢' if n['sentiment']>0 else '🔴' if n['sentiment']<0 else '⚪'} {n['headline']}"
                for n in all_news[:5]
            ]
            avg_news_sentiment = np.mean([n['sentiment'] for n in impactful_news]) if impactful_news else 0
            score = 15
            if not is_death_cross:     score += 20
            if health['ROE'] > 0.12:   score += 20
            if ai_roi_7 > 0.5:         score += 30
            if avg_news_sentiment > 0: score += 15
            score = max(0, min(100, score))
            if score >= 70:   verdict, v_col, action, pct = "Strong Buy",    "v-green",  "ACTION: BUY NOW",         25
            elif score >= 35: verdict, v_col, action, pct = "Hold / Neutral","v-orange", "ACTION: MONITOR",          10
            else:             verdict, v_col, action, pct = "Sell / Avoid",  "v-red",    "ACTION: SELL / STAY AWAY", 0
            sl_price = cur_p * (0.95 if rsi > 70 else 0.85 if rsi < 30 else 0.90)
            st.subheader(f"📊 {display_name} Analysis ({ticker})")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                if deviation > 15:
                    st.warning(f"⚠️ MEAN REVERSION RISK: Price is {deviation:.1f}% above AI baseline trend.")
                else:
                    st.success(f"✅ TREND ALIGNMENT: Price is within {deviation:.1f}% of AI baseline.")
            with col_f2:
                st.info(f"💎 AI FAIR VALUE RANGE: {sym}{fair_low:,.2f} - {sym}{fair_high:,.2f}")
            st.markdown(f'<div class="impact-announcement">{crossover_msg}</div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Conviction Score",        f"{score}/100")
            m2.metric("Current Price",           f"{sym}{cur_p:,.2f}")
            m3.metric("7d AI Growth (News-Adj)", f"{ai_roi_7:.2f}%")
            m4.metric("7d AI Target",            f"{sym}{target_p_7:,.2f}")
            st.markdown(f'<div class="verdict-box {v_col}">Strategic Verdict: {verdict} | {action}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="stop-loss-box">🛑 AGGRESSIVE STOP LOSS: {sym}{sl_price:,.2f}</div>', unsafe_allow_html=True)
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("### 🏥 Company Health")
                st.table(pd.DataFrame({
                    "Metric":  ["ROE", "P/B Ratio", "Debt/Equity", "Current Ratio"],
                    "Status":  [f"{health['ROE']*100:.1f}%", f"{health['PB']}x", health['Debt'], health['CurrentRatio']],
                    "Rating":  ["✅ Prime" if health['ROE'] > 0.15 else "⚠️ Weak",
                                "✅ Healthy" if health['PB'] < 3.0 else "⚠️ Expensive", "✅ Safe", "✅ Liquid"]
                }))
                st.markdown("### 📰 Recent Headlines")
                for h in headlines_display:
                    st.markdown(f'<div class="news-card">{h}</div>', unsafe_allow_html=True)
            with col_r:
                st.markdown("### ⚖️ Strategy & Fibonacci Limits")
                st.markdown(f"""<div class="phase-card">
                    <h4 style="color:#1f77b4">PHASE 1: IMMEDIATE</h4>
                    <p><b>Invest Today:</b> {sym}{total_capital*(pct/100):,.2f} ({pct}% allocation)</p>
                    <hr>
                    <h4 style="color:#1f77b4">PHASE 2: STAGED ENTRY (FIBONACCI)</h4>
                    <div class="fib-box">🔹 Target 1 (0.382): {sym}{fibs['0.382']*fx:,.2f}</div>
                    <div class="fib-box">🔹 Target 2 (0.500): {sym}{fibs['0.500']*fx:,.2f}</div>
                    <div class="fib-box">🔹 Target 3 (0.618): {sym}{fibs['0.618']*fx:,.2f}</div>
                </div>""", unsafe_allow_html=True)
            if impactful_news:
                st.markdown("### 🔥 News Likely to Impact Price")
                for news in impactful_news[:7]:
                    emoji    = '🟢' if news['sentiment'] > 0.1 else '🔴' if news['sentiment'] < -0.1 else '⚪'
                    date_str = news['date'].strftime("%b %d, %Y")
                    st.markdown(f"""
                    <div class="impact-news">
                        <b>{emoji} {news['headline']}</b><br>
                        <span style="font-size:0.85rem;">{date_str} · {news['source']} · Sentiment: {news['sentiment']:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("---")
            st.subheader("🤖 AI Stock 180-Day Projection")
            fig, ax = plt.subplots(figsize=(12, 6))
            try:
                forecast_plot = forecast.copy()
                forecast_plot[['yhat', 'yhat_lower', 'yhat_upper']] *= fx
                m.plot(forecast_plot, ax=ax)
            except Exception:
                ax.plot(df['ds'], df['y'] * fx, 'b-', label='Historical')
            ax.plot(df['ds'], df['50_Day_Moving_Average']  * fx, label='50-Day MA',  color='orange', linewidth=2)
            ax.plot(df['ds'], df['200_Day_Moving_Average'] * fx, label='200-Day MA', color='red',    linewidth=2)
            if cross_point:
                ax.scatter(cross_point[0], cross_point[1] * fx, color='gold', s=300,
                           marker='*', label=f"{cross_point[2]} CROSS POINT", zorder=5)
            ax.set_xlim([datetime.datetime.now() - datetime.timedelta(days=180),
                         datetime.datetime.now() + datetime.timedelta(days=180)])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.legend(loc='upper left')
            st.pyplot(fig)
            st.markdown("---")
            st.subheader("📊 12-Month Relative Volume Trend")
            vol_fig, vol_ax = plt.subplots(figsize=(12, 4))
            vol_df = df.tail(252).copy()
            colors = ['#2e7d32' if i > 0 and vol_df.iloc[i]['y'] >= vol_df.iloc[i-1]['y']
                      else '#c62828' for i in range(len(vol_df))]
            vol_ax.bar(vol_df['ds'], vol_df['vol'], color=colors, alpha=0.7)
            vol_ax.set_ylabel("Shares Traded")
            vol_ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            st.pyplot(vol_fig)
            st.caption("🔵 **Volume color:** Green bars = price increased from previous day; Red bars = price decreased.")
            st.info(f"📈 **Volume Insight:** {volume_trend_message(vol_df)}")
        else:
            st.error(f"Could not fetch data for {user_query}. Error: {hist_error}")


# 9. AUTOMATIC PORTFOLIO ANALYSIS
if uploaded_file is not None:
    st.markdown("---")
    st.header("📁 Uploaded File Analysis")
    with st.spinner("Processing uploaded file..."):
        result = process_uploaded_file(uploaded_file)
    if result and result['type'] == 'tabular':
        df_holdings = result['data']
        results_list = []
        errors_list  = []
        total_value  = 0
        total_cost   = 0
        progress_bar = st.progress(0)
        status_text  = st.empty()
        for idx, row in df_holdings.iterrows():
            ticker         = str(row['ticker']).strip().upper()
            shares         = float(row['shares']) if 'shares' in row else 1
            purchase_price = row.get('purchase price', None)
            status_text.text(f"Analyzing {ticker}... ({idx+1}/{len(df_holdings)})")
            progress_bar.progress((idx + 1) / len(df_holdings))
            analysis, error = analyze_ticker_basic(ticker, display_currency)
            if analysis:
                value = shares * analysis['price']
                total_value += value
                if pd.notna(purchase_price):
                    cost     = shares * purchase_price
                    total_cost += cost
                    gain     = value - cost
                    gain_pct = (gain / cost) * 100 if cost != 0 else 0
                else:
                    gain = gain_pct = None
                results_list.append({
                    'Ticker':         ticker,
                    'Name':           analysis['name'],
                    'Sector':         analysis['sector'],
                    'Industry':       analysis.get('industry', ''),
                    'Shares':         shares,
                    'Current Price':  analysis['price'],
                    'Current Value':  value,
                    'Purchase Price': purchase_price if pd.notna(purchase_price) else None,
                    'Gain/Loss $':    gain,
                    'Gain/Loss %':    gain_pct
                })
            else:
                errors_list.append(f"{ticker}: {error}")
        progress_bar.empty()
        status_text.empty()
        if errors_list:
            with st.expander("⚠️ Errors encountered during analysis"):
                for err in errors_list:
                    st.write(err)
        if results_list:
            df_results = pd.DataFrame(results_list)
            if total_value > 0:
                df_results['Allocation %'] = df_results['Current Value'] / total_value * 100
            sym = "$" if display_currency == "USD" else "€"
            st.metric("Total Current Value", f"{sym}{total_value:,.2f}")
            if total_cost > 0:
                total_gain     = total_value - total_cost
                total_gain_pct = (total_gain / total_cost) * 100
                st.metric("Total Cost",      f"{sym}{total_cost:,.2f}")
                st.metric("Total Gain/Loss", f"{sym}{total_gain:,.2f} ({total_gain_pct:.1f}%)")
            st.subheader("Holdings")
            display_df = df_results.copy()
            display_df['Current Price']  = display_df['Current Price'].apply(lambda x: f"{sym}{x:,.2f}")
            display_df['Current Value']  = display_df['Current Value'].apply(lambda x: f"{sym}{x:,.2f}")
            if 'Purchase Price' in display_df.columns:
                display_df['Purchase Price'] = display_df['Purchase Price'].apply(
                    lambda x: f"{sym}{x:,.2f}" if pd.notna(x) else "N/A")
            if 'Gain/Loss $' in display_df.columns:
                display_df['Gain/Loss $'] = display_df['Gain/Loss $'].apply(
                    lambda x: f"{sym}{x:,.2f}" if pd.notna(x) else "N/A")
                display_df['Gain/Loss %'] = display_df['Gain/Loss %'].apply(
                    lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
            display_df['Allocation %'] = display_df['Allocation %'].apply(
                lambda x: f"{x:.1f}%" if x > 0 else "0%")
            show_cols = ['Ticker', 'Name', 'Sector', 'Industry', 'Shares', 'Current Price',
                         'Current Value', 'Purchase Price', 'Gain/Loss $', 'Gain/Loss %', 'Allocation %']
            show_cols = [c for c in show_cols if c in display_df.columns]
            st.dataframe(display_df[show_cols])
            if 'Sector' in df_results.columns:
                st.subheader("Sector Allocation")
                sector_data = df_results.groupby('Sector')['Current Value'].sum().reset_index()
                fig, ax = plt.subplots()
                ax.pie(sector_data['Current Value'], labels=sector_data['Sector'], autopct='%1.1f%%')
                ax.axis('equal')
                st.pyplot(fig)
                sector_alloc_dict = dict(zip(
                    sector_data['Sector'],
                    sector_data['Current Value'] / total_value * 100))
                suggestions = suggest_diversification(sector_alloc_dict, total_value)
                if suggestions:
                    st.subheader("💡 Quick Diversification Tips")
                    for s in suggestions:
                        st.info(s)
                with st.spinner("📡 Fetching live market intelligence for all sectors... (one-time, cached for 1hr)"):
                    roadmap, overweight, underweight, missing = generate_investment_roadmap(
                        df_results, sector_data, total_value, display_currency)
                render_investment_roadmap(roadmap, overweight, underweight, missing, sym)
        else:
            st.error("No valid tickers could be analyzed. Check the file format and ticker symbols.")
    else:
        st.error("Failed to process the uploaded file. See details above.")
