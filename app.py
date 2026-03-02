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
import os
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

# ─── OPTIONAL DEPS ───────────────────────────
missing_libs = []
try:
    from pypdf import PdfReader
    import docx
    import reticker
    PDF_DOCX_AVAILABLE = True
except ImportError:
    PDF_DOCX_AVAILABLE = False
    missing_libs.append("pypdf python-docx reticker")

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    missing_libs.append("easyocr torch")

# ─── PAGE CONFIG ─────────────────────────────
st.set_page_config(page_title="Strategic AI Investment Architect", layout="wide", page_icon="🏛️")
st.markdown("""
<style>
  [data-testid="stMetricValue"]{font-size:26px!important;font-weight:800!important;color:#1f77b4}
  .phase-card{background:#f4f6f9;color:#1a1a1a;padding:20px;border-radius:10px;border:1px solid #dcdcdc;min-height:440px}
  .news-card{background:#fff;color:#1a1a1a;padding:15px;border-radius:8px;border-left:5px solid #0288d1;margin-bottom:10px;font-size:14px;box-shadow:1px 1px 5px rgba(0,0,0,.1)}
  .fib-box{background:#e3f2fd;color:#0d47a1;padding:10px;border-radius:5px;margin-top:5px;border-left:4px solid #1565c0;font-family:monospace;font-weight:bold}
  .impact-news{background:#fff3e0;border-left:8px solid #ff9800;padding:10px;margin-bottom:8px;border-radius:5px}
  @media(prefers-color-scheme:dark){
    .phase-card{background:#1e2129;color:#fff;border:1px solid #3d414b}
    .news-card{background:#262730;color:#fff;border-left:5px solid #00b0ff}
    .fib-box{background:#0d47a1;color:#e3f2fd;border-left:4px solid #00b0ff}
    .impact-news{background:#332e1f;color:#ffe0b2;border-left:8px solid #ffb74d}
  }
  .impact-announcement{background:#fff3cd;color:#856404;padding:15px;border-radius:5px;border-left:8px solid #ffc107;margin-bottom:20px;font-weight:bold}
  .stop-loss-box{background:#fff1f1;border-left:8px solid #ff4b4b;padding:15px;margin-bottom:20px;color:#b71c1c;font-weight:bold}
  .verdict-box{padding:20px;border-radius:8px;margin-bottom:20px;font-weight:bold;font-size:22px;text-align:center;color:#fff;text-transform:uppercase}
  .v-green{background:#2e7d32}.v-orange{background:#f57c00}.v-red{background:#c62828}
  .disclaimer-container{background:#262730;color:#aaa;padding:15px;border-radius:5px;font-size:12px;margin-bottom:20px;border:1px solid #444}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="disclaimer-container">🚨 <b>LEGAL:</b> Educational tool only. Not financial advice. Fibonacci targets are contingency buy orders. AI projections are mathematical models.</div>', unsafe_allow_html=True)
st.title("🏛️ Strategic AI Investment Architect (V12.0)")
if missing_libs:
    st.sidebar.warning(f"Missing optional libs: pip install {' '.join(missing_libs)}")

# ─── RATE LIMITER ────────────────────────────
class RateLimiter:
    def __init__(self, calls_per_second=0.4):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0
    def wait(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed + random.uniform(0.05, 0.2))
        self.last_call = time.time()

rate_limiter = RateLimiter(0.4)

# ─── EXCHANGE RATE ────────────────────────────
@st.cache_data(ttl=3600)
def get_exchange_rate(from_curr, to_curr):
    if from_curr == to_curr:
        return 1.0
    try:
        rate_limiter.wait()
        data = yf.download(f"{from_curr}{to_curr}=X", period="2d", progress=False, auto_adjust=True)
        if not data.empty:
            close = data["Close"] if "Close" in data.columns else data.iloc[:, 0]
            return float(close.dropna().iloc[-1])
    except Exception:
        pass
    return 1.0

# ─── TICKER RESOLUTION ───────────────────────
@st.cache_data(ttl=3600)
def resolve_smart_ticker(user_input):
    raw = user_input.strip().upper()
    try:
        rate_limiter.wait()
        t = yf.Ticker(raw)
        if t.fast_info.get("lastPrice"):
            name = t.info.get("longName") or t.info.get("shortName") or raw
            return raw, name, "", t.fast_info.get("currency", "USD"), None
        rate_limiter.wait()
        s = yf.Search(raw, max_results=3)
        for res in (s.tickers or []):
            sym = res.get("symbol", "")
            t2 = yf.Ticker(sym)
            if t2.fast_info.get("lastPrice"):
                name = res.get("longname") or res.get("shortname") or sym
                return sym, name, "", t2.fast_info.get("currency", "USD"), None
    except Exception as e:
        return raw, raw, "", "USD", str(e)
    return raw, raw, "", "USD", "Ticker not found"

# ─── HISTORICAL DATA ─────────────────────────
@st.cache_data(ttl=1800)
def fetch_ohlcv(ticker, suffix):
    for attempt in range(2):
        try:
            rate_limiter.wait()
            raw = yf.download(ticker, period="5y", progress=False, auto_adjust=True)
            if not raw.empty:
                df = raw.reset_index()
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                df = df.rename(columns={"Date": "ds", "Close": "y", "Volume": "vol",
                                        "Open": "open", "High": "high", "Low": "low"})
                df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
                df = df.sort_values("ds").reset_index(drop=True)
                df["y"] = pd.to_numeric(df["y"], errors="coerce")
                df["vol"] = pd.to_numeric(df.get("vol", 0), errors="coerce").fillna(0)
                df = df.dropna(subset=["y"])
                if len(df) >= 60:
                    return df, None
        except Exception as e:
            if attempt == 1:
                return None, str(e)
        time.sleep(1)
    try:
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=1825)
        raw = web.DataReader(f"{ticker}{suffix}", "stooq", start, end)
        if not raw.empty:
            df = raw.reset_index().rename(columns={"Date": "ds", "Close": "y",
                                                    "Volume": "vol", "Open": "open",
                                                    "High": "high", "Low": "low"})
            df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
            df = df.sort_values("ds").reset_index(drop=True)
            df["y"] = pd.to_numeric(df["y"], errors="coerce")
            df = df.dropna(subset=["y"])
            return df, None
    except Exception as e:
        return None, str(e)
    return None, "No data from any source."

# ─── FUNDAMENTALS ────────────────────────────
@st.cache_data(ttl=3600)
def get_fundamental_health(ticker):
    h = {"ROE": 0, "Debt": 0, "PB": 0, "Margin": "N/A",
         "CurrentRatio": "N/A", "EPS_growth": "N/A", "ForwardPE": "N/A"}
    try:
        info = yf.Ticker(ticker).info
        h["ROE"] = info.get("returnOnEquity", 0) or 0
        h["Debt"] = info.get("debtToEquity", 0) or 0
        h["PB"] = info.get("priceToBook", 0) or 0
        h["Margin"] = f"{(info.get('profitMargins', 0) or 0)*100:.1f}%"
        h["CurrentRatio"] = str(info.get("currentRatio", "N/A"))
        h["EPS_growth"] = f"{(info.get('earningsGrowth', 0) or 0)*100:.1f}%"
        h["ForwardPE"] = str(info.get("forwardPE", "N/A"))
    except Exception:
        pass
    try:
        soup = BeautifulSoup(
            requests.get(f"https://finviz.com/quote.ashx?t={ticker}",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=10).text, "html.parser")
        def fvz(label):
            td = soup.find("td", string=label)
            return td.find_next_sibling("td").text.strip() if td else None
        for key, label, div in [("ROE","ROE",100),("PB","P/B",1),("Debt","Debt/Eq",1)]:
            v = fvz(label)
            if v and v not in ("-",""):
                try: h[key] = float(v.replace("%","")) / div
                except ValueError: pass
        for key, label in [("CurrentRatio","Current Ratio"),("Margin","Profit Margin")]:
            v = fvz(label)
            if v: h[key] = v
    except Exception:
        pass
    return h

# ─── TECHNICAL INDICATORS ────────────────────
def calculate_technicals(df):
    delta = df["y"].diff()
    gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_val = float((100 - (100 / (1 + rs))).iloc[-1])

    cur = float(df["y"].iloc[-1])
    diff = max(cur - float(df["y"].tail(126).min()), cur * 0.05)
    fibs = {k: cur - diff * v for k, v in
            [("0.236",0.236),("0.382",0.382),("0.500",0.500),("0.618",0.618),("0.786",0.786)]}

    if "high" in df.columns and "low" in df.columns:
        h, l, cp = df["high"], df["low"], df["y"].shift(1)
        tr = pd.concat([h-l,(h-cp).abs(),(l-cp).abs()],axis=1).max(axis=1)
        atr = float(tr.ewm(span=14,adjust=False).mean().iloc[-1])
    else:
        atr = cur * 0.02

    ema12 = df["y"].ewm(span=12,adjust=False).mean()
    ema26 = df["y"].ewm(span=26,adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9,adjust=False).mean()

    ma20 = df["y"].rolling(20).mean()
    std20 = df["y"].rolling(20).std()

    df = df.copy()
    df["ma50"] = df["y"].rolling(50).mean()
    df["ma200"] = df["y"].rolling(200).mean()

    techs = {
        "rsi": rsi_val, "fib": fibs, "atr": atr,
        "macd": float(macd_line.iloc[-1]),
        "macd_signal": float(signal_line.iloc[-1]),
        "macd_hist": float((macd_line - signal_line).iloc[-1]),
        "bb_upper": float((ma20 + 2*std20).iloc[-1]),
        "bb_lower": float((ma20 - 2*std20).iloc[-1]),
        "bb_mid": float(ma20.iloc[-1]),
    }
    return techs, df

def detect_crossover(df):
    is_death = float(df["ma50"].iloc[-1]) < float(df["ma200"].iloc[-1])
    cross_point = None
    msg = "Trend stable – no recent crossover."
    for i in range(max(1, len(df)-60), len(df)):
        p = i - 1
        if df["ma50"].iloc[p] < df["ma200"].iloc[p] and df["ma50"].iloc[i] >= df["ma200"].iloc[i]:
            cross_point = (df["ds"].iloc[i], float(df["ma50"].iloc[i]), "GOLDEN")
            msg = "🚀 GOLDEN CROSS: 50-Day MA crossed ABOVE 200-Day MA."
        elif df["ma50"].iloc[p] > df["ma200"].iloc[p] and df["ma50"].iloc[i] <= df["ma200"].iloc[i]:
            cross_point = (df["ds"].iloc[i], float(df["ma50"].iloc[i]), "DEATH")
            msg = "⚠️ DEATH CROSS: 50-Day MA crossed BELOW 200-Day MA."
    return msg, cross_point, is_death

# ─── PROPHET FORECAST ────────────────────────
def run_prophet(df, periods=180):
    try:
        m = Prophet(
            daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True,
            changepoint_prior_scale=0.05,       # FIX: was 0.08, reduces overfitting
            changepoint_range=0.85,
            seasonality_mode="multiplicative",   # FIX: better for stock price data
            interval_width=0.80,
        )
        m.fit(df[["ds","y"]].copy())
        future = m.make_future_dataframe(periods=periods, freq="B")  # FIX: business days only
        fc = m.predict(future)
        hist_min, hist_max = float(df["y"].min()), float(df["y"].max())
        for col in ["yhat","yhat_lower","yhat_upper"]:
            fc[col] = fc[col].clip(hist_min*0.4, hist_max*2.5)
        return fc, m
    except Exception as e:
        st.warning(f"Prophet failed: {e}")
        return None, None

# ─── NEWS & SENTIMENT ────────────────────────
@st.cache_data(ttl=1800)
def get_enhanced_news(ticker):
    headlines, seen = [], set()
    def add(headline, source, ts=None):
        if not headline or headline in seen: return
        seen.add(headline)
        dt = datetime.datetime.fromtimestamp(ts) if isinstance(ts,(int,float)) else datetime.datetime.now()
        headlines.append({"date": dt, "headline": headline,
                           "sentiment": TextBlob(headline).sentiment.polarity, "source": source})
    try:
        soup = BeautifulSoup(requests.get(f"https://finviz.com/quote.ashx?t={ticker}",
            headers={"User-Agent":"Mozilla/5.0"},timeout=10).text,"html.parser")
        tbl = soup.find(id="news-table")
        if tbl:
            for row in tbl.find_all("tr")[:15]:
                a = row.find("a")
                if a and a.text: add(a.text.strip(),"Finviz")
    except Exception: pass
    try:
        rate_limiter.wait()
        for item in (yf.Ticker(ticker).news or [])[:15]:
            title = item.get("content",{}).get("title") or item.get("title","")
            ts = item.get("content",{}).get("pubDate") or item.get("providerPublishTime")
            if isinstance(ts,str):
                try: ts = int(datetime.datetime.fromisoformat(ts.replace("Z","+00:00")).timestamp())
                except: ts = None
            add(title,"Yahoo",ts)
    except Exception: pass
    headlines.sort(key=lambda x: x["date"], reverse=True)
    return headlines

def filter_impactful_news(news_list, threshold=0.25):
    keywords = {"earnings","revenue","guidance","forecast","beat","miss","dividend","buyback",
                "acquisition","merger","ipo","split","fed","rate","inflation","recession",
                "lawsuit","sec","fine","ai","product","launch","upgrade","downgrade","target",
                "layoff","bankruptcy","partnership","contract","deal"}
    return [n for n in news_list
            if abs(n["sentiment"]) > threshold
            or any(kw in n["headline"].lower() for kw in keywords)]

# ─── VOLUME TREND ────────────────────────────
def volume_trend_message(df):
    if len(df) < 20: return "Insufficient volume data."
    df = df.copy()
    df["pc"] = df["y"].diff()
    up, dn = df[df["pc"] > 0], df[df["pc"] < 0]
    if not len(up) or not len(dn): return "Neutral volume pattern."
    ur = float((up["vol"] > up["vol"].mean()).mean())
    dr = float((dn["vol"] > dn["vol"].mean()).mean())
    if ur > 0.55 and dr < 0.45: return "✅ Volume confirms uptrend – accumulation phase."
    elif ur < 0.45 and dr > 0.55: return "⚠️ Volume divergence – distribution / bearish."
    elif ur > 0.55 and dr > 0.55: return "⚖️ High volume both ways – high uncertainty."
    return "➡️ Neutral volume trend."

# ─── CONVICTION SCORE ────────────────────────
def compute_conviction(health, techs, ai_roi_30, avg_sentiment, is_death, deviation):
    bd = {}
    bd["Trend (MA)"] = 0 if is_death else 25
    pb = health.get("PB",0) or 0
    bd["Valuation (P/B)"] = 15 if 0<pb<3 else 8 if pb<5 else 0
    roe = health.get("ROE",0) or 0
    bd["Profitability (ROE)"] = 20 if roe>0.20 else 12 if roe>0.10 else 5 if roe>0 else 0
    bd["AI Momentum (30d)"] = 25 if ai_roi_30>5 else 15 if ai_roi_30>1 else 5 if ai_roi_30>-2 else 0
    bd["News Sentiment"] = 10 if avg_sentiment>0.1 else 5 if avg_sentiment>0 else 0
    bd["Overextension"] = -5 if deviation > 20 else 0
    rsi = techs.get("rsi",50)
    bd["RSI Signal"] = 5 if 45<rsi<65 else 2 if rsi<35 else 0
    return max(0,min(100,sum(bd.values()))), bd

# ─── STOP LOSS (ATR + BB + %) ─────────────────
def compute_stop_loss(cur_p, techs):
    atr = techs.get("atr", cur_p*0.02)
    rsi = techs.get("rsi", 50)
    bb_lower = techs.get("bb_lower", cur_p*0.95)
    atr_stop = cur_p - 2.0*atr
    pct_stop = cur_p * (0.92 if rsi>65 else 0.85 if rsi<30 else 0.88)
    stop = max(atr_stop, bb_lower, pct_stop)
    return {"stop": stop, "risk_pct": (cur_p-stop)/cur_p*100, "atr_stop": atr_stop, "bb_stop": bb_lower}

def kelly_position(win_prob, win_loss_ratio, capital, max_pct=0.25):
    if win_loss_ratio <= 0: return 0
    kelly = win_prob - (1-win_prob)/win_loss_ratio
    return capital * max(0, min(kelly, max_pct))

# ─── PORTFOLIO EXTRACTOR ─────────────────────
class PortfolioExtractor:
    def __init__(self):
        self.ocr_reader = None
        self._te = None
        if PDF_DOCX_AVAILABLE:
            try:
                self._te = reticker.TickerExtractor(deduplicate=True,
                    match_config=reticker.TickerMatchConfig(
                        prefixed_uppercase=True, unprefixed_uppercase=True,
                        prefixed_lowercase=True, prefixed_titlecase=True, separators="-."))
            except Exception: pass

    def _ocr(self, path):
        if not OCR_AVAILABLE: return "", "OCR not available"
        try:
            if self.ocr_reader is None: self.ocr_reader = easyocr.Reader(["en"], gpu=False)
            return "\n".join(self.ocr_reader.readtext(path, detail=0, paragraph=True)), None
        except Exception as e: return "", str(e)

    def _extract_text(self, path):
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            try: return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages), None
            except Exception as e: return "", str(e)
        elif ext in (".docx",".doc"):
            try: return "\n".join(p.text for p in docx.Document(path).paragraphs), None
            except Exception as e: return "", str(e)
        elif ext in (".png",".jpg",".jpeg",".tiff",".bmp"): return self._ocr(path)
        return "", "Unsupported"

    @staticmethod
    def _find_tickers_regex(text):
        bl = {"ETF","USD","EUR","GBP","JPY","CAD","AUD","CHF","THE","AND","FOR","INC","LTD",
              "LLC","NYSE","NASDAQ","SEC","IPO","CEO","CFO","CTO","ESG","RSI","ATR","N/A",
              "YTD","QTD","EPS","PE","AI","IT","US","UK"}
        return list(set(m for m in re.findall(r'\b[A-Z]{1,5}\b', text) if m not in bl))

    @staticmethod
    def _extract_qty(text, ticker):
        for pat in [rf'{ticker}\s+(\d[\d,]*(?:\.\d+)?)',
                    rf'(\d[\d,]*(?:\.\d+)?)\s+{ticker}',
                    rf'(\d[\d,]*(?:\.\d+)?)\s+shares?\s+of\s+{ticker}']:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try: return float(m.group(1).replace(",",""))
                except ValueError: pass
        return 1.0

    def process_file(self, file_path):
        text, err = self._extract_text(file_path)
        if err: return [], f"Error: {err}", ""
        if not text.strip(): return [], "No text extracted.", ""
        preview = text[:1200] + "..." if len(text) > 1200 else text
        raw_tickers = (self._te.extract(text) if self._te else self._find_tickers_regex(text))
        raw_tickers = list(set(raw_tickers))[:20]
        holdings, errors = [], []
        for t in raw_tickers:
            try:
                rate_limiter.wait()
                info = yf.Ticker(t).info
                price = info.get("regularMarketPrice") or info.get("currentPrice")
                if price:
                    holdings.append({"ticker":t,"name":info.get("longName",t) or t,
                                     "shares":self._extract_qty(text,t),
                                     "sector":info.get("sector","Unknown"),"current_price":price})
                else: errors.append(f"{t}: no price")
            except Exception as e: errors.append(f"{t}: {e}")
        return holdings, preview, f"Found {len(raw_tickers)}, validated {len(holdings)}. Errors: {errors}"

# ─── FILE PROCESSOR ──────────────────────────
def process_uploaded_file(uploaded_file):
    ext = Path(uploaded_file.name.lower()).suffix
    if ext in (".csv",".xlsx",".xls"):
        try:
            df = pd.read_csv(uploaded_file) if ext == ".csv" else pd.read_excel(uploaded_file)
            df.columns = df.columns.str.strip().str.lower()
            if "ticker" not in df.columns: st.error("Must have 'Ticker' column."); return None
            if "shares" not in df.columns: df["shares"] = 1
            if "purchase price" in df.columns:
                df["purchase price"] = pd.to_numeric(df["purchase price"], errors="coerce")
            return {"type":"tabular","data":df}
        except Exception as e: st.error(f"Read error: {e}"); return None
    elif ext in (".pdf",".docx",".doc",".png",".jpg",".jpeg",".tiff",".bmp"):
        if ext in (".pdf",".docx",".doc") and not PDF_DOCX_AVAILABLE:
            st.error("PDF/Word support not installed."); return None
        extractor = PortfolioExtractor()
        tmp = f"/tmp/upload_{uploaded_file.name}"
        with open(tmp,"wb") as f: f.write(uploaded_file.getbuffer())
        holdings, preview, msg = extractor.process_file(tmp)
        os.remove(tmp)
        with st.expander("Extraction Details"): st.info(msg); st.code(preview[:1000])
        if holdings: return {"type":"tabular","data":pd.DataFrame(holdings)}
        st.warning("No valid tickers found."); return None
    else: st.error("Unsupported file type."); return None

# ─── DIVERSIFICATION ─────────────────────────
SECTOR_EXAMPLES = {
    "Technology":["AAPL","MSFT","QQQ"],"Healthcare":["JNJ","UNH","XLV"],
    "Financial Services":["JPM","BAC","XLF"],"Consumer Cyclical":["AMZN","TSLA","XLY"],
    "Industrials":["HON","CAT","XLI"],"Energy":["XOM","CVX","XLE"],
    "Utilities":["NEE","DUK","XLU"],"Real Estate":["PLD","AMT","XLRE"],
    "Communication Services":["META","GOOGL","XLC"],"Consumer Defensive":["PG","KO","XLP"],
    "Basic Materials":["LIN","BHP","XLB"],
}
def suggest_diversification(sector_alloc):
    tips = []
    for sector, examples in SECTOR_EXAMPLES.items():
        alloc = sector_alloc.get(sector,0)
        ex = ", ".join(examples[:3])
        if alloc == 0: tips.append(f"💡 No exposure to **{sector}**. Consider: {ex}")
        elif alloc < 8: tips.append(f"💡 Low weight ({alloc:.1f}%) in **{sector}**. Consider: {ex}")
    return tips[:6]

@st.cache_data(ttl=3600)
def analyze_ticker_basic(ticker, display_currency):
    try:
        rate_limiter.wait()
        info = yf.Ticker(ticker).info
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        if not price: return None, "No price"
        fx = get_exchange_rate(info.get("currency","USD"), display_currency)
        return {"name":info.get("longName",ticker) or ticker,
                "sector":info.get("sector","Unknown"),
                "price":price*fx,"beta":info.get("beta",1.0) or 1.0}, None
    except Exception as e: return None, str(e)

# ─── SIDEBAR ─────────────────────────────────
st.sidebar.header("⚙️ Configuration")
user_query = st.sidebar.text_input("Ticker / Company Name", value="NVDA")
display_currency = st.sidebar.selectbox("Display Currency", ["USD","EUR","GBP"])
total_capital = st.sidebar.number_input("Available Capital", value=10000, min_value=100, step=500)
forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 30, 365, 180, step=30)
st.sidebar.markdown("---")
st.sidebar.header("📁 Upload Portfolio")
uploaded_file = st.sidebar.file_uploader("CSV, Excel, PDF, Word, Image",
    type=["csv","xlsx","xls","pdf","docx","doc","png","jpg","jpeg","tiff","bmp"])

# ─── DEEP AUDIT ──────────────────────────────
if st.sidebar.button("🚀 Run Deep Audit"):
    with st.spinner(f"Analysing {user_query}..."):
        ticker, name, suffix, native_curr, resolve_err = resolve_smart_ticker(user_query)
        if resolve_err: st.warning(f"Ticker resolution: {resolve_err}")
        display_name = name or ticker

        df, hist_err = fetch_ohlcv(ticker, suffix)
        if hist_err: st.error(f"Data error: {hist_err}"); st.stop()
        if df is None or len(df) < 60: st.error("Not enough data."); st.stop()

        fx = get_exchange_rate(native_curr, display_currency)
        sym = {"USD":"$","EUR":"€","GBP":"£"}.get(display_currency,"$")
        cur_p = float(df["y"].iloc[-1]) * fx

        techs, df = calculate_technicals(df)
        cross_msg, cross_point, is_death = detect_crossover(df)

        fc, prophet_model = run_prophet(df, periods=forecast_horizon)
        if fc is not None:
            last_ds = df["ds"].iloc[-1]
            row_now = fc[fc["ds"] == last_ds]
            trend_val = float(row_now["yhat"].values[0])*fx if not row_now.empty else cur_p
            deviation = (cur_p - trend_val) / trend_val * 100
            fair_low, fair_high = trend_val*0.95, trend_val*1.05
            future_rows = fc[fc["ds"] > last_ds]
            target_30 = float(future_rows.iloc[min(29,len(future_rows)-1)]["yhat"])*fx if len(future_rows) else cur_p
            target_horizon = float(future_rows.iloc[-1]["yhat"])*fx if len(future_rows) else cur_p
        else:
            trend_val=cur_p; deviation=0.0; fair_low,fair_high=cur_p*0.95,cur_p*1.05
            target_30,target_horizon=cur_p*1.03,cur_p*1.08

        if is_death: target_30 *= 0.97
        ai_roi_30 = (target_30 - cur_p) / cur_p * 100

        health = get_fundamental_health(ticker)
        all_news = get_enhanced_news(ticker)
        impactful = filter_impactful_news(all_news)
        avg_sentiment = float(np.mean([n["sentiment"] for n in impactful])) if impactful else 0.0

        score, breakdown = compute_conviction(health, techs, ai_roi_30, avg_sentiment, is_death, deviation)
        sl = compute_stop_loss(cur_p, techs)
        avg_win = max(abs(ai_roi_30)/100, 0.02)
        kelly_amt = kelly_position(score/100, avg_win/max(sl["risk_pct"]/100,0.01), total_capital)

        if score >= 70: verdict,v_col,action = "Strong Buy","v-green","ACTION: BUY NOW"
        elif score >= 45: verdict,v_col,action = "Hold / Neutral","v-orange","ACTION: MONITOR"
        else: verdict,v_col,action = "Sell / Avoid","v-red","ACTION: SELL / STAY AWAY"

        st.subheader(f"📊 {display_name} ({ticker}) · {display_currency}")
        c1,c2 = st.columns(2)
        with c1:
            (st.warning if deviation > 15 else st.success)(
                f"{'⚠️ MEAN REVERSION RISK: +' if deviation>15 else '✅ TREND ALIGNED: '}{deviation:.1f}% vs AI trend.")
        with c2:
            st.info(f"💎 AI FAIR VALUE: {sym}{fair_low:,.2f} – {sym}{fair_high:,.2f}")

        st.markdown(f'<div class="impact-announcement">{cross_msg}</div>', unsafe_allow_html=True)

        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("Conviction", f"{score}/100")
        m2.metric("Current Price", f"{sym}{cur_p:,.2f}")
        m3.metric("30d Target", f"{sym}{target_30:,.2f}", f"{ai_roi_30:+.1f}%")
        m4.metric(f"{forecast_horizon}d Target", f"{sym}{target_horizon:,.2f}")
        m5.metric("RSI(14)", f"{techs['rsi']:.0f}")

        st.markdown(f'<div class="verdict-box {v_col}">Strategic Verdict: {verdict} | {action}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stop-loss-box">🛑 DYNAMIC STOP LOSS: {sym}{sl["stop"]:,.2f} ({sl["risk_pct"]:.1f}% risk) | ATR: {sym}{sl["atr_stop"]:,.2f} | BB: {sym}{sl["bb_stop"]:,.2f}</div>', unsafe_allow_html=True)

        with st.expander("🔍 Conviction Score Breakdown"):
            for factor, pts in breakdown.items():
                colour = "#2e7d32" if pts>0 else "#c62828" if pts<0 else "#888"
                st.markdown(f"**{factor}**: {pts:+d} pts "
                            f'<span style="display:inline-block;width:{abs(pts)/25*100:.0f}%;height:12px;background:{colour};border-radius:4px;vertical-align:middle"></span>',
                            unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("### 🏥 Fundamentals")
            st.table(pd.DataFrame([
                ("ROE",f"{health['ROE']*100:.1f}%","✅" if health["ROE"]>0.15 else "⚠️"),
                ("P/B",f"{health['PB']:.2f}x","✅" if 0<health["PB"]<3 else "⚠️"),
                ("Debt/Equity",f"{health['Debt']:.2f}","✅" if health["Debt"]<1.0 else "⚠️"),
                ("Profit Margin",health["Margin"],""),
                ("Current Ratio",health["CurrentRatio"],""),
                ("Forward P/E",health["ForwardPE"],""),
                ("EPS Growth",health["EPS_growth"],""),
            ], columns=["Metric","Value","Signal"]))
            st.markdown("### 📐 Technicals")
            bb_pos = ("🔴 Overbought" if cur_p/fx>techs["bb_upper"] else
                      "🟢 Oversold" if cur_p/fx<techs["bb_lower"] else "⚪ Mid-range")
            st.table(pd.DataFrame([
                ("RSI(14)",f"{techs['rsi']:.1f}","⚠️ OB" if techs["rsi"]>70 else "✅ OS" if techs["rsi"]<30 else "✅ OK"),
                ("MACD",f"{techs['macd']:.3f}","📈" if techs["macd_hist"]>0 else "📉"),
                ("Bollinger",bb_pos,""),
                ("ATR(14)",f"{sym}{techs['atr']*fx:.2f}","Volatility"),
            ], columns=["Indicator","Value","Signal"]))
            st.markdown("### 📰 Headlines")
            for n in all_news[:6]:
                e = "🟢" if n["sentiment"]>0.05 else "🔴" if n["sentiment"]<-0.05 else "⚪"
                st.markdown(f'<div class="news-card">{e} {n["headline"]}</div>', unsafe_allow_html=True)

        with col_r:
            st.markdown("### ⚖️ Strategy")
            fib = techs["fib"]
            rr = abs(ai_roi_30)/max(sl["risk_pct"],0.1)
            st.markdown(f"""<div class="phase-card">
<h4 style="color:#1f77b4">PHASE 1 – IMMEDIATE ENTRY</h4>
<p><b>Kelly Allocation:</b> {sym}{kelly_amt:,.2f} ({kelly_amt/total_capital*100:.1f}% of capital)</p>
<hr>
<h4 style="color:#1f77b4">PHASE 2 – FIBONACCI RETRACEMENT TARGETS</h4>
<div class="fib-box">0.236 → {sym}{fib['0.236']*fx:,.2f}</div>
<div class="fib-box">0.382 → {sym}{fib['0.382']*fx:,.2f}</div>
<div class="fib-box">0.500 → {sym}{fib['0.500']*fx:,.2f}</div>
<div class="fib-box">0.618 → {sym}{fib['0.618']*fx:,.2f}</div>
<div class="fib-box">0.786 → {sym}{fib['0.786']*fx:,.2f}</div>
<hr>
<h4 style="color:#1f77b4">RISK/REWARD</h4>
<p>Stop Loss: {sym}{sl['stop']:,.2f} | Risk: {sl['risk_pct']:.1f}%</p>
<p><b>R/R Ratio (30d): {rr:.2f}x</b></p>
</div>""", unsafe_allow_html=True)

        if impactful:
            st.markdown("### 🔥 High-Impact News")
            for n in impactful[:7]:
                e = "🟢" if n["sentiment"]>0.1 else "🔴" if n["sentiment"]<-0.1 else "⚪"
                st.markdown(f'<div class="impact-news"><b>{e} {n["headline"]}</b><br>'
                            f'<span style="font-size:.85rem;">{n["date"].strftime("%b %d, %Y")} · {n["source"]} · {n["sentiment"]:+.2f}</span></div>',
                            unsafe_allow_html=True)

        st.markdown("---")
        st.subheader(f"🤖 AI Forecast ({forecast_horizon} Business Days)")
        fig, ax = plt.subplots(figsize=(13,6))
        if fc is not None and prophet_model is not None:
            fc_plot = fc.copy()
            for col in ["yhat","yhat_lower","yhat_upper"]: fc_plot[col] *= fx
            prophet_model.plot(fc_plot, ax=ax)
        else:
            ax.plot(df["ds"], df["y"]*fx, "b-", lw=1.5, label="Historical")
        ax.plot(df["ds"], df["ma50"]*fx, label="50d MA", color="orange", lw=2)
        ax.plot(df["ds"], df["ma200"]*fx, label="200d MA", color="red", lw=2)
        ma20 = df["y"].rolling(20).mean(); std20 = df["y"].rolling(20).std()
        ax.fill_between(df["ds"],(ma20-2*std20)*fx,(ma20+2*std20)*fx, alpha=0.1, color="purple", label="BB(2σ)")
        if cross_point: ax.scatter(cross_point[0],cross_point[1]*fx,color="gold",s=300,marker="*",zorder=5,label=f"{cross_point[2]} CROSS")
        ax.axhline(sl["stop"],color="#ff4b4b",linestyle="--",lw=1.5,label=f"Stop {sym}{sl['stop']:,.2f}")
        ax.set_xlim([datetime.datetime.now()-datetime.timedelta(days=180),
                     datetime.datetime.now()+datetime.timedelta(days=forecast_horizon)])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        plt.xticks(rotation=30); plt.legend(loc="upper left",fontsize=8); plt.tight_layout()
        st.pyplot(fig)

        st.subheader("📉 MACD (12/26/9)")
        mfig, max_ = plt.subplots(figsize=(13,3))
        e12=df["y"].ewm(span=12,adjust=False).mean(); e26=df["y"].ewm(span=26,adjust=False).mean()
        ml=e12-e26; sl_=ml.ewm(span=9,adjust=False).mean(); hl=ml-sl_
        rec=df.tail(252)
        max_.bar(rec["ds"],hl.tail(252),color=["#2e7d32" if v>=0 else "#c62828" for v in hl.tail(252)],alpha=0.7)
        max_.plot(rec["ds"],ml.tail(252),color="blue",lw=1.2,label="MACD")
        max_.plot(rec["ds"],sl_.tail(252),color="orange",lw=1.2,label="Signal")
        max_.axhline(0,color="black",lw=0.5,linestyle="--")
        max_.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        plt.xticks(rotation=30); plt.legend(fontsize=8); plt.tight_layout(); st.pyplot(mfig)

        st.subheader("📊 12-Month Volume")
        vfig,vax=plt.subplots(figsize=(13,3))
        vdf=df.tail(252).copy()
        vc=["#2e7d32" if i>0 and vdf["y"].iloc[i]>=vdf["y"].iloc[i-1] else "#c62828" for i in range(len(vdf))]
        vax.bar(vdf["ds"],vdf["vol"],color=vc,alpha=0.7)
        vax.set_ylabel("Shares Traded")
        vax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        plt.xticks(rotation=30); plt.tight_layout(); st.pyplot(vfig)
        st.info(f"📈 Volume Insight: {volume_trend_message(vdf)}")

# ─── PORTFOLIO ANALYSIS ──────────────────────
if uploaded_file is not None:
    st.markdown("---"); st.header("📁 Portfolio Analysis")
    with st.spinner("Processing..."): result = process_uploaded_file(uploaded_file)
    if result and result["type"] == "tabular":
        df_h = result["data"]
        rows,errs=[],[]
        total_val=total_cost=0.0
        bar=st.progress(0); status=st.empty()
        for idx,row in df_h.iterrows():
            tkr=str(row["ticker"]).strip().upper()
            shares=float(row.get("shares",1))
            pp=row.get("purchase price",None)
            status.text(f"Analysing {tkr}... ({idx+1}/{len(df_h)})")
            bar.progress((idx+1)/len(df_h))
            ana,err=analyze_ticker_basic(tkr,display_currency)
            if ana:
                val=shares*ana["price"]; total_val+=val
                gain=gain_pct=None
                if pp is not None and pd.notna(pp):
                    cost=shares*float(pp); total_cost+=cost
                    gain=val-cost; gain_pct=gain/cost*100 if cost else 0
                rows.append({"Ticker":tkr,"Name":ana["name"],"Sector":ana["sector"],
                             "Shares":shares,"Price":ana["price"],"Value":val,
                             "Purchase Price":pp if pp is not None and pd.notna(pp) else None,
                             "Gain/Loss $":gain,"Gain/Loss %":gain_pct,"Beta":ana["beta"]})
            else: errs.append(f"{tkr}: {err}")
        bar.empty(); status.empty()
        if errs:
            with st.expander("⚠️ Errors"): st.write("\n".join(errs))
        if rows:
            df_r=pd.DataFrame(rows)
            sym={"USD":"$","EUR":"€","GBP":"£"}.get(display_currency,"$")
            df_r["Alloc %"]=df_r["Value"]/total_val*100
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Total Value",f"{sym}{total_val:,.2f}")
            if total_cost>0:
                tg=total_val-total_cost
                c2.metric("Total Cost",f"{sym}{total_cost:,.2f}")
                c3.metric("Gain/Loss",f"{sym}{tg:,.2f}",f"{tg/total_cost*100:+.1f}%")
            c4.metric("Portfolio Beta",f"{float(np.average(df_r['Beta'].fillna(1),weights=df_r['Value'])):.2f}")
            st.subheader("Holdings")
            disp=df_r.copy()
            for col in ["Price","Value"]: disp[col]=disp[col].apply(lambda x: f"{sym}{x:,.2f}")
            disp["Purchase Price"]=disp["Purchase Price"].apply(lambda x: f"{sym}{x:,.2f}" if pd.notna(x) else "N/A")
            disp["Gain/Loss $"]=disp["Gain/Loss $"].apply(lambda x: f"{sym}{x:,.2f}" if pd.notna(x) else "N/A")
            disp["Gain/Loss %"]=disp["Gain/Loss %"].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A")
            disp["Alloc %"]=disp["Alloc %"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(disp.drop(columns=["Beta"],errors="ignore"))
            ch1,ch2=st.columns(2)
            with ch1:
                st.subheader("Sector Allocation")
                sec=df_r.groupby("Sector")["Value"].sum().reset_index()
                fig,ax=plt.subplots()
                ax.pie(sec["Value"],labels=sec["Sector"],autopct="%1.1f%%",startangle=140)
                ax.axis("equal"); st.pyplot(fig)
            with ch2:
                st.subheader("Top Holdings")
                top=df_r.nlargest(10,"Value")
                fig2,ax2=plt.subplots()
                ax2.barh(top["Ticker"],top["Value"]); ax2.set_xlabel(f"Value ({sym})")
                ax2.invert_yaxis(); plt.tight_layout(); st.pyplot(fig2)
            sec_alloc=dict(zip(sec["Sector"],sec["Value"]/total_val*100))
            tips=suggest_diversification(sec_alloc)
            if tips:
                st.subheader("💡 Diversification Suggestions")
                for t in tips: st.info(t)
            top1=float(df_r["Alloc %"].max())
            if top1>25: st.warning(f"⚠️ Concentration risk: largest position is {top1:.1f}% of portfolio.")
        else:
            st.error("No valid tickers analysed.")
