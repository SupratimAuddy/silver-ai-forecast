import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(layout="wide", page_title="AI Silver Forecast Dashboard")

st_autorefresh(interval=20000, key="live_refresh")

# UNIVERSAL STYLING: Precision Alignment & Read-Only Logic
st.markdown("""
<style>
/* 1. STRICTURE READ-ONLY: Hides all toolbars and developer menus */
[data-testid="stElementToolbar"], 
[data-testid="stTableActionMenu"],
[data-testid="stHeader"] {
    display: none !important;
}

/* 2. THE ALIGNMENT FIX: Baseline Lock at -30px */
[data-testid="stMetricDelta"] {
    position: absolute !important;
    bottom: -30px !important;
    left: 0 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    background-color: transparent !important;
}

/* 3. UNIVERSAL COLOR LOGIC & PILL REMOVAL */
[data-testid="stMetricDelta"] > div {
    background-color: transparent !important;
    padding: 0 !important;
}

/* 4. HIDE TABLE INDEX: Strictly removes the serial number/index column */
.stTable thead tr th:first-child,
.stTable tbody tr th:first-child { 
    display: none !important; 
}

/* Custom Component Styling */
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
.main-title { font-size: 32px; font-weight: 700; margin-bottom: 5px; }
.status-badge { font-size: 16px; margin-bottom: 25px; padding: 5px 12px; border-radius: 5px; display: inline-block; background-color: #1a1a1a; border: 1px solid #333; }
.section-title { font-size: 24px; font-weight: 600; margin-top: 25px; }
.stMetric { background-color: #111; padding: 15px; border-radius: 10px; margin-bottom: 0px; position: relative;}

/* Synchronized High/Low Text */
.high-low-text { font-size: 14px; font-weight: 600; padding-left: 15px; margin-top: -10px; margin-bottom: 20px; }
.hl-green { color: #10b981 !important; }
.hl-red { color: #ef4444 !important; }

.stTable { background-color: transparent; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Silver 999 (1KG) AI Forecast Dashboard</div>", unsafe_allow_html=True)

ist = pytz.timezone("Asia/Kolkata")
now = datetime.now(ist)
st.caption(f"Last Updated: {now.strftime('%d %b %Y | %I:%M:%S %p IST')}")

# ----------------------------------
# MARKET STATE ENGINE
# ----------------------------------
weekday = now.weekday()
hour = now.hour

if weekday >= 5: 
    market_state = "OFF_HOURS"
elif 9 <= hour < 23: 
    market_state = "INDIA_LIVE"
elif hour >= 23 or hour < 3: 
    market_state = "US_LIVE"
else:
    market_state = "OFF_HOURS" 

if market_state == "INDIA_LIVE":
    st.markdown("<div class='status-badge'><span style='color: #10b981;'>🟢 MARKET OPEN (INDIA LIVE)</span></div>", unsafe_allow_html=True)
elif market_state == "US_LIVE":
    st.markdown("<div class='status-badge'><span style='color: #3b82f6;'>🔵 MARKET OPEN (US COMEX LIVE)</span></div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='status-badge'><span style='color: #f59e0b;'>🟠 MARKETS CLOSED (Weekend Monte Carlo Ticker Active)</span></div>", unsafe_allow_html=True)

# ----------------------------------
# DATA FETCHING
# ----------------------------------
@st.cache_data(ttl=3600)
def fetch_verified_market_data(): 
    tickers_mapping = {
        "SI=F": "Silver_USD_oz", "INR=X": "USDINR", "GC=F": "Gold",
        "CL=F": "Crude", "^BSESN": "Sensex", "^TNX": "US_Rate"
    }
    data = yf.download(list(tickers_mapping.keys()), period="2y", interval="1d")
    df_c = data["Close"].copy().rename(columns=tickers_mapping).ffill().dropna()
    df_h = data["High"].copy().rename(columns=tickers_mapping).ffill().dropna()
    df_l = data["Low"].copy().rename(columns=tickers_mapping).ffill().dropna()
    
    latest_usdinr = float(df_c["USDINR"].iloc[-1])
    usd_delta = latest_usdinr - float(df_c["USDINR"].iloc[-2])
    
    premium = 0.07127
    df_c["Ind_Rate"] = (df_c["Silver_USD_oz"] * 32.1507 * df_c["USDINR"]) * (1 + premium)
    df_c["US_Base"] = (df_c["Silver_USD_oz"] * 32.1507 * df_c["USDINR"])
    
    highs = {"Ind": float((df_h["Silver_USD_oz"].iloc[-1] * 32.1507 * latest_usdinr) * (1 + premium)), "US": float(df_h["Silver_USD_oz"].iloc[-1] * 32.1507 * latest_usdinr)}
    lows = {"Ind": float((df_l["Silver_USD_oz"].iloc[-1] * 32.1507 * latest_usdinr) * (1 + premium)), "US": float(df_l["Silver_USD_oz"].iloc[-1] * 32.1507 * latest_usdinr)}
    
    return df_c, highs, lows, latest_usdinr, usd_delta

with st.spinner("Fetching Precise Data..."):
    df_daily, daily_highs, daily_lows, live_usd, usd_delta = fetch_verified_market_data()

actual_current_price = float(df_daily["Ind_Rate"].iloc[-1])
us_price = float(df_daily["US_Base"].iloc[-1])

# ----------------------------------
# OFF-HOURS MONTE CARLO
# ----------------------------------
def calculate_monte_carlo(df, base_price):
    recent_30d = df.tail(30)
    pct_s = recent_30d["Ind_Rate"].pct_change().dropna()
    np.random.seed(datetime.now().minute + datetime.now().hour) 
    shock = (np.random.normal(recent_30d["Gold"].pct_change().mean(), recent_30d["Gold"].pct_change().std()/10) * 0.5)
    return float(base_price * (1 + shock + np.random.normal(pct_s.mean(), pct_s.std()/10)))

sim_ticker_price = calculate_monte_carlo(df_daily, actual_current_price) if market_state == "OFF_HOURS" else actual_current_price
sim_ticker_delta = sim_ticker_price - actual_current_price

# ----------------------------------
# METRICS DISPLAY
# ----------------------------------
def hl_html_inr(high_val, low_val):
    return f"<div class='high-low-text'><span class='hl-green'>🔼 High: ₹{high_val:,.2f}</span> &nbsp;|&nbsp; <span class='hl-red'>🔽 Low: ₹{low_val:,.2f}</span></div>"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Indian Market Rate", f"₹{actual_current_price:,.2f}")
c1.markdown(hl_html_inr(daily_highs['Ind'], daily_lows['Ind']), unsafe_allow_html=True)

c2.metric("US Market Rate", f"₹{us_price:,.2f}")
c2.markdown(hl_html_inr(daily_highs['US'], daily_lows['US']), unsafe_allow_html=True)

# THE TICKER FIX: Dynamic Prefix (+ or -) and Baseline Alignment
delta_prefix = "+" if sim_ticker_delta >= 0 else "-"
c3.metric("AI Simulated Rate", f"₹{sim_ticker_price:,.2f}", f"{delta_prefix}{abs(sim_ticker_delta):,.2f} (vs Indian Market Rate)")
c3.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

usd_prefix = "+" if usd_delta >= 0 else "-"
c4.metric("Live USD/INR Rate", f"₹{live_usd:,.2f}", f"{usd_prefix}{abs(usd_delta):,.2f}")
c4.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# ----------------------------------
# CHART & TABLE
# ----------------------------------
@st.cache_data(ttl=3600)
def run_final_forecast(df): 
    df_p = df.reset_index()[["Date", "Ind_Rate", "Gold", "Crude", "Sensex", "US_Rate"]].rename(columns={"Date": "ds", "Ind_Rate": "y"})
    future = pd.DataFrame({'ds': pd.date_range(start=df_p['ds'].max() + pd.Timedelta(days=1), periods=20)})
    for col in ['Gold', 'Crude', 'Sensex', 'US_Rate']:
        m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True).fit(df_p[['ds', col]].rename(columns={col: 'y'}))
        future[col] = m.predict(future)['yhat'].values
    main_m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
    for col in ['Gold', 'Crude', 'Sensex', 'US_Rate']: main_m.add_regressor(col)
    main_m.fit(df_p)
    forecast = main_m.predict(pd.concat([df_p[['ds', 'Gold', 'Crude', 'Sensex', 'US_Rate']], future]))
    return forecast[forecast['ds'] > df_p['ds'].max()].query("ds.dt.weekday < 5").head(10).assign(Forecast=lambda x: x.yhat.round(2), Date=lambda x: x.ds.dt.strftime("%d %b %Y"))

with st.spinner("Analyzing Cycles..."):
    forecast_next10 = run_final_forecast(df_daily)

st.markdown("<div class='section-title'>Actuals vs Organic 10-Day AI Forecast</div>", unsafe_allow_html=True)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_daily.index[-10:].tolist(), y=df_daily["Ind_Rate"].values[-10:].tolist(), name="Actual History", line=dict(color="#3b82f6", width=2), mode='lines+markers'))
fig.add_trace(go.Scatter(x=forecast_next10["ds"].tolist(), y=forecast_next10["Forecast"].tolist(), name="AI Prediction", line=dict(color="#10b981", dash="dash", width=3), mode='lines+markers'))
fig.update_layout(template="plotly_dark", height=450, legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5), margin=dict(t=40, b=100))
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'staticPlot': True})

st.markdown("<div class='section-title'>AI Prediction</div>", unsafe_allow_html=True)
min_f, max_f = forecast_next10["Forecast"].min(), forecast_next10["Forecast"].max()
f_range = max_f - min_f if max_f != min_f else 1
forecast_next10["Signal"] = forecast_next10["Forecast"].apply(lambda p: "STRONG BUY" if ((p-min_f)/f_range) <= 0.2 else "BUY" if ((p-min_f)/f_range) <= 0.4 else "STRONG SELL" if ((p-min_f)/f_range) >= 0.8 else "SELL" if ((p-min_f)/f_range) >= 0.6 else "HOLD")
forecast_next10["Confidence (%)"] = forecast_next10["Forecast"].apply(lambda p: round(40 + (abs(((p-min_f)/f_range) - 0.5) * 120), 1))

def highlight_signal(val):
    if "BUY" in val: return "background-color:#166534;color:white;font-weight:600;"
    elif "SELL" in val: return "background-color:#991b1b;color:white;font-weight:600;"
    else: return "background-color:#854d0e;color:white;font-weight:600;"

st.table(forecast_next10[["Date", "Forecast", "Signal", "Confidence (%)"]].style.format({"Forecast": "₹{:,.2f}", "Confidence (%)": "{:.1f}%"}).applymap(highlight_signal, subset=["Signal"]))