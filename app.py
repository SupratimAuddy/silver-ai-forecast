import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime
import pytz
from streamlit_autorefresh import st_autorefresh

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(layout="wide")

# -------------------------------------------------
# CUSTOM STYLING (Readability Boost)
# -------------------------------------------------
st.markdown("""
<style>
.big-font { font-size:28px !important; font-weight:600; }
.metric-label { font-size:16px !important; color:#bbb; }
.section-title { font-size:26px !important; font-weight:700; margin-top:30px; }
.sub-section { font-size:20px !important; font-weight:600; margin-top:20px; }
.table-title { font-size:22px !important; font-weight:700; margin-top:30px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-font'>Silver 999 (1KG) Live + AI Forecast Dashboard</div>", unsafe_allow_html=True)

# -------------------------------------------------
# AUTO REFRESH (60 sec)
# -------------------------------------------------
st_autorefresh(interval=60000, key="live_refresh")

TOTAL_EXTRA = 0.05  # 5% Indian loading

# -------------------------------------------------
# TIME
# -------------------------------------------------
ist = pytz.timezone("Asia/Kolkata")
now = datetime.now(ist)
st.caption(f"Last Updated: {now.strftime('%d %b %Y | %I:%M:%S %p IST')}")

# -------------------------------------------------
# DAILY DATA (Cached)
# -------------------------------------------------
@st.cache_data(ttl=3600)
def load_daily_data():
    silver = yf.download("SI=F", period="365d", interval="1d")["Close"]
    usdinr = yf.download("INR=X", period="365d", interval="1d")["Close"]

    df = pd.concat([silver, usdinr], axis=1)
    df.columns = ["Silver_USD", "USDINR"]
    df.dropna(inplace=True)

    df["International_INR_KG"] = df["Silver_USD"] * 32.1507 * df["USDINR"]
    df["Indian_Physical_INR_KG"] = df["International_INR_KG"] * (1 + TOTAL_EXTRA)

    return df

df_daily = load_daily_data()

# -------------------------------------------------
# LIVE DATA
# -------------------------------------------------
silver_live = yf.download("SI=F", period="2d", interval="1m")["Close"]
usdinr_live = yf.download("INR=X", period="2d", interval="1m")["Close"]

df_live = pd.concat([silver_live, usdinr_live], axis=1)
df_live.columns = ["Silver_USD", "USDINR"]
df_live.dropna(inplace=True)

df_live["International_INR_KG"] = df_live["Silver_USD"] * 32.1507 * df_live["USDINR"]
df_live["Indian_Physical_INR_KG"] = df_live["International_INR_KG"] * (1 + TOTAL_EXTRA)

daily_price = df_daily["International_INR_KG"].iloc[-1]
live_price = df_live["International_INR_KG"].iloc[-1]
live_indian = df_live["Indian_Physical_INR_KG"].iloc[-1]

usd_last_close = df_daily["USDINR"].iloc[-1]
usd_live = df_live["USDINR"].iloc[-1]
usd_diff = usd_live - usd_last_close

# -------------------------------------------------
# LIVE METRICS
# -------------------------------------------------
st.markdown("<div class='section-title'>🌍 Live Market Rates</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("International (Daily Close INR/KG)", f"{daily_price:.2f}")
col2.metric("US Live Converted (INR/KG)", f"{live_price:.2f}")
col3.metric("Indian Physical 999 (Live INR/KG)", f"{live_indian:.2f}")

st.divider()

col4, col5 = st.columns(2)
col4.metric("USDINR Yesterday Close", f"{usd_last_close:.2f}")
col5.metric("USDINR Live", f"{usd_live:.2f}", f"{usd_diff:.2f}")

if now.weekday() < 5:
    st.success("● LIVE MARKET DATA")
else:
    st.warning("Weekend – Futures Market Closed")

# -------------------------------------------------
# MONTE CARLO
# -------------------------------------------------
st.markdown("<div class='section-title'>🎯 Monte Carlo Risk Model (Indian Physical 999)</div>", unsafe_allow_html=True)

returns = df_daily["Indian_Physical_INR_KG"].pct_change().dropna()
mu = returns.mean()
sigma = returns.std()
S0 = df_daily["Indian_Physical_INR_KG"].iloc[-1]

simulations = 1000
days = 10

paths = np.zeros((days, simulations))

for i in range(simulations):
    price = S0
    for d in range(days):
        shock = np.random.normal(mu, sigma)
        price = price * (1 + shock)
        paths[d, i] = price

mc_final = paths[-1]
expected_price = np.mean(mc_final)
prob_up = np.mean(mc_final > S0) * 100
prob_down = 100 - prob_up

colA, colB, colC = st.columns(3)
colA.metric("Expected Price (10 Days)", f"{expected_price:.2f}")
colB.metric("Probability Up (%)", f"{prob_up:.1f}")
colC.metric("Probability Down (%)", f"{prob_down:.1f}")

# -------------------------------------------------
# PROPHET FORECAST
# -------------------------------------------------
@st.cache_data(ttl=3600)
def generate_forecast(df):
    df_prophet = df.reset_index()[["Date", "Indian_Physical_INR_KG"]]
    df_prophet.columns = ["ds", "y"]

    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=10)
    forecast = model.predict(future)

    result = forecast.tail(10)[["ds", "yhat"]]
    result["Forecast"] = result["yhat"].round(2)
    result["Date"] = result["ds"].dt.strftime("%d %b %Y")

    result.reset_index(drop=True, inplace=True)
    result.index += 1

    return result

forecast_next10 = generate_forecast(df_daily)

forecast_next10["Change"] = forecast_next10["Forecast"].diff()

def signal(x):
    if x > 1000:
        return "BUY"
    elif x < -1000:
        return "SELL"
    else:
        return "HOLD"

forecast_next10["Signal"] = forecast_next10["Change"].apply(signal)

forecast_next10["Confidence (%)"] = (
    abs(forecast_next10["Change"]) / (sigma * S0) * 100
).round(1)

best_index = forecast_next10["Forecast"].idxmin()
best_day = forecast_next10.loc[best_index, "Date"]
best_price = forecast_next10.loc[best_index, "Forecast"]

st.markdown(f"<div class='sub-section'>🏆 Best Day To Buy: <b>{best_day}</b> at ₹{best_price:.2f}</div>", unsafe_allow_html=True)

# -------------------------------------------------
# CHART
# -------------------------------------------------
st.markdown("<div class='section-title'>📈 Indian Physical Silver Trend & 10-Day Forecast</div>", unsafe_allow_html=True)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_daily.tail(10).index,
    y=df_daily.tail(10)["Indian_Physical_INR_KG"],
    name="Last 10 Days",
    line=dict(color="purple", width=3)
))

fig.add_trace(go.Scatter(
    x=forecast_next10["ds"],
    y=forecast_next10["Forecast"],
    name="Forecast",
    line=dict(color="orange", width=3, dash="dash")
))

fig.update_layout(
    template="plotly_dark",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# TABLE
# -------------------------------------------------
st.markdown("<div class='table-title'>📊 10-Day Forecast Summary</div>", unsafe_allow_html=True)

def highlight_signal(val):
    if val == "BUY":
        return "background-color:#14532d;color:white;font-weight:600;"
    elif val == "SELL":
        return "background-color:#7f1d1d;color:white;font-weight:600;"
    else:
        return "background-color:#78350f;color:white;font-weight:600;"

styled_table = forecast_next10[
    ["Date", "Forecast", "Signal", "Confidence (%)"]
].style.format({
    "Forecast": "{:.2f}",
    "Confidence (%)": "{:.1f}"
}).applymap(highlight_signal, subset=["Signal"])

st.write(styled_table)