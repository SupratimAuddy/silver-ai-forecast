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
st.set_page_config(layout="wide", page_title="Silver AI Dashboard")

# ----------------------------------
# AUTO REFRESH (60 sec)
# ----------------------------------
st_autorefresh(interval=60000, key="live_refresh")

# ----------------------------------
# CUSTOM CSS (Better Readability)
# ----------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 10px;
}
.section-title {
    font-size: 24px;
    font-weight: 600;
    margin-top: 35px;
}
.stMetric {
    background-color: #111;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>Silver 999 (1KG) AI Forecast Dashboard</div>", unsafe_allow_html=True)

TOTAL_EXTRA = 0.05

# ----------------------------------
# TIME
# ----------------------------------
ist = pytz.timezone("Asia/Kolkata")
now = datetime.now(ist)
st.caption(f"Last Updated: {now.strftime('%d %b %Y | %I:%M:%S %p IST')}")

# ----------------------------------
# LOAD DAILY DATA (Cached)
# ----------------------------------
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

# ----------------------------------
# LIVE DATA
# ----------------------------------
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

# ----------------------------------
# LIVE SECTION
# ----------------------------------
st.markdown("<div class='section-title'>Live Market Rates</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("International (Daily Close INR/KG)", f"{daily_price:.2f}")
col2.metric("US Live Converted (INR/KG)", f"{live_price:.2f}")
col3.metric("Indian Physical 999 (Live INR/KG)", f"{live_indian:.2f}")

st.divider()

col4, col5 = st.columns(2)
col4.metric("Dollar Yesterday Close", f"{usd_last_close:.2f}")
col5.metric("Dollar Live", f"{usd_live:.2f}", f"{usd_diff:.2f}")

# ----------------------------------
# MONTE CARLO (Indian Physical)
# ----------------------------------
st.markdown("<div class='section-title'>Monte Carlo Risk Model</div>", unsafe_allow_html=True)

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
        price *= (1 + shock)
        paths[d, i] = price

mc_final = paths[-1]
expected_price = np.mean(mc_final)
prob_up = np.mean(mc_final > S0) * 100
prob_down = 100 - prob_up

colA, colB, colC = st.columns(3)
colA.metric("Expected Price (10 Days)", f"{expected_price:.2f}")
colB.metric("Probability Up (%)", f"{prob_up:.1f}")
colC.metric("Probability Down (%)", f"{prob_down:.1f}")

# ----------------------------------
# PROPHET FORECAST
# ----------------------------------
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

    return result

forecast_next10 = generate_forecast(df_daily)

# ----------------------------------
# SIGNAL LOGIC
# ----------------------------------
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

forecast_next10.reset_index(drop=True, inplace=True)
forecast_next10.insert(0, "No.", range(1, 11))

# ----------------------------------
# CHART
# ----------------------------------
st.markdown("<div class='section-title'>Indian Physical Trend & Forecast</div>", unsafe_allow_html=True)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_daily.tail(10).index,
    y=df_daily.tail(10)["Indian_Physical_INR_KG"],
    name="Last 10 Days"
))

fig.add_trace(go.Scatter(
    x=forecast_next10["ds"],
    y=forecast_next10["Forecast"],
    name="Forecast",
    line=dict(dash="dash")
))

fig.update_layout(
    template="plotly_dark",
    height=450,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.25,
        xanchor="center",
        x=0.5
    ),
    margin=dict(t=40, b=100)
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------
# TABLE
# ----------------------------------
st.markdown("<div class='section-title'>10-Day Forecast Summary</div>", unsafe_allow_html=True)

def highlight_signal(val):
    if val == "BUY":
        return "background-color:#166534;color:white;font-weight:600;"
    elif val == "SELL":
        return "background-color:#991b1b;color:white;font-weight:600;"
    else:
        return "background-color:#854d0e;color:white;font-weight:600;"

styled_table = forecast_next10[
    ["No.", "Date", "Forecast", "Signal", "Confidence (%)"]
].style.format({
    "Forecast": "{:.2f}",
    "Confidence (%)": "{:.1f}"
}).applymap(highlight_signal, subset=["Signal"])

st.write(styled_table)

st.caption("⚠ Educational tool only. Not financial advice.")