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

st_autorefresh(interval=60000, key="live_refresh")

st.markdown("""
<style>
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
.main-title { font-size: 32px; font-weight: 700; margin-bottom: 5px; }
.status-badge { font-size: 16px; margin-bottom: 25px; padding: 5px 12px; border-radius: 5px; display: inline-block; background-color: #1a1a1a; border: 1px solid #333; }
.section-title { font-size: 24px; font-weight: 600; margin-top: 25px; }
.stMetric { background-color: #111; padding: 15px; border-radius: 10px; margin-bottom: 0px;}
.high-low-text { font-size: 13px; font-weight: 600; padding-left: 15px; margin-top: -10px; margin-bottom: 20px;}
.hl-green { color: #10b981; }
.hl-red { color: #ef4444; }
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
# CONSTANTS & CALIBRATION
# ----------------------------------
OUNCES_PER_KG = 32.1507
INDIAN_MARKET_PREMIUM = 0.07127 

# ----------------------------------
# LOAD DATA
# ----------------------------------
@st.cache_data(ttl=3600)
def fetch_current_market_data(): 
    tickers_mapping = {
        "SI=F": "Silver_USD_oz",
        "INR=X": "USDINR",
        "GC=F": "Gold",
        "CL=F": "Crude",
        "^BSESN": "Sensex",
        "^TNX": "US_Rate"
    }
    
    tickers_list = list(tickers_mapping.keys())
    data = yf.download(tickers_list, period="2y", interval="1d")
    
    df_close = data["Close"].copy().rename(columns=tickers_mapping).ffill().dropna()
    df_high = data["High"].copy().rename(columns=tickers_mapping).ffill().dropna()
    df_low = data["Low"].copy().rename(columns=tickers_mapping).ffill().dropna()

    df_close["International_INR_KG"] = df_close["Silver_USD_oz"] * OUNCES_PER_KG * df_close["USDINR"]
    df_close["US_Market_INR_KG"] = df_close["International_INR_KG"] 
    df_close["Indian_Physical_INR_KG"] = df_close["International_INR_KG"] * (1 + INDIAN_MARKET_PREMIUM)
    
    latest_usdinr = float(df_close["USDINR"].iloc[-1])
    prev_usdinr = float(df_close["USDINR"].iloc[-2])
    usdinr_change = latest_usdinr - prev_usdinr

    highs = {
        "US": float(df_high["Silver_USD_oz"].iloc[-1] * OUNCES_PER_KG * latest_usdinr),
        "Ind": float((df_high["Silver_USD_oz"].iloc[-1] * OUNCES_PER_KG * latest_usdinr) * (1 + INDIAN_MARKET_PREMIUM))
    }
    lows = {
        "US": float(df_low["Silver_USD_oz"].iloc[-1] * OUNCES_PER_KG * latest_usdinr),
        "Ind": float((df_low["Silver_USD_oz"].iloc[-1] * OUNCES_PER_KG * latest_usdinr) * (1 + INDIAN_MARKET_PREMIUM))
    }

    return df_close, highs, lows, latest_usdinr, usdinr_change

with st.spinner("Fetching Live Market Data..."):
    df_daily, daily_highs, daily_lows, live_usd, usd_delta = fetch_current_market_data()

latest_data = df_daily.iloc[-1]

actual_current_price = float(latest_data["Indian_Physical_INR_KG"])
us_price = float(latest_data['US_Market_INR_KG'])

# ----------------------------------
# OFF-HOURS MONTE CARLO ENGINE
# ----------------------------------
def calculate_monte_carlo_live_price(df, base_price):
    recent_30d = df.tail(30)
    pct_silver = recent_30d["Indian_Physical_INR_KG"].pct_change().dropna()
    pct_gold = recent_30d["Gold"].pct_change().dropna()
    pct_crude = recent_30d["Crude"].pct_change().dropna()
    
    np.random.seed(datetime.now().minute + datetime.now().hour) 
    shock_gold = np.random.normal(pct_gold.mean(), pct_gold.std() / 10)
    shock_crude = np.random.normal(pct_crude.mean(), pct_crude.std() / 10)
    
    corr_gold = pct_silver.corr(pct_gold)
    corr_crude = pct_silver.corr(pct_crude)
    if np.isnan(corr_gold): corr_gold = 0.5
    if np.isnan(corr_crude): corr_crude = 0.2

    synthetic_return = (shock_gold * corr_gold) + (shock_crude * corr_crude)
    shock_silver_base = np.random.normal(pct_silver.mean(), pct_silver.std() / 10) 
    final_synthetic_return = synthetic_return + shock_silver_base
    
    return float(base_price * (1 + final_synthetic_return))

if market_state == "OFF_HOURS":
    simulated_ticker_price = calculate_monte_carlo_live_price(df_daily, actual_current_price)
else:
    simulated_ticker_price = actual_current_price

simulated_ticker_delta = simulated_ticker_price - actual_current_price

# ----------------------------------
# PURE MULTI-STAGE AI ENGINE
# ----------------------------------
@st.cache_data(ttl=3600)
def run_prophet_ai_forecast(df): 
    df_prophet = df.reset_index()[["Date", "Indian_Physical_INR_KG", "Gold", "Crude", "Sensex", "US_Rate"]]
    df_prophet.rename(columns={"Date": "ds", "Indian_Physical_INR_KG": "y"}, inplace=True)

    future_dates = pd.DataFrame({'ds': pd.date_range(start=df_prophet['ds'].max() + pd.Timedelta(days=1), periods=20)})
    macro_predictions = {}
    
    for col in ['Gold', 'Crude', 'Sensex', 'US_Rate']:
        macro_model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
        macro_df = df_prophet[['ds', col]].rename(columns={col: 'y'})
        macro_model.fit(macro_df)
        macro_pred = macro_model.predict(future_dates)
        macro_predictions[col] = macro_pred['yhat'].values

    for col in ['Gold', 'Crude', 'Sensex', 'US_Rate']:
        future_dates[col] = macro_predictions[col]

    historical_regressors = df_prophet[['ds', 'Gold', 'Crude', 'Sensex', 'US_Rate']]
    all_regressors = pd.concat([historical_regressors, future_dates[['ds', 'Gold', 'Crude', 'Sensex', 'US_Rate']]])

    main_model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
    main_model.add_regressor('Gold')
    main_model.add_regressor('Crude')
    main_model.add_regressor('Sensex')
    main_model.add_regressor('US_Rate')
    main_model.fit(df_prophet)

    forecast = main_model.predict(all_regressors)
    
    last_historical_date = df_prophet['ds'].max()
    future_only = forecast[forecast['ds'] > last_historical_date]
    future_trading_days = future_only[future_only['ds'].dt.weekday < 5].head(10)

    result = future_trading_days[["ds", "yhat"]].copy()
    result["Forecast"] = result["yhat"].round(2)
    result["Date"] = result["ds"].dt.strftime("%d %b %Y")

    return result

with st.spinner("Training Organic Multi-Stage AI Models..."):
    forecast_next10 = run_prophet_ai_forecast(df_daily)

# ----------------------------------
# METRICS DISPLAY (4 COLUMNS)
# ----------------------------------
def hl_html_inr(high_val, low_val):
    return f"<div class='high-low-text'><span class='hl-green'>🔼 High: ₹{high_val:,.2f}</span> &nbsp;|&nbsp; <span class='hl-red'>🔽 Low: ₹{low_val:,.2f}</span></div>"

c1, c2, c3, c4 = st.columns(4)

# 1. Indian Market
c1.metric("Indian Market Rate", f"₹{actual_current_price:,.2f}")
c1.markdown(hl_html_inr(daily_highs['Ind'], daily_lows['Ind']), unsafe_allow_html=True)

# 2. US Market (Base INR/KG)
c2.metric("US Market Rate", f"₹{us_price:,.2f}")
c2.markdown(hl_html_inr(daily_highs['US'], daily_lows['US']), unsafe_allow_html=True)

# 3. Live USD/INR Rate
c3.metric("Live USD/INR Rate", f"₹{live_usd:,.2f}", f"{usd_delta:,.2f}")

# 4. AI Simulated Ticker
c4.metric("AI Simulated Ticker", f"₹{simulated_ticker_price:,.2f}", f"{simulated_ticker_delta:,.2f} (vs Actual)")

st.caption("💡 **Indian Market Rate**: Real-world price (INR/KG). | **US Market**: Converted Base (INR/KG). | **AI Simulated Ticker**: Live weekend prediction reacting to global macro volatility.")

# ----------------------------------
# SWING-TRADE SIGNAL LOGIC
# ----------------------------------
forecast_next10.reset_index(drop=True, inplace=True)

min_forecast = forecast_next10["Forecast"].min()
max_forecast = forecast_next10["Forecast"].max()
forecast_range = max_forecast - min_forecast if max_forecast != min_forecast else 1

def determine_signal(predicted_price):
    normalized_position = (predicted_price - min_forecast) / forecast_range
    if normalized_position <= 0.2: return "STRONG BUY"     
    elif normalized_position <= 0.4: return "BUY"          
    elif normalized_position >= 0.8: return "STRONG SELL"  
    elif normalized_position >= 0.6: return "SELL"         
    else: return "HOLD"                                    

def calculate_confidence(predicted_price):
    normalized_position = (predicted_price - min_forecast) / forecast_range
    conf = 40 + (abs(normalized_position - 0.5) * 2 * 60)
    return round(conf, 1)

forecast_next10["Signal"] = forecast_next10["Forecast"].apply(determine_signal)
forecast_next10["Confidence (%)"] = forecast_next10["Forecast"].apply(calculate_confidence)

forecast_next10.insert(0, "No.", range(1, 11))

# ----------------------------------
# CHART & TABLE
# ----------------------------------
st.markdown("<div class='section-title'>Actuals vs Organic 10-Day AI Forecast</div>", unsafe_allow_html=True)

fig = go.Figure()
hist_x = df_daily.index[-10:].tolist()
hist_y = df_daily["Indian_Physical_INR_KG"].values[-10:].tolist()
pred_x = forecast_next10["ds"].tolist()
pred_y = forecast_next10["Forecast"].tolist()

fig.add_trace(go.Scatter(x=hist_x, y=hist_y, name="Actual History", line=dict(color="#3b82f6", width=2), mode='lines+markers'))
fig.add_trace(go.Scatter(x=[hist_x[-1], pred_x], y=[hist_y[-1], pred_y], showlegend=False, line=dict(color="#10b981", dash="dash", width=3)))
fig.add_trace(go.Scatter(x=pred_x, y=pred_y, name="AI Prediction", line=dict(color="#10b981", dash="dash", width=3), mode='lines+markers'))

fig.update_layout(template="plotly_dark", height=450, legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5), margin=dict(t=40, b=100))
st.plotly_chart(fig, use_container_width=True)

st.markdown("<div class='section-title'>AI Prediction</div>", unsafe_allow_html=True)

def highlight_signal(val):
    if "BUY" in val: return "background-color:#166534;color:white;font-weight:600;"
    elif "SELL" in val: return "background-color:#991b1b;color:white;font-weight:600;"
    else: return "background-color:#854d0e;color:white;font-weight:600;"

display_columns = ["Date", "Forecast", "Signal", "Confidence (%)"]
display_df = forecast_next10.set_index("No.")[display_columns]

styled_table = display_df.style.format({"Forecast": "₹{:,.2f}", "Confidence (%)": "{:.1f}%"}).map(highlight_signal, subset=["Signal"])
st.dataframe(styled_table, use_container_width=True)