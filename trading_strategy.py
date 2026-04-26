import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
import datetime

# ==============================
# Title
# ==============================
st.title("📊 Trading Strategy Tester Dashboard")

# ==============================
# User Input
# ==============================
stock = st.selectbox(
    "Select Stock",
    ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
)

start_date = st.date_input("Start Date", datetime.date(2022, 1, 1))
end_date = st.date_input("End Date", datetime.date.today())

# ==============================
# Fetch Data
# ==============================
df = yf.download(stock, start=start_date, end=end_date, auto_adjust=True)

if df.empty:
    st.warning("No data found")
    st.stop()

# Fix multi-index issue
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()

# ==============================
# 1. Stock Price Chart
# ==============================
st.subheader("📈 Stock Price")

fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(df['Date'], df['Close'], color='blue')
ax1.set_title(f"{stock} Close Price")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price (USD)")
st.pyplot(fig1)
plt.close(fig1)

# ==============================
# 2. Moving Average Strategy
# ==============================
df['MA20'] = df['Close'].rolling(20).mean()
df['MA50'] = df['Close'].rolling(50).mean()

st.subheader("📊 Moving Averages")

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(df['Date'], df['Close'], label="Close Price", alpha=0.7)
ax2.plot(df['Date'], df['MA20'], label="MA20", color='orange')
ax2.plot(df['Date'], df['MA50'], label="MA50", color='green')
ax2.legend()
st.pyplot(fig2)
plt.close(fig2)

# ==============================
# 3. Daily Returns
# ==============================
df['Daily Return'] = df['Close'].pct_change()

st.subheader("📉 Daily Returns")

fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.plot(df['Date'], df['Daily Return'], color='purple')
ax3.set_title(f"{stock} Daily Return")
st.pyplot(fig3)
plt.close(fig3)

# ==============================
# 4. Volatility
# ==============================
df['Volatility'] = df['Daily Return'].rolling(20).std()

st.subheader("🌊 Volatility (20-day)")

fig4, ax4 = plt.subplots(figsize=(12, 4))
ax4.plot(df['Date'], df['Volatility'], color='red')
ax4.set_title(f"{stock} Volatility")
st.pyplot(fig4)
plt.close(fig4)

# ==============================
# 5. ARIMA Forecast
# ==============================
st.subheader("🔮 Future Prediction (ARIMA)")

try:
    clean_series = df['Close'].dropna()

    model = ARIMA(clean_series, order=(2, 1, 2))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=30)

    future_dates = pd.date_range(
        start=df['Date'].iloc[-1],
        periods=30,
        freq='B'
    )

    st.write("Next 30 Days Prediction:")
    st.write(forecast)

    fig5, ax5 = plt.subplots(figsize=(12, 4))
    ax5.plot(df['Date'], df['Close'], label="Historical")
    ax5.plot(future_dates, forecast, label="Forecast", linestyle='--', color='red')
    ax5.legend()
    st.pyplot(fig5)
    plt.close(fig5)

except Exception as e:
    st.error(f"Prediction Error: {e}")
