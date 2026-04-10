import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
import datetime

st.title("Trading Strategy Tester Dashboard")

stock = st.selectbox(
    "Stock Select Karein",
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

# ✅ Multi-index fix
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()
df.columns = [str(col).strip() for col in df.columns]

# ✅ Close column ko explicitly float mein convert karo
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna(subset=['Close'])  # sirf Close ke NaN drop karo

st.success("Data loaded!")
st.write("Columns:", df.columns.tolist())
st.write(df.head())

# ==============================
# 1. Stock Price Chart
# ==============================
st.subheader("📈 Stock Price Chart")

fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(df['Date'], df['Close'].values, color='blue')
ax1.set_title(f"{stock} - Close Price")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price (USD)")
st.pyplot(fig1)
plt.close(fig1)

# ==============================
# 2. Moving Average Strategy
# ==============================
df['MA20'] = df['Close'].rolling(20).mean()
df['MA50'] = df['Close'].rolling(50).mean()

st.subheader("📊 Moving Average Strategy")

fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(df['Date'], df['Close'].values, label="Close Price", alpha=0.7)
ax2.plot(df['Date'], df['MA20'].values, label="MA20", color='orange')
ax2.plot(df['Date'], df['MA50'].values, label="MA50", color='green')
ax2.legend()
ax2.set_title(f"{stock} - Moving Averages")
st.pyplot(fig2)
plt.close(fig2)

# ==============================
# 3. Daily Returns
# ==============================
df['Daily Return'] = df['Close'].pct_change()
df = df.dropna()  # ab sab columns ban chuki hain, ab dropna

st.subheader("📉 Daily Returns")

fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.plot(df['Date'], df['Daily Return'].values, color='purple')
ax3.set_title(f"{stock} - Daily Return")
ax3.set_xlabel("Date")
ax3.set_ylabel("Return")
st.pyplot(fig3)
plt.close(fig3)

# ==============================
# 4. Market Volatility
# ==============================
df['Volatility'] = df['Daily Return'].rolling(20).std()

st.subheader("🌊 Market Volatility")

fig4, ax4 = plt.subplots(figsize=(12, 4))
ax4.plot(df['Date'], df['Volatility'].values, color='red')
ax4.set_title(f"{stock} - Rolling Volatility (20-day)")
ax4.set_xlabel("Date")
ax4.set_ylabel("Std Dev")
st.pyplot(fig4)
plt.close(fig4)

# ==============================
# 5. ARIMA Forecast
# ==============================
st.subheader("🔮 Future Price Prediction (ARIMA)")

try:
    close_values = df['Close'].values.flatten().astype(float)
    clean_series = pd.Series(close_values).dropna()

    model = ARIMA(clean_series, order=(2, 1, 2))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=30)
    future_dates = pd.date_range(df['Date'].iloc[-1], periods=31, freq='B')[1:]

    st.write("Next 30 Days Prediction:")
    st.write(forecast)

    fig5, ax5 = plt.subplots(figsize=(12, 4))
    ax5.plot(df['Date'], df['Close'].values, label="Historical Price")
    ax5.plot(future_dates, forecast.values, label="Forecast", color='red', linestyle='--')
    ax5.legend()
    ax5.set_title(f"{stock} - ARIMA Forecast")
    st.pyplot(fig5)
    plt.close(fig5)

except Exception as e:
    st.error(f"Prediction Error: {e}")
    