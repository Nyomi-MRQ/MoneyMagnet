import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from webull import webull
import yfinance as yf

# Set up the Webull client
wb = webull()

# Streamlit page configuration
st.set_page_config(
    page_title="Money Dashboard",
    page_icon="💰",
    layout="wide"
)

# Load the dashboard image
image_path = "/Users/nyomi/Documents/MoneyMagnet/currency-dashboard-image.png"
try:
    image = Image.open(image_path)
except Exception as e:
    st.error(f"Error loading image: {e}")
    image = None

def get_input():
    start_date = st.sidebar.text_input("Start Date", "2020-08-08")
    end_date = st.sidebar.text_input("End Date", datetime.today().strftime('%Y-%m-%d'))
    asset_class = st.sidebar.selectbox("Select Asset Class", ["Stocks", "Crypto"])
    show_tables = st.sidebar.checkbox("Show Data Tables and Charts", value=True)
    return start_date, end_date, asset_class, show_tables

def validate_date(date_text):
    try:
        return pd.to_datetime(date_text)
    except ValueError:
        st.error(f"Invalid date format: {date_text}. Please use YYYY-MM-DD format.")
        return None

def get_tickers(asset_class):
    if asset_class == "Stocks":
        return ['AAPL', 'AMZN', 'GOOGL', 'META']
    elif asset_class == "Crypto":
        return ['BTC-USD', 'ETH-USD']  # Use Yahoo Finance symbols for cryptocurrencies
    else:
        return []

def fetch_data_from_yfinance(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    if data.empty:
        st.error(f"No data found for {symbol}.")
    return data

def display_data(data, asset_class, show_tables):
    if show_tables and image:
        st.image(image, use_column_width=True)

    for ticker, df in data.items():
        df['Date'] = df.index

        if show_tables:
            st.header(f"{ticker} Data")
            st.dataframe(df.style.set_table_styles([
                {'selector': 'thead th', 'props': [('background-color', '#333333'), ('color', 'white'), ('font-size', '16px')]},
                {'selector': 'tbody tr', 'props': [('background-color', '#f5f5f5'), ('color', '#333333'), ('font-size', '14px')]}
            ]))

            st.header(f"{ticker} Data Statistics")
            st.dataframe(df.describe().style.set_table_styles([
                {'selector': 'thead th', 'props': [('background-color', '#333333'), ('color', 'white'), ('font-size', '16px')]},
                {'selector': 'tbody tr', 'props': [('background-color', '#f5f5f5'), ('color', '#333333'), ('font-size', '14px')]}
            ]))

            st.header(f"{ticker} Close Price")
            fig_close = plt.figure(figsize=(16, 8))
            plt.plot(df['Date'], df['Close'], label='Close Price')
            plt.title(f'{ticker} Close Price')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(fig_close)

            if asset_class in ["Stocks", "Crypto"] and 'Volume' in df.columns:
                st.header(f"{ticker} Volume")
                fig_volume = plt.figure(figsize=(16, 8))
                plt.bar(df['Date'], df['Volume'], color='blue', label='Volume')
                plt.title(f'{ticker} Volume')
                plt.xlabel('Date')
                plt.ylabel('Volume')
                plt.legend()
                st.pyplot(fig_volume)

            st.header(f"{ticker} Candlestick Chart")
            fig_candlestick = go.Figure(
                data=[go.Candlestick(
                    x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    increasing_line_color='green',
                    decreasing_line_color='red'
                )]
            )
            fig_candlestick.update_layout(width=1200, height=800)
            st.plotly_chart(fig_candlestick)

        st.header(f"{ticker} Bollinger Bands, RSI, & Fibonacci Strategy")
        df = bollinger_bands(df)
        df = RSI(df)
        df = calculate_fibonacci(df)
        df = strategy(df)
        plot_strategy(df, ticker)

        # Enable Auto Trading feature
        if st.checkbox(f"Enable Auto Trading for {ticker}"):
            execute_trades(df, ticker, wb)

def bollinger_bands(data, window_size=20):
    rolling_mean = data['Close'].rolling(window=window_size).mean()
    rolling_std = data['Close'].rolling(window=window_size).std()
    data['UpperBand'] = rolling_mean + (2 * rolling_std)
    data['LowerBand'] = rolling_mean - (2 * rolling_std)
    return data

def RSI(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    RS = avg_gain / avg_loss
    RSI = 100 - (100 / (1 + RS))
    data['RSI'] = RSI
    return data

def calculate_fibonacci(data):
    max_price = data['Close'].max()
    min_price = data['Close'].min()
    diff = max_price - min_price
    
    data['Fib_23.6'] = max_price - 0.236 * diff
    data['Fib_38.2'] = max_price - 0.382 * diff
    data['Fib_50.0'] = max_price - 0.500 * diff
    data['Fib_61.8'] = max_price - 0.618 * diff
    data['Fib_78.6'] = max_price - 0.786 * diff

    return data

def strategy(data):
    position = 0
    buy_price = []
    sell_price = []
    for i in range(len(data)):
        if data['Close'][i] < data['LowerBand'][i] and data['RSI'][i] < 30 and position == 0:
            position = 1
            buy_price.append(data['Close'][i])
            sell_price.append(np.nan)
        elif data['Close'][i] > data['UpperBand'][i] and data['RSI'][i] > 70 and position == 1:
            position = 0
            sell_price.append(data['Close'][i])
            buy_price.append(np.nan)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
    
    data['Buy'] = buy_price
    data['Sell'] = sell_price
    return data

def plot_strategy(data, ticker):
    fig_strategy = plt.figure(figsize=(16, 8))
    plt.plot(data['Date'], data['Close'], label='Close Price', alpha=0.5)
    plt.plot(data['Date'], data['UpperBand'], label='Upper Band', alpha=0.5)
    plt.plot(data['Date'], data['LowerBand'], label='Lower Band', alpha=0.5)
    plt.fill_between(data['Date'], data['UpperBand'], data['LowerBand'], color='grey', alpha=0.3)

    plt.axhline(data['Fib_23.6'].iloc[-1], color='gray', linestyle='--', label='Fib 23.6%')
    plt.axhline(data['Fib_38.2'].iloc[-1], color='gray', linestyle='--', label='Fib 38.2%')
    plt.axhline(data['Fib_50.0'].iloc[-1], color='gray', linestyle='--', label='Fib 50%')
    plt.axhline(data['Fib_61.8'].iloc[-1], color='gray', linestyle='--', label='Fib 61.8%')
    plt.axhline(data['Fib_78.6'].iloc[-1], color='gray', linestyle='--', label='Fib 78.6%')

    plt.scatter(data.index, data['Buy'], label='Buy', marker='^', color='green', alpha=1)
    plt.scatter(data.index, data['Sell'], label='Sell', marker='v', color='red', alpha=1)
    plt.legend()
    plt.title(f'{ticker} Bollinger Band, RSI & Fibonacci Strategy')
    plt.xlabel('Date')
    plt.ylabel('Price')
    st.pyplot(fig_strategy)

def execute_trades(data, ticker, wb):
    for i in range(len(data)):
        if pd.notna(data['Buy'][i]):
            # Execute a buy order on Webull
            order_id = wb.place_order(
                stock=ticker, 
                action='BUY', 
                orderType='LMT', 
                price=data['Buy'][i],  # Limit price for the order
                enforce='gtc',  # Use 'enforce' instead of 'timeInForce'
                quant=1  # Quantity of shares
            )
            st.write(f"Executed BUY order for {ticker} at {data['Buy'][i]} (Order ID: {order_id})")
        elif pd.notna(data['Sell'][i]):
            # Execute a sell order on Webull
            order_id = wb.place_order(
                stock=ticker, 
                action='SELL', 
                orderType='LMT', 
                price=data['Sell'][i],  # Limit price for the order
                enforce='gtc',  # Use 'enforce' instead of 'timeInForce'
                quant=1  # Quantity of shares
            )
            st.write(f"Executed SELL order for {ticker} at {data['Sell'][i]} (Order ID: {order_id})")

# Main Execution
start_date, end_date, asset_class, show_tables = get_input()
start_date = validate_date(start_date)
end_date = validate_date(end_date)

if start_date and end_date:
    tickers = get_tickers(asset_class)
    if tickers:
        data = {}
        for ticker in tickers:
            df = fetch_data_from_yfinance(ticker, start_date, end_date)
            if not df.empty:
                data[ticker] = df
        display_data(data, asset_class, show_tables)
    else:
        st.write("No tickers available for the selected asset class.")
else:
    st.write("Please enter valid start and end dates.")
