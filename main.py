import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, LSTM # type: ignore

st.title('Stock Price Prediction')

# Sidebar inputs
st.sidebar.header('User Input')
stock_symbol = st.sidebar.text_input("Stock Ticker Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", dt.date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", dt.date(2024, 6, 28))
prediction_time = st.sidebar.slider("Prediction Time (days)", 30, 120, 60)

# Load stock data
@st.cache
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

data = load_data(stock_symbol, start_date, end_date)

# Prepare data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

x_train = []
y_train = []

for x in range(prediction_time, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_time:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# Load test data
test_data = load_data(stock_symbol, start_date, dt.datetime.now())
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_time:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

x_test = []

for x in range(prediction_time, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_time:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the predictions
st.subheader(f"{stock_symbol} Prices")
fig, ax = plt.subplots()
ax.plot(actual_prices, color="black", label=f"Actual {stock_symbol} price")
ax.plot(predicted_prices, color="green", label=f"Predicted {stock_symbol} price")
ax.set_xlabel("Time")
ax.set_ylabel(f"{stock_symbol} prices")
ax.legend()
st.pyplot(fig)

# Predict Next Day
real_data = [model_inputs[len(model_inputs) + 1 - prediction_time:len(model_inputs + 1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)

st.subheader(f"Prediction for next day: {prediction[0][0]}")
