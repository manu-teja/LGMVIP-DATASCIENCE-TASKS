import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import datetime
import requests

# Load the data from the URL
url = "https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv"
response = requests.get(url)
with open('NSE-TATAGLOBAL.csv', 'wb') as file:
    file.write(response.content)

data = pd.read_csv('NSE-TATAGLOBAL.csv')
data = data[['Date', 'Close']]
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
scaler = MinMaxScaler()
data['Close'] = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

def create_sequences(data, seq_length):
    X, Y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, 0])
        Y.append(data[i+seq_length, 0])
    return np.array(X), np.array(Y)

seq_length = 10
train_X, train_Y = create_sequences(train_data.values, seq_length)
test_X, test_Y = create_sequences(test_data.values, seq_length)

train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

model = Sequential()
model.add(LSTM(50, input_shape=(seq_length, 1)))  # Add the missing closing parenthesis here
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_X, train_Y, epochs=100, batch_size=8, verbose=2)

train_predict = model.predict(train_X)
test_predict = model.predict(test_X)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

train_data['Predictions'] = np.nan
test_data['Predictions'] = np.nan
train_data['Predictions'].iloc[seq_length:seq_length + len(train_predict)] = train_predict.reshape(-1)
test_data['Predictions'].iloc[-len(test_predict):] = test_predict.reshape(-1)

plt.figure(figsize=(16, 8))
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.plot(data['Close'], label='Actual Price')
plt.plot(train_data['Predictions'], label='Training Predictions')
plt.plot(test_data['Predictions'], label='Testing Predictions')
plt.legend()
plt.show()
