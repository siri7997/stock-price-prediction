# =======================================================
# LSTM Stock Price Prediction - Apple (AAPL)
# =======================================================

# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# -------------------------------------------------------
# 2. Load Stock Data
# -------------------------------------------------------
df = yf.download('AAPL', start='2015-01-01', end='2024-12-31')
print(df.head())

# Use only the 'Close' price
df1 = df[['Close']]

# -------------------------------------------------------
# 3. Preprocessing
# -------------------------------------------------------
scaler = MinMaxScaler(feature_range=(0,1))
df1_scaled = scaler.fit_transform(np.array(df1).reshape(-1,1))

# Train-Test Split (65%-35%)
training_size = int(len(df1_scaled) * 0.65)
test_size = len(df1_scaled) - training_size
train_data = df1_scaled[0:training_size]
test_data = df1_scaled[training_size:]

print("Training size:", training_size)
print("Test size:", test_size)

# -------------------------------------------------------
# 4. Create Dataset Matrix
# -------------------------------------------------------
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# -------------------------------------------------------
# 5. Build LSTM Model
# -------------------------------------------------------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,  # reduce for speed
    batch_size=64,
    verbose=1
)

# -------------------------------------------------------
# 6. Predictions
# -------------------------------------------------------
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1,1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

# -------------------------------------------------------
# 7. Evaluation Metrics
# -------------------------------------------------------
print("\n--- Evaluation Metrics ---")
print("Train RMSE:", math.sqrt(mean_squared_error(y_train_actual, train_predict)))
print("Test RMSE:", math.sqrt(mean_squared_error(y_test_actual, test_predict)))
print("Test MAE:", mean_absolute_error(y_test_actual, test_predict))
print("Test R² Score:", r2_score(y_test_actual, test_predict))

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("Test MAPE:", mean_absolute_percentage_error(y_test_actual, test_predict))

# -------------------------------------------------------
# 8. Plot Predictions vs Actual
# -------------------------------------------------------
look_back = time_step
trainPredictPlot = np.empty_like(df1_scaled)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

testPredictPlot = np.empty_like(df1_scaled)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1_scaled)-1, :] = test_predict

plt.figure(figsize=(12,6))
plt.plot(scaler.inverse_transform(df1_scaled), label="Original Data")
plt.plot(trainPredictPlot, label="Train Predictions")
plt.plot(testPredictPlot, label="Test Predictions")
plt.title("AAPL Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

# -------------------------------------------------------
# 9. Forecast Next 30 Days
# -------------------------------------------------------
x_input = test_data[-time_step:].reshape(1,-1)
temp_input = list(x_input[0])
lst_output = []
n_steps = time_step
i = 0

while(i < 30):
    if(len(temp_input) > n_steps):
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1, n_steps, 1)
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i += 1
    else:
        x_input = np.array(temp_input).reshape(1, n_steps, 1)
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i += 1

# -------------------------------------------------------
# 10. Plot Forecast
# -------------------------------------------------------
day_new = np.arange(1, 101)
day_pred = np.arange(101, 131)

plt.figure(figsize=(12,6))
plt.plot(day_new, scaler.inverse_transform(df1_scaled[-100:]), label='Previous 100 days')
plt.plot(day_pred, scaler.inverse_transform(np.array(lst_output).reshape(-1,1)), label='Predicted Next 30 days')
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.title("30-Day Stock Price Forecast (AAPL)")
plt.legend()
plt.grid()
plt.show()

# Full extended forecast
df3 = np.array(df1_scaled.tolist() + lst_output).reshape(-1,1)
df3 = scaler.inverse_transform(df3)
plt.figure(figsize=(12,6))
plt.plot(df3, label="Extended Forecast")
plt.title("Extended Stock Price Forecast")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.show()
