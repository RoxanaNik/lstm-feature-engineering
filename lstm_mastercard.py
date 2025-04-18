# lstm_mastercard.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import callbacks

# Load Mastercard stock data
df = pd.read_csv('Mastercard_stock_history.csv')
df.index = pd.to_datetime(df['Date'])
df.drop(columns=['Date'], inplace=True)

# Split into train and test sets
train = df[(df.index.year < 2021) & (df.index.year >= 2016)]
test = df[df.index.year >= 2021]

# Normalize 'Close' column
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(train['Close'].values.reshape(-1, 1))

# Prepare training sequences
X_train, y_train = [], []
for i in range(80, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-80:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae', 'mape'])

# Train the model with early stopping
early_stopping = callbacks.EarlyStopping(patience=6, min_delta=0.001, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=100, batch_size=24, callbacks=[early_stopping])

# Plot training metrics
history_df = pd.DataFrame(history.history)
for metric in ['mse', 'mae', 'mape']:
    plt.plot(history_df[metric], label=metric.upper())
    plt.title(metric.upper())
    plt.legend()
    plt.show()

# Prepare test data
dataset_total = df['Close']
inputs = dataset_total[len(dataset_total) - len(test['Close'].values) - 80:].values.reshape(-1, 1)
inputs_scaled = sc.transform(inputs)

X_test = []
for i in range(80, len(inputs_scaled)):
    X_test.append(inputs_scaled[i-80:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make predictions
predicted_prices = model.predict(X_test)
predicted_prices = sc.inverse_transform(predicted_prices)

# Store predictions
predictions = pd.DataFrame({
    'Actuals': test['Close'].values,
    'Predictions': predicted_prices.flatten()
})

# Plot actual vs predicted prices
plt.figure(figsize=(14, 8))
plt.title('Mastercard Close Stock Price Prediction')
plt.plot(predictions['Actuals'], label='Actual')
plt.plot(predictions['Predictions'], label='Predicted')
plt.legend()
plt.show()



