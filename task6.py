import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN

# Load data
data = pd.read_csv('LSTM-Multivariate_pollution.csv')


# Preprocessing
encoder = LabelEncoder()
data['wnd_dir'] = encoder.fit_transform(data['wnd_dir'])  # Encode the 'wnd_dir' feature

# Select relevant features and target
features = ['temp', 'press', 'wnd_dir', 'wnd_spd', 'pollution']  
target = 'pollution'

# Fill missing values and scale the data
data.fillna(method='ffill', inplace=True)
scaler = MinMaxScaler()

# Scale all the features (including pollution)
scaled_data = scaler.fit_transform(data[features])

# Function to create sequences of the time series data
def create_sequences(data, target_column, sequence_length=24):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])  # All columns including pollution
        y.append(data[i + sequence_length, target_column])  # Target column: pollution shifted
    return np.array(X), np.array(y)

# Define sequence length and create sequences
sequence_length = 24  # E.g., 24 hours (1 day)
X, y = create_sequences(scaled_data, target_column=-1, sequence_length=sequence_length)

# Split data into training and testing sets (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Check the shape of the train and test sets
print("Training data shape:", X_train.shape, y_train.shape)
print("Testing data shape:", X_test.shape, y_test.shape)

# Function to build LSTM or RNN model
def build_model(model_type='LSTM', input_shape=(24, 5)):  # Input shape is now (24, 5) with 5 features
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(50, return_sequences=False, input_shape=input_shape))
    elif model_type == 'RNN':
        model.add(SimpleRNN(50, return_sequences=False, input_shape=input_shape))
    model.add(Dense(1))  # Output layer for pollution prediction
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Build LSTM and RNN models
lstm_model = build_model(model_type='LSTM', input_shape=(24, 5))  # 5 features now, including pollution
rnn_model = build_model(model_type='RNN', input_shape=(24, 5))

# Train the LSTM model and capture the history
lstm_history = lstm_model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Train the RNN model and capture the history
rnn_history = rnn_model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the models
lstm_loss = lstm_model.evaluate(X_test, y_test)
rnn_loss = rnn_model.evaluate(X_test, y_test)

print(f"LSTM Loss: {lstm_loss}")
print(f"RNN Loss: {rnn_loss}")

# Function to plot training history (loss curve)
def plot_training_history(history, model_type):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_type} Model Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Plot the loss curves for LSTM and RNN models
plot_training_history(lstm_history, 'LSTM')
plot_training_history(rnn_history, 'RNN')

# Predict the pollution on the test set
lstm_predictions = lstm_model.predict(X_test)
rnn_predictions = rnn_model.predict(X_test)

# Rescale the predicted and actual pollution values back to the original scale
y_test_rescaled = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], y_test.reshape(-1, 1)), axis=1))[:, -1]
lstm_predictions_rescaled = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], lstm_predictions), axis=1))[:, -1]
rnn_predictions_rescaled = scaler.inverse_transform(np.concatenate((X_test[:, -1, :-1], rnn_predictions), axis=1))[:, -1]

# Plot the actual vs predicted values for LSTM
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled, label='Actual Pollution')
plt.plot(lstm_predictions_rescaled, label='LSTM Predicted Pollution')
plt.title('LSTM Actual vs Predicted Pollution')
plt.xlabel('Time')
plt.ylabel('Pollution')
plt.legend()
plt.show()

# Plot the actual vs predicted values for RNN
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled, label='Actual Pollution')
plt.plot(rnn_predictions_rescaled, label='RNN Predicted Pollution')
plt.title('RNN Actual vs Predicted Pollution')
plt.xlabel('Time')
plt.ylabel('Pollution')
plt.legend()
plt.show()
