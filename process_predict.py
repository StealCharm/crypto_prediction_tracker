from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import History
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import matplotlib.pylab as plt
from tensorflow.keras.layers import Dropout
import os
df = pd.read_csv("BTC-USD.csv")

def read_file():
    path = r"csv_database/"
    # do this later

def normalize(database):
    scaler = MinMaxScaler()
    database["Open"] = scaler.fit_transform(database["Open"].values.reshape(-1, 1))
    return database
def create_sequences(data, seq_length):
    x = []
    y = []

    for i in range(0, len(data) - seq_length):
        x.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])

    return np.array(x), np.array(y)

def LSTM_model(X_train):
    sample_model = Sequential()
    sample_model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    sample_model.add(Dropout(0.2))
    sample_model.add(LSTM(50, return_sequences=True))
    sample_model.add(Dropout(0.2))
    sample_model.add(LSTM(50))
    sample_model.add(Dropout(0.2))
    sample_model.add(Dense(1))

    sample_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# the Model needs to be a class for it to understand
# for the time being, just place it here but it iwill be placed in its
# own file
def unit_test():
    df = pd.read_csv("BTC-USD.csv")

    new_df = normalize(df)

    seq_length = 60
    X, y = create_sequences(new_df[['Open']].values, seq_length)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

    sample_model = Sequential()
    sample_model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    sample_model.add(Dropout(0.2))
    sample_model.add(LSTM(50, return_sequences=True))
    sample_model.add(Dropout(0.2))
    sample_model.add(LSTM(50))
    sample_model.add(Dropout(0.2))
    sample_model.add(Dense(1))

    sample_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    history = sample_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data = (X_test, y_test) ,verbose=2)

    sample_model_loss = history.history['loss']
    sample_model_accuracy = history.history['accuracy']
    sample_model_val_loss = history.history['val_loss']
    sample_model_val_accuracy = history.history['val_accuracy']

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

unit_test()