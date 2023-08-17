import numpy as np

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_dataset(data: np.array, time_steps=30):
    X, y = [], []
    for index in range(len(data)-time_steps-1):
        X.append(data[index: index+time_steps, 0])
        y.append(data[index+time_steps, 0])
    
    return np.array(X), np.array(y)


def data_reshaping(data: np.array):
    """
    LSTM expects the data to be 3-dimensional - (samples, time_steps, features)
    """
    return data.reshape(data.shape[0], data.shape[1], 1)


def lstm_model(data: np.array):
    """
    LSTM DL Model
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(data.shape[1], 1)),
        LSTM(128, return_sequences=True),
        LSTM(128, recurrent_dropout=0.25),
        Dense(64),
        Dropout(0.4),
        Dense(1)
    ])
    return model


def model_run(model, X, y, X_test, y_test, stocks, stock_id, epochs=150, bs=32):
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
    checkpoint = keras.callbacks.ModelCheckpoint(f'{stocks[stock_id]}_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = keras.losses.MeanSquaredError()
    model.compile(loss=loss_fn, optimizer=optimizer)

    history = model.fit(X, y, validation_data=(X_test, y_test), epochs=epochs, 
                        batch_size=bs, verbose=1, callbacks=[early_stop, checkpoint])
    return history


def get_30_days_predictions(time_steps: int, scaled_data: np.array, model, future_days: int =30):
    x_input = scaled_data[-time_steps:].reshape(1,-1).tolist()
    x_input = x_input[0]
    pred_output = []
    for day in range(future_days):
        if day == 0:
            inp = np.array(x_input)
            inp = inp.reshape(1, time_steps, 1)
            y_pred = model.predict(inp, verbose=0)
            x_input.extend(y_pred[0].tolist())
        else:
            inp = np.array(x_input[1:])
            inp = inp.reshape(1, time_steps, 1)
            y_pred = model.predict(inp, verbose=0)
            x_input.extend(y_pred[0].tolist())
            x_input = x_input[1:]
        pred_output.extend(y_pred.tolist())

    return pred_output


def calc_rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(np.mean((y_true-y_pred)**2))

def calc_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true-y_pred)/y_true) * 100
