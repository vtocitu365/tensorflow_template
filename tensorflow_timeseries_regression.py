#!/usr/bin/env python
# coding: utf-8

# # Timeseries

# In[33]:

import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

# Timeseries LSTM Autoencoder
def vtoc_lstm_autoencoder(n_steps, n_horizon, n_features, lr):
    serie_size=n_steps
    encoder_decoder = Sequential()
    encoder_decoder.add(LSTM(n_steps, activation='relu', input_shape=(n_steps, n_features), return_sequences=True))
    encoder_decoder.add(LSTM(6, activation='relu', return_sequences=True))
    encoder_decoder.add(LSTM(1, activation='relu'))
    encoder_decoder.add(RepeatVector(serie_size))
    encoder_decoder.add(LSTM(serie_size, activation='relu', return_sequences=True))
    encoder_decoder.add(LSTM(6, activation='relu', return_sequences=True))
    encoder_decoder.add(TimeDistributed(Dense(1)))
    encoder_decoder.summary()

    adam = Adam(learning_rate=lr)
    encoder_decoder.compile(loss='mse', optimizer=adam)
    return encoder_decoder


# In[34]:


from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *

#Timeseries LSTM Model
def vtoc_lstm_model(n_steps, n_horizon, n_features, lr):

    tf.keras.backend.clear_session()

    model = Sequential()
    model.add(LSTM(72, activation='relu', input_shape=(n_steps, n_features), return_sequences=True)),
    model.add(LSTM(48, activation='relu', return_sequences=True)),
    model.add(Flatten()),
    model.add(Dropout(0.3)),
    model.add(Dense(128, activation='relu')),
    model.add(Dropout(0.3)),
    model.add(Dense(n_horizon))

    loss = Huber()
    optimizer = Adam(learning_rate=lr)

    model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
def vtoc_cnn_model(n_steps, n_horizon, n_features, lr=3e-4):

    tf.keras.backend.clear_session()

    model = Sequential()
    #tf.keras.layers.Input(shape=(n_steps, n_features)),
    #tf.keras.layers.Flatten(input_shape=(n_steps, n_features)),
    model.add(Conv1D(64, kernel_size=6, activation='relu', input_shape=(n_steps,n_features))),
    model.add(MaxPooling1D(2)),
    model.add(Conv1D(64, kernel_size=3, activation='relu')),
    model.add(MaxPooling1D(2)),
    model.add(Flatten()),
    #tf.keras.layers.Dropout(0.3),
    model.add(Dense(128)),
    model.add(Dropout(0.3)),
    model.add(Dense(n_horizon))

    loss= Huber()
    optimizer =Adam(learning_rate=lr)

    model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
def vtoc_dnn_model(n_steps, n_horizon, n_features, lr):
    tf.keras.backend.clear_session()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(n_steps, n_features)), #Use with evaluate_timeseries_model
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_horizon)
    ], name='dnn')

    loss=tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(loss=loss, optimizer='adam', metrics=['mae'])

    return model

#Timeseries LSTM-CNN Model
def lstm_cnn_model(n_steps, n_horizon, n_features, lr):

    tf.keras.backend.clear_session()

    model = Sequential()
    model.add(Input(shape=(n_steps,n_features))),
    model.add(Conv1D(64, kernel_size=6, activation='relu')),
    model.add(MaxPooling1D(2)),
    model.add(Conv1D(64, kernel_size=3, activation='relu')),
    model.add(MaxPooling1D(2)),
    model.add(LSTM(72, activation='relu', return_sequences=True)),
    model.add(LSTM(48, activation='relu', return_sequences=False)),
    model.add(Flatten()),
    model.add(Dropout(0.3)),
    model.add(Dense(128)),
    model.add(Dropout(0.3)),
    model.add(Dense(n_horizon))

    loss = Huber()
    optimizer = Adam(learning_rate=lr)

    model.compile(loss=loss, optimizer=optimizer, metrics=['mae'])
def lstm_cnn_skip_model(n_steps, n_horizon, n_features, lr):

    tf.keras.backend.clear_session()


    inputs = tf.keras.layers.Input(shape=(n_steps,n_features), name='main')

    conv1 = tf.keras.layers.Conv1D(64, kernel_size=6, activation='relu')(inputs)
    max_pool_1 = tf.keras.layers.MaxPooling1D(2)(conv1)
    conv2 = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu')(max_pool_1)
    max_pool_2 = tf.keras.layers.MaxPooling1D(2)(conv2)
    lstm_1 = tf.keras.layers.LSTM(72, activation='relu', return_sequences=True)(max_pool_2)
    lstm_2 = tf.keras.layers.LSTM(48, activation='relu', return_sequences=False)(lstm_1)
    flatten = tf.keras.layers.Flatten()(lstm_2)

    skip_flatten = tf.keras.layers.Flatten()(inputs)

    concat = tf.keras.layers.Concatenate(axis=-1)([flatten, skip_flatten])
    drop_1 = tf.keras.layers.Dropout(0.3)(concat)
    dense_1 = tf.keras.layers.Dense(128, activation='relu')(drop_1)
    drop_2 = tf.keras.layers.Dropout(0.3)(dense_1)
    output = tf.keras.layers.Dense(n_horizon)(drop_2)

    model = tf.keras.Model(inputs=inputs, outputs=output, name='lstm_skip')

    loss = tf.keras.losses.Huber()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(loss=loss, optimizer='adam', metrics=['mae'])

    return model


# In[35]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import numpy as np
import datetime

# Prepare timeseries data
def multi_baseline_eror(data, pred_cols):
    df = data[pred_cols]
    #fill nans with linear interpolation because this is how we will fill when using the data in the models.
    df_filled = df.interpolate("linear")
    mm = MinMaxScaler()
    df_scaled = mm.fit_transform(df_filled)
    df_prep = pd.DataFrame(df_scaled, columns=pred_cols)
    y_true = df_prep[pred_cols[0]]
    y_pred_forecast = df_prep[pred_cols[1]]

    ### persistence 1 day
    #shift series by 24 hours
    # realign y_true to have the same length and time samples
    y_preds_persistance_1_day = y_true.shift(24).dropna()
    persistence_1_day_mae = tf.keras.losses.MAE(y_true[y_preds_persistance_1_day.index], y_preds_persistance_1_day).numpy()
    persistence_1_day_mape = tf.keras.losses.MAPE(np.maximum(y_true[y_preds_persistance_1_day.index], 1e-5), np.maximum(y_preds_persistance_1_day, 1e-5)).numpy()


    ### persistence 3 day average
    #shift by 1, 2, 3 days. Realign to have same lengths. Average days and calcualte MAE.

    shift_dfs = list()
    for i in range(1, 4):
        shift_dfs.append(pd.Series(y_true.shift(24 * i), name=f"d{i}"))

    y_persistance_3d = pd.concat(shift_dfs, axis=1).dropna()
    y_persistance_3d["avg"] = (y_persistance_3d["d1"] + y_persistance_3d["d2"] + y_persistance_3d["d3"])/3
    d3_idx = y_persistance_3d.index
    persistence_3day_avg_mae = tf.keras.losses.MAE(y_true[d3_idx], y_persistance_3d['avg']).numpy()
    persistence_3day_avg_mape = tf.keras.losses.MAPE(np.maximum(y_true[d3_idx], 1e-5), np.maximum(y_persistance_3d['avg'], 1e-5)).numpy()


    ref_error = pd.DataFrame({
        "Method": ["TSO Forecast", "Persistence 1 Day", "Persitence 3 Day Avg"],
        "MAE": [tf.keras.losses.MAE(y_true, y_pred_forecast).numpy(),
                persistence_1_day_mae,
                persistence_3day_avg_mae],
        "MAPE":[tf.keras.losses.MAPE(np.maximum(y_true, 1e-5), np.maximum(y_pred_forecast, 1e-5)).numpy(),
                persistence_1_day_mape,
                persistence_3day_avg_mape]},
        index=[i for i in range(3)])
    return ref_error

def split_data(series, train_fraq, test_len=8760):
    #slice the last year of data for testing 1 year has 8760 hours
    test_slice = len(series)-test_len

    test_data = series[test_slice:]
    train_val_data = series[:test_slice]

    #make train and validation from the remaining
    train_size = int(len(train_val_data) * train_fraq)

    train_data = train_val_data[:train_size]
    val_data = train_val_data[train_size:]

    return train_data, val_data, test_data

def create_window_data(dataset, look_back=1, n_features=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :n_features]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :n_features])
    return np.array(dataX), np.array(dataY)

def window_dataset(data, n_steps, n_horizon, batch_size, shuffle_buffer, multi_var=False, expand_dims=False):
    #create a window with n steps back plus the size of the prediction length
    window = n_steps + n_horizon

    #expand dimensions to 3D to fit with LSTM inputs
    #creat the inital tensor dataset
    if expand_dims:
        ds = tf.expand_dims(data, axis=-1)
        ds = tf.data.Dataset.from_tensor_slices(ds)
    else:
        ds = tf.data.Dataset.from_tensor_slices(data)

    #create the window function shifting the data by the prediction length
    ds = ds.window(window, shift=n_horizon, drop_remainder=True)

    #flatten the dataset and batch into the window size
    ds = ds.flat_map(lambda x : x.batch(window))
    ds = ds.shuffle(shuffle_buffer)

    #create the supervised learning problem x and y and batch
    if multi_var:
        ds = ds.map(lambda x : (x[:-n_horizon], x[-n_horizon:, :1]))
    else:
        ds = ds.map(lambda x : (x[:-n_horizon], x[-n_horizon:]))

    ds = ds.batch(batch_size).prefetch(1)

    return ds


def create_multi_dataset(data, pred_cols, multivar):
    date_time = pd.to_datetime(data.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    timestamp_s = date_time.map(datetime.datetime.timestamp)
    for column in pred_cols:
        # Getting rid of outliers
        data.loc[data[column] == -9999.0, column] = 0.0
    ref_error = multi_baseline_eror(data, pred_cols)
    data=MinMaxScaler().fit_transform(data)
    tf.random.set_seed(42)
    train_fraq=0.65
    lr = 3e-4
    n_steps = 14#24*30
    n_horizon = 14
    batch_size = 64#256
    shuffle_buffer = 100
    if multivar:
        n_features=len(data[0])
    else:
        n_features=1
    train_data, val_data, test_data = split_data(data, train_fraq=train_fraq, test_len=8760)

    (X_train, Y_train)=create_window_data(train_data, look_back=n_horizon, n_features=n_features)
    (X_test, Y_test)=create_window_data(test_data, look_back=n_horizon, n_features=n_features)
    (X_val, Y_val)=create_window_data(val_data, look_back=n_horizon, n_features=n_features)
    #train_ds = window_dataset(train_data, n_steps, n_horizon, batch_size, shuffle_buffer, multi_var=multivar)
    #test_ds = window_dataset(test_data, n_steps, n_horizon, batch_size, shuffle_buffer, multi_var=multivar)
    #val_ds = window_dataset(val_data, n_steps, n_horizon, batch_size, shuffle_buffer, multi_var=multivar)
    #split_sequences(train_ds, n_steps, extra_lag=False, long_lag_step=7, max_step=30, idx=0, multivar=False)
    #print(f"Train Data Shape: {train_ds.shape}")
    #print(f"Val Data Shape: {val_ds.shape}")
    #kfold = KFold(n_folds=5, shuffle=True, random_state=1)
    model=vtoc_dnn_model(n_steps, n_horizon, n_features, lr)
    model.summary()
    model_hist = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=5, verbose=3)
    #model.fit(X_train, X_train, epochs=10, verbose=3, shuffle=True)
    model.evaluate(X_train, Y_train)
    validation_encoded = model.predict(X_val)
    #_, acc = model.evaluate(test, verbose=2)
    #scores.append(acc)
    return model_hist.history

def evaluate_multi_timeseries_model(timeseries_df, n_folds=5, multivar=False):
    df_processed = create_multi_dataset(timeseries_df, pred_cols = ['wv (m/s)', 'max. wv (m/s)'], multivar=multivar)
    print(df_processed)
    return


# In[36]:


import os
import tensorflow as tf
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
# On macOS the zip extracts to a folder, so we need to find the CSV inside it
extracted = os.path.splitext(zip_path)[0]
if os.path.isdir(extracted):
    csv_path = os.path.join(extracted, 'jena_climate_2009_2016.csv')
else:
    csv_path = extracted



# In[37]:


import pandas as pd
timeseries_df = pd.read_csv(csv_path)
device = '/device:GPU:0' if tf.config.list_physical_devices('GPU') else '/device:CPU:0'
print(f"Training on: {device}")
with tf.device(device):
    evaluate_multi_timeseries_model(timeseries_df, n_folds=5, multivar=True)


# In[38]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import datetime

def create_dataset(X, y, delay=24, lookback=48):
    window_length = lookback + delay
    Xs, ys = [], []
    for i in range(lookback, len(X)-delay):
        v = X.iloc[i-lookback:i].to_numpy() # every one hour, we take the past 48 hours of features
        Xs.append(v)
        w = y.iloc[i+delay] # Every timestep, we take the temperature the next delay (here one day)
        ys.append(w)
    return(np.array(Xs), np.array(ys))

def preprocessing(data):

    # Getting rid of outliers
    data.loc[data['wv (m/s)'] == -9999.0, 'wv (m/s)'] = 0.0
    data.loc[data['max. wv (m/s)'] == -9999.0, 'max. wv (m/s)'] = 0.0

    # Taking values every hours
    data = data[5::6]# df[start,stop,step]

    wv = data.pop('wv (m/s)')
    max_wv = data.pop('max. wv (m/s)')

    # Convert to radians.
    wd_rad = data.pop('wd (deg)')*np.pi / 180

    # Calculate the wind x and y components.
    data['Wx'] = wv*np.cos(wd_rad)
    data['Wy'] = wv*np.sin(wd_rad)

    # Calculate the max wind x and y components.
    data['max Wx'] = max_wv*np.cos(wd_rad)
    data['max Wy'] = max_wv*np.sin(wd_rad)

    date_time = pd.to_datetime(data.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    timestamp_s = date_time.map(datetime.datetime.timestamp)

    day = 24*60*60 # Time is second within a single day
    year = 365.2425*day # Time in second withon a year

    data['Day sin'] = np.sin(timestamp_s * (2*np.pi / day))
    data['Day cos'] = np.cos(timestamp_s * (2*np.pi / day))
    data['Year sin'] = np.sin(timestamp_s * (2*np.pi / year))
    data['Year cos'] = np.cos(timestamp_s * (2*np.pi / year))

    return(data)

def split(data):

    n = data.shape[0]

    train_df = data.iloc[0: n * 70 //100] # "iloc" because we have to select the lines at the indicies 0 to int(n*0.7) compared to "loc"
    val_df = data.iloc[n * 70 //100 : n * 90 //100]
    test_df = data.iloc[n * 90 //100:]

    return(train_df, val_df, test_df)

def naive_eval_arr(X, y, lookback, delay):
    batch_maes = []
    for i in range(0, len(X)):
        preds = X[i, -1, 1] #For all elements in the batch, we are saying the prediction of temperature is equal to the last temperature recorded within the 48 hours
        mae = np.mean(np.abs(preds - y[i]))
        batch_maes.append(mae)
    return(np.mean(batch_maes))

def split_sequences(sequences, n_steps, extra_lag=False, long_lag_step=7, max_step=30, idx=0, multivar=False):
    #if not adding extra lag features adjust max_step and n_steps to aling
    if not extra_lag:
        max_step=n_steps
        n_steps+=1


    X, y = list(), list()
    for i in range(len(sequences)):

        # find the end of this pattern
        #end_ix = i + n_steps
        end_ix = i + max_step

        #create a list with the indexes we want to include in each sample
        slices = [x for x in range(end_ix-1,end_ix-n_steps, -1)] + [y for y in range(end_ix-n_steps, i, -long_lag_step)]

        #reverse the slice indexes
        slices = list(reversed(slices))

        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break


        # gather input and output parts of the pattern
        seq_x = sequences[slices, :]
        seq_y = sequences[end_ix, :]

        X.append(seq_x)
        y.append(seq_y)

    X = np.array(X)
    y = np.array(y)

    if multivar:
        #unstack the 3rd dimension and select the first element(energy load)
        y = y[:,idx]

    return X, y

def generator(data, lookback, delay, min_index, max_index, shuffle=True, batch_size=32, step=1):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while True:
        if shuffle == True: # pay attention ! We are not shuffeling timesteps but elemnts within a batch ! It is important to keep the data in time order
            rows = np.random.randint(min_index + lookback, max_index-delay-1, size=batch_size) # return an array containing size elements ranging from min_index+lookback to max_index
        else:
            if i + batch_size >= max_index-delay-1: #Since we are incrementing on "i". If its value is greater than the max_index --> start from the begining
                i = min_index + lookback # We need to start from the indice lookback, since we want to take lookback elements here.
            rows = np.arange(i, min(i + batch_size, max_index)) # Just creating an array that contain the indices of each sample in the batch
            i+=len(rows) # rows represents the number of sample in one batch

        samples = np.zeros((len(rows), lookback//step, data.shape[-1])) # shape = (batch_size, lookback, nbr_of_features)
        targets = np.zeros((len(rows),)) #Shape = (batch_size,)

        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step) #From one indice given by rows[j], we are picking loockback previous elements in the dataset
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1] #We only want to predict the temperature for now,since [1], the second column
        yield samples, targets # The yield that replace the return to create a generator and not a regular function.

def evaluate_timeseries_model(df, n_folds=5, multivar=False):
    scores, histories = list(), list()
    df_processed = preprocessing(df)
    train_df, val_df, test_df = split(df_processed)
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean)/train_std # As simple as that !
    val_df = (val_df - train_mean)/train_std
    test_df = (test_df - train_mean)/train_std
    lookback = 48 # Looking at all features for the past 2 days
    delay = 24 # Trying to predict the temperature for the next day
    batch_size = 64 # Features will be batched 32 by 32.
    X_train, Y_train = create_dataset(train_df, train_df['T (degC)'], delay = delay, lookback = lookback)
    X_val, Y_val = create_dataset(val_df, val_df['T (degC)'], delay = delay)
    naive_loss_arr = naive_eval_arr(X_val, Y_val, lookback = lookback, delay = delay)

    naive_loss_arr = round(naive_eval_arr(X_val, Y_val, lookback = lookback, delay = delay),2) # Round the value


    data_train = train_df.to_numpy()
    (X_train, Y_train) = generator(data = data_train, lookback = lookback, delay =delay, min_index = 0,
                                   max_index = len(data_train), shuffle = True, batch_size = batch_size)

    data_val = val_df.to_numpy()
    (X_val, Y_val) = generator(data = data_val, lookback = lookback, delay =delay, min_index = 0,
                               max_index = len(data_val), batch_size = batch_size)

    data_test = test_df.to_numpy()
    test_gen = generator(data=data_test, lookback=lookback, delay=delay, min_index=0,
                         max_index=len(data_test), batch_size=batch_size)
    X_test, Y_test = next(test_gen)
    tscv = TimeSeriesSplit(n_splits=5, test_size=2)

    '''for train_index, test_index in tscv.split(X_train):
        X_train, X_test = X_train[train_index], X_train[test_index]
        Y_train, Y_test = Y_train[train_index], Y_train[test_index]'''
    print(naive_loss_arr)
    lr = 3e-4
    n_steps=48#24*30
    n_horizon=24
    #batch_size=64
    if multivar:
        n_features=5
    else:
        n_features=1
    model=vtoc_cnn_model(n_steps, n_horizon, n_features, lr)

    print(X_train.shape, Y_train.shape)
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, verbose=3, shuffle=True)
    #model.fit(X_train, X_train, epochs=10, verbose=3, shuffle=True)
    train_encoded = model.predict(X_train)
    validation_encoded = model.predict(X_val)
    print('Encoded time-series shape', train_encoded.shape)
    print('Encoded time-series sample', train_encoded[0])
    return model.evaluate(X_test, Y_test)


# In[39]:


import pandas as pd
import numpy as np
#csv_path = "/kaggle/input/energy-consumption-generation-prices-and-weather/energy_dataset.csv"
timeseries_df = pd.read_csv(csv_path)
#evaluate_timeseries_model(timeseries_df, n_folds=5, multivar=False)


# # Regression

# In[31]:


from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_datasets as tfds
# Basic ANN
class EarlyStoppingCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        if logs['val_loss'] < 0.54:
            print("Val loss threshold reached. Stopping.")
            self.model.stop_training = True

def val_dnn_model(epochs, X_train, Y_train, X_val, Y_val, callbacks=None):
    model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')])

    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, callbacks=callbacks)

    return history, model


# In[32]:


train = tfds.load('german_credit_numeric', split=['train'], batch_size=-1, as_supervised=True)
X=tfds.as_numpy(train[0][0])
Y=tfds.as_numpy(train[0][1])
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)
hist_regularized, model_regularized = val_dnn_model(100, X_train, Y_train, X_test, Y_test, callbacks=[EarlyStoppingCallback()])
print(f"Training Set:   {model_regularized.evaluate(X_train, Y_train)}")
print(f"Validation Set: {model_regularized.evaluate(X_test, Y_test)}")


# In[40]:


from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from sklearn.model_selection import train_test_split
def vtoc_regression_model(norm, model_type):
    if model_type=='linear':
        model = Sequential()
        model.add(norm)
        model.add(Dense(1))
    else:
        model = Sequential()
        model.add(norm)
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1))

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss='mean_absolute_error')

    return model


def eval_regression_data(dataset, target):
    dataset = dataset.astype('float32')
    dataset.isna().sum()
    dataset=dataset.dropna()
    #dataset['Origin'] = dataset['Origin'].map({1.0: 'USA', 2.0: 'Europe', 3.0: 'Japan'})
    Y=dataset[target]
    X=dataset.loc[:, dataset.columns != target]
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)
    horsepower = np.array(X_train['Horsepower']).astype('float32')

    horsepower_normalizer = Normalization(axis=None)
    horsepower_normalizer.adapt(horsepower)

    linear_model=vtoc_regression_model(horsepower_normalizer, model_type='linear')
    horsepower_test = np.array(X_test['Horsepower']).astype('float32')
    history = linear_model.fit(horsepower, Y_train, validation_data=(horsepower_test, Y_test), epochs=100, verbose=2)
    test_results = linear_model.predict(horsepower_test)
    return test_results


# In[41]:


url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)
eval_regression_data(raw_dataset, 'MPG')

