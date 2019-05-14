import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pandas import Series
from . import plot_utils as plt


class DataFetcher():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, split, cols):
        dataframe = pd.read_csv(filename)
        self.i_split = int(len(dataframe) * split)
        self.cols = cols
        self.data_train = dataframe.get(cols).values[:self.i_split]
        self.data_test  = dataframe.get(cols).values[self.i_split:]
        self.data_total = dataframe.get(cols).values[:]
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_total  = len(self.data_total)
        self.len_train_windows = None

        #normalise the data
        total_data = self.data_total.copy()
        last_ob = total_data[0][0]
        self.scaler, self.data_total_norm = self.transform_data(total_data)
        self.data_train_norm = self.data_total_norm[:self.i_split]
        self.data_test_norm  = self.data_total_norm[self.i_split:]

    def train_data(self, norm):
        if norm: 
            return self.data_train_norm.copy()
        return self.data_train.copy()

    def total_data(self, norm):
        if norm: 
            return self.data_total_norm.copy()
        return self.data_total.copy()

    def get_scaler(self):
        return self.scaler

    # create a differenced series
    def difference(self, dataset, interval=1):
        diff = []
        for i in range(interval, len(dataset)):
            # new value is the difference between the two adjacent points
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return Series(diff)

    # invert differenced forecast
    def inverse_difference(self, last_ob, forecast):
        # invert first forecast
        inverted = []

        first = forecast[0] + last_ob
        inverted.append(first)
        # propagate difference forecast using inverted first value
        for i in range(1, len(forecast)):
            inverted.append(forecast[i] + inverted[i-1])
        return inverted

    def reshape_scaled_data(self, last_ob, forecast):
        inverted = []

        first = forecast[0] 
        inverted.append(first)

        for i in range(1, len(forecast)):
            inverted.append(forecast[i])
        return inverted

    def transform_data(self, raw_values):
        """difference the data then scale it to values between -1, 1"""

        # transform data to be stationary: values are now the differences between adjacent points
        # diff_series = self.difference(raw_values, 1)
        # diff_values = diff_series.values
        # diff_values = diff_values.reshape(len(diff_values), 1)
        
        # rescale values to be between -1, 1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_values = scaler.fit_transform(raw_values)
        scaled_values = scaled_values.reshape(len(scaled_values), 1)
        return scaler, scaled_values

    # inverse data transform on forecasts
    def inverse_transform(self, last_ob, dataset):
        
        dataset = dataset.copy().transpose()

        #inverse scaling
        inv_scale = self.scaler.inverse_transform(dataset)

        inv_scale = inv_scale[0, :]
        return inv_scale

        #invert difference
        # inv_diff = self.inverse_difference(last_ob, inv_scale)
        # return inv_diff

    # inverse data transform on forecasts
    def inverse_transform_forecasts(self, true_data, forecasts, seq_len):
        inverted = []
        print("Inverse Transform Forecasts...")

        #iterate through each forecast
        for i in range(len(forecasts)):
            if (i % 10) == 0:
                print("forecast #", str(i))

            # create array from forecast
            forecast = np.array(forecasts[i])
            # print("   FORECAST shape: ", str(forecast.shape))
            forecast = forecast.reshape(1, len(forecast))

            # invert scaling
            inv_scale = self.scaler.inverse_transform(forecast)
            inv_scale = inv_scale[0, :].flatten()

            # invert differencing
            index = i * seq_len
            last_ob = true_data[index]
            inv_diff = self.reshape_scaled_data(last_ob, inv_scale)
            inverted.append(inv_diff)
            
        inverted = inverted
        # print("INVERTED shape: ", str(inverted.shape))
        return inverted

    def get_len_train(self):
        return self.len_train

    def get_test_data(self, seq_len, normalise):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test_norm[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x,y

    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def get_total_data(self, seq_len, normalise):
        """Use for debugging: if we want to plot the total data or perform some other operations on it"""
        data_x = []
        data_y = []
        for i in range(self.len_total ):
            x, y = self._next_window_total(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        data_y = np.array(data_y)

        # shift the data up so we can see it in the plot
        return data_y

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def generate_test_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_test - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_test - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)


    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_train_norm[i:i+seq_len]
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    def _next_window_total(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_total_norm[i:i+seq_len]
        x = window[:-1]
        y = window[-1, [0]]
        return x, y
