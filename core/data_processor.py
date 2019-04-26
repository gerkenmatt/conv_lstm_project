import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pandas import Series


class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, split, cols):
        dataframe = pd.read_csv(filename)
        i_split = int(len(dataframe) * split)
        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test  = dataframe.get(cols).values[i_split:]
        self.data_total = dataframe.get(cols).values[:]
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_total  = len(self.data_total)
        self.len_train_windows = None

    # create a differenced series
    def difference(self, dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            # new value is the difference between the two adjacent points
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return Series(diff)

    # invert differenced forecast
    def inverse_difference(self, last_ob, dataset):
        # invert first forecast
        inverted = list()
        inverted.append(dataset[0] + last_ob)
        # propagate difference forecast using inverted first value
        for i in range(1, len(dataset)):
            inverted.append(dataset[i] + inverted[i-1])
        return inverted


    # difference the data then scale it to values between -1, 1
    def transform_data(self, raw_values):

        # transform data to be stationary: values are now the differences between adjacent points
        diff_series = self.difference(raw_values, 1)
        diff_values = diff_series.values
        diff_values = diff_values.reshape(len(diff_values), 1)
        
        # rescale values to be between -1, 1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_values = scaler.fit_transform(diff_values)
        scaled_values = scaled_values.reshape(len(scaled_values), 1)
        return scaler, scaled_values

    # inverse data transform on forecasts
    def inverse_transform(self, last_ob, dataset, scaler):
        
        print("INVERSE TRANSFORM: ")
        print("   dataset.shape: ", str(dataset.shape))
        print("   last_ob: ", str(last_ob))

        #inverse scaling
        inv_scale = scaler.inverse_transform(dataset.flatten())
        print("inv_scale: ", str(inv_scale))

        inv_scale = inv_scale[0, :]

        #invert difference
        inv_diff = self.inverse_difference(last_ob, inv_scale)
        return inv_diff


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
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

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
        return np.array(data_x), np.array(data_y), float(self.data_train[0][0])

    def get_total_data(self, seq_len, normalise):
        print("len_train: ", str(self.len_train) )
        print("len_test: ", str(self.len_test) )
        print("len_total: ", str(self.len_total) )
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
        window = self.data_train[i:i+seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    def _next_window_total(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_total[i:i+seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        
        #using single window most of the time, so this doesn't matter
        for window in window_data:
            normalised_window = []

            # iterate through each time-point in the series
            for col_i in range(window.shape[1]):
                # normalisation arithmetic
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)

            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)

