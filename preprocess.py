import numpy as np
import os
from scipy.io import loadmat

class CWRUPreprocessor():
    def __init__(self, data_dir, data_list, data_info, condition_list, window_size=1200, num_data=4000):
        self.data_dir = data_dir
        self.data_list = data_list
        self.data_info = data_info
        self.condition_list = condition_list
        self.window_size = window_size
        self.num_data = num_data

    def get_data(self):
        num_classes = len(self.data_list)
        X_data = np.zeros((self.num_data * num_classes, self.window_size))
        Y_data = np.zeros((self.num_data * num_classes, num_classes))

        for k, filename in enumerate(self.data_list):
            signal = loadmat(os.path.join(self.data_dir, filename))[self.data_info[k]].squeeze()

            # sliding window
            ix = list(range(0, len(signal) - self.window_size,
                            int((len(signal) - 100) / self.num_data)))[:self.num_data]

            for i, idx in enumerate(ix):
                X_data[i + k * self.num_data] = signal[idx:idx + self.window_size]
                Y_data[i + k * self.num_data] = np.eye(num_classes)[k]

        return X_data, Y_data
