import pickle
import os
import numpy as np
from tensorflow.keras.utils import Sequence
from scipy.signal.windows import tukey
import matplotlib.pyplot as plt


class DataGeneratorPickles(Sequence):

    def __init__(self, data_dir, filename, mini_batch_size, batch_size, model_t, r=None):
        """
        Initializes a data generator object
          :param data_dir: the directory in which data are stored
          :param output_size: output size
          :param batch_size: The size of each batch returned by __getitem__
        """
        super().__init__()

        self.indices = None
        self.data_dir = data_dir
        self.filename = filename
        self.mini_batch_size = mini_batch_size
        self.model_t = model_t
        self.r = r
        self.x, self.y, self.z = self.prepareXYZ(data_dir, filename)
        self.batch_size = np.min((self.x.shape[0], batch_size))
        N = int((self.x.shape[0]) / self.batch_size)  # how many iteration
        lim = int(N * self.batch_size)  # how many samples

        self.x = self.x[:lim]
        self.y = self.y[:lim]
        self.z = self.z[:lim]

        self.x = self.x.reshape(self.x.shape[0], -1, self.mini_batch_size)
        self.y = self.y.reshape(self.y.shape[0], -1, self.mini_batch_size)
        self.z = np.repeat(self.z, self.x.shape[1], axis=1)

        self.x = self.x.reshape(-1, self.mini_batch_size, 1)
        self.y = self.y.reshape(-1, self.mini_batch_size, 1)
        self.z = self.z.reshape(-1, 1, self.z.shape[-1])

        if model_t in ['lstm', 'tcn', 'ssm']:
            self.z = np.repeat(self.z, self.x.shape[1], axis=1)

        self.batch_size = np.min((self.x.shape[0], batch_size))

        self.training_steps = (self.x.shape[0] // self.batch_size)
        self.on_epoch_end()

    def prepareXYZ(self, data_dir, filename):
        file_data = open(os.path.normpath('/'.join([data_dir, filename])), 'rb')
        Z = pickle.load(file_data)
        x = np.array(Z['x'][:, :], dtype=np.float32)
        y = np.array(Z['y'][:, :], dtype=np.float32)
        z = np.array(Z['z'][:, :], dtype=np.float32)
        if z.shape[0] < z.shape[1]:
            z = z.T

        x = x * np.array(tukey(x.shape[1], alpha=0.000005), dtype=np.float32).reshape(1, -1)
        y = y * np.array(tukey(x.shape[1], alpha=0.000005), dtype=np.float32).reshape(1, -1)

        if x.shape[0] == 1:
            x = np.repeat(x, y.shape[0], axis=0)

        N = int((x.shape[1]) / self.mini_batch_size)  # how many iteration
        lim = int(N * self.mini_batch_size)  # how many samples

        x = x[:, :lim]
        y = y[:, :lim]

        z = z[:, np.newaxis, :]

        return x, y, z

    def getXY(self):

        Xs, Ys = [], []
        for idx in range(self.__len__()):
            indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

            X = np.array(self.x[indices])
            Y = np.array(self.y[indices])
            if self.r is not None:
                Y = Y[:, self.r:]
            Xs.append(X)
            Ys.append(Y)

        return np.array(Xs), np.array(Ys)

    def on_epoch_end(self):
        self.indices = np.arange(self.x.shape[0])

    def __len__(self):
        return int(self.training_steps)

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):

        # get the indices of the requested batch
        indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        X = np.array(self.x[indices])
        Y = np.array(self.y[indices])
        Z = np.array(self.z[indices])
        if self.r is not None:
            Y = Y[:, self.r:]

        return ((X, Z), Y)