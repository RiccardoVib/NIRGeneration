import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal.windows import tukey


class DataGeneratorPickles(Dataset):
    def __init__(self, data_dir, filename, seq_len, r=None):
        self.data_dir = data_dir
        self.filename = filename
        self.seq_len = seq_len
        self.r = r

        self.x, self.y, self.z = self.prepareXYZ()

        self.x = self.x[:]
        self.y = self.y[:]
        self.z = self.z[:, :, 0:1]
        
        self.x = self.x.reshape(self.x.shape[0], -1, self.seq_len)
        self.y = self.y.reshape(self.y.shape[0], -1, self.seq_len)
        self.z = np.repeat(self.z, self.x.shape[1], axis=1)

        self.x = self.x.reshape(-1, self.seq_len, 1)
        self.y = self.y.reshape(-1, self.seq_len, 1)
        self.z = self.z.reshape(-1, 1, self.z.shape[-1])

    def prepareXYZ(self):
        full_path = os.path.normpath(os.path.join(self.data_dir, self.filename))
        with open(full_path, 'rb') as file_data:
            Z = pickle.load(file_data)

        x = np.array(Z['x'][:, :], dtype=np.float32)
        y = np.array(Z['y'][:, :], dtype=np.float32)
        z = np.array(Z['z'][:, :], dtype=np.float32)

        if z.shape[0] < z.shape[1]:
            z = z.T

        alpha = 0.000005
        window = tukey(x.shape[1], alpha=alpha).astype(np.float32).reshape(1, -1)
        x = x * window
        y = y * window

        if x.shape[0] == 1:
            x = np.repeat(x, y.shape[0], axis=0)

        #N = int(x.shape[1] / self.mini_batch_size)

        z = z[:, np.newaxis, :]
        x = x[:, :48000*2]
        y = y[:, :48000*2]

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

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        X = self.x[idx]
        Y = self.y[idx]
        Z = self.z[idx]

        if self.r is not None:
            Y = Y[:, self.r:]

        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        Z = torch.tensor(Z, dtype=torch.float32)

        return (X, Z), Y
