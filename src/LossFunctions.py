# Copyright (C) 2025 Riccardo Simionato, University of Oslo
# Inquiries: riccardo.simionato.vib@gmail.com.com
#
# This code is free software: you can redistribute it and/or modify it under the terms
# of the GNU Lesser General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Less General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this code.
# If not, see <http://www.gnu.org/licenses/>.
#
# If you use this code or any part of it in any program or publication, please acknowledge
# its authors by adding a reference to this publication:
#
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

class NMSELoss(tf.keras.losses.Loss):
    def __init__(self, delta=1e-6, name="NMSE", **kwargs):
        super().__init__(name=name, **kwargs)
        self.delta = delta
    def call(self, y_true, y_pred):

        loss = tf.divide(tf.keras.losses.MeanSquaredError()(y_true, y_pred), tf.norm(y_true, ord=1) + self.delta)
        return loss

    def get_config(self):
        config = {
            'delta': self.delta
        }
        base_config = super().get_config()
        return {**base_config, **config}



class STFT_loss(tf.keras.losses.Loss):
    def __init__(self, m=[32, 64, 128, 256], MSE=False, name="STFT", **kwargs):
        super().__init__(name=name, **kwargs)
        self.m = m
        self.delta = 1e-6#0.000001,
        self.MSE = MSE#0.000001,
    def call(self, y_true, y_pred):

        y_true = tf.reshape(y_true, [1, -1])
        y_pred = tf.reshape(y_pred, [1, -1])
        loss = 0

        for i in range(len(self.m)):
            Y_true = K.abs(tf.signal.stft(y_true, frame_length=self.m[i], frame_step=self.m[i] // 4, pad_end=False))
            Y_pred = K.abs(tf.signal.stft(y_pred, frame_length=self.m[i], frame_step=self.m[i] // 4, pad_end=False))

            Y_true = K.pow(Y_true, 2)
            Y_pred = K.pow(Y_pred, 2)

            l_true = K.log(Y_true + 1)
            l_pred = K.log(Y_pred + 1)

            loss += tf.divide(tf.norm((Y_true - Y_pred), ord=1), tf.norm(Y_true + self.delta, ord=1) + self.delta)
            loss += tf.divide(tf.norm((l_true - l_pred), ord=1), tf.norm(l_true + self.delta, ord=1) + self.delta)

        if self.MSE:
            mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        else:
            mse = 0
        return 0.001*(loss/len(self.m)) + mse

    def get_config(self):
        config = {
            'm': self.m,
            'delta': self.delta,
            'MSE': self.MSE
        }
        base_config = super().get_config()
        return {**base_config, **config}



class phase_loss(tf.keras.losses.Loss):
    def __init__(self, m=[32, 64, 128, 256, 512, 1024], fft_size=2048, num_samples=1200, name="phase", **kwargs):
        super().__init__(name=name, **kwargs)
        self.m = m
        self.fft_size = fft_size
        self.num_samples = num_samples

    def call(self, y_true, y_pred):

        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        y_true = tf.reshape(y_true, [1, -1])
        y_pred = tf.reshape(y_pred, [1, -1])
        loss = 0
        pads = [[0, 0] for _ in range(2)]
        pad_amount = int((self.fft_size - self.num_samples) // 2)  # Symmetric even padding like librosa.
        pads[1] = [pad_amount, pad_amount]
        y_true = tf.pad(y_true, pads, mode='CONSTANT', constant_values=0)
        y_pred = tf.pad(y_pred, pads, mode='CONSTANT', constant_values=0)

        for i in range(len(self.m)):

            stft_t = tf.signal.stft(y_true, frame_length=self.m[i], frame_step=self.m[i] // 4, pad_end=True)
            stft_p = tf.signal.stft(y_pred, frame_length=self.m[i], frame_step=self.m[i] // 4, pad_end=True)
            Y_true = (tf.math.imag(stft_t))
            Y_pred = (tf.math.imag(stft_p))
            #r_t = (tf.math.real(stft_t))
            #r_p = (tf.math.real(stft_p))
            #phase = tf.math.atan2(phase)

            #Y_true = tf.math.atan2(r_t, i_t)
            #Y_pred = tf.math.atan2(r_p, i_p)

            #loss += tf.divide(tf.norm((Y_true - Y_pred), ord=1), tf.norm(Y_true, ord=1))
            loss += tf.norm((Y_true - Y_pred), ord=1)

        return loss / len(self.m)

    def get_config(self):
        config = {
            'm': self.m,
            'fft_size': self.fft_size,
            'num_samples': self.num_samples
        }
        base_config = super().get_config()
        return {**base_config, **config}
