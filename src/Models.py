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
from ConditioningLayer import FiLM, GLU

class _Conv(tf.keras.layers.Layer):
    def __init__(self, filter_length, mini_batch_size, batch_size, cond_type, dry_wet):
        super(Conv, self).__init__()
        self.mini_batch_size = mini_batch_size
        self.filter_length = filter_length
        self.batch_size = batch_size
        self.cond_type = cond_type
        self.dry_wet = dry_wet

        if self.dry_wet:
            self.dry = tf.Variable(tf.random.normal((1, 1, 1), stddev=1.), trainable=True)

        if cond_type == 'film':
            self.pre = tf.keras.layers.Dense(filter_length*2)
            self.prelu = tf.keras.layers.PReLU()
            self.pre2 = tf.keras.layers.Dense(filter_length*2)
            self.prelu2 = tf.keras.layers.PReLU()
            self.film = FiLM(filter_length)
            self.glu = GLU(filter_length)
            filter_shape = (filter_length, 1, 1)
            self.filters = tf.Variable(tf.random.normal((filter_shape), stddev=0.0001), trainable=True)

        else:
            self.pre = tf.keras.layers.Dense(filter_length*2)
            self.prelu = tf.keras.layers.PReLU()
            self.pre2 = tf.keras.layers.Dense(filter_length*2)
            self.prelu2 = tf.keras.layers.PReLU()
            self.filters = tf.keras.layers.Dense(filter_length, activation='softsign')
        #inputs = tf.Variable(tf.ones((self.batch_size, self.mini_batch_size, 1)))
        #c = tf.Variable(tf.ones((self.batch_size, 1, 4)))

    def call(self, inputs, c):

        if self.cond_type == 'film':
            filters = tf.reshape(self.filters, [1, self.filter_length])
            filters = tf.repeat(filters, self.batch_size, axis=0)
            f = tf.nn.softsign(filters)
            c = c[:, 0, :]
            c_ = self.pre(c)
            c_ = self.prelu(c_)
            c_2 = self.pre2(c_)
            c_2 = self.prelu2(c_2)
            c_3 = self.film(f, c_2)
            filters = self.glu(c_3)
            filters = tf.reshape(filters, [self.batch_size, self.filter_length, 1, 1])

        else:
            c_ = self.pre(c)
            c_ = self.prelu(c_)
            c_2 = self.pre2(c_)
            c_2 = self.prelu2(c_2)
            filters = self.filters(c_2)
            filters = tf.reshape(filters, [self.batch_size, self.filter_length, 1, 1])
        
        inputs_padded = tf.pad(inputs, [[0, 0], [self.filter_length-1, self.filter_length-1], [0, 0]])

        outputs_ = []
        for i in range(self.batch_size):
            outputs = tf.nn.convolution(
                inputs_padded[i:i+1],
                filters[i],
                strides=1,
                padding='VALID',
                data_format=None,
                dilations=None,
                name='filter'
            )

            outputs_.append(outputs[0,:self.mini_batch_size])

        outputs = tf.stack(outputs_)
        if self.dry_wet:
            return self.dry * inputs + outputs
        else:
            return outputs


class Conv(tf.keras.layers.Layer):
    def __init__(self, filter_length, mini_batch_size, batch_size, cond_type, dry_wet):
        super(Conv, self).__init__()
        self.mini_batch_size = mini_batch_size
        self.filter_length = filter_length
        self.batch_size = batch_size
        self.cond_type = cond_type
        self.dry_wet = dry_wet

        self.pre0 = tf.keras.layers.Dense(filter_length)  # //2
        self.prelu = tf.keras.layers.PReLU()
        self.film = FiLM(filter_length)
        self.glu = GLU(filter_length)

        self.filters = tf.keras.layers.Dense(filter_length, activation='softsign')
        # inputs = tf.Variable(tf.ones((self.batch_size, self.mini_batch_size, 1)))
        # c = tf.Variable(tf.ones((self.batch_size, 1, 4)))

    def call(self, inputs, c):

        c_ = self.pre0(c)
        c_ = self.prelu(c_)
        c_ = self.film(c_, c)
        c_ = self.glu(c_)
        filters = self.filters(c_)

        conv_len = self.mini_batch_size  + self.filter_length - 1

        x_padded = tf.pad(inputs, [[0,0], [0, conv_len - self.mini_batch_size ], [0,0]])
        h_padded = tf.pad(filters, [[0,0], [0,0], [0, conv_len - self.filter_length]])

        # Convert to complex
        X = tf.signal.fft(tf.cast(x_padded[:,:,0], tf.complex64))
        H = tf.signal.fft(tf.cast(h_padded[:,0,:], tf.complex64))

        y = tf.signal.ifft(X * H)

        y = tf.math.real(y[:, :self.mini_batch_size])

        # fft_size = self.filter_length + self.filter_length - 1
        # H = tf.signal.fft(tf.cast(tf.pad(filters[:,0,:], [[0,0], [0, fft_size - self.filter_length]]), tf.complex64))
        #
        # # Process input in blocks
        # y = tf.zeros(self.mini_batch_size + fft_size - 1, dtype=tf.float32)
        #
        # for i in range(0, self.mini_batch_size, self.filter_length):
        #     # Extract block
        #     block = inputs[:, i:i + self.filter_length, 0]
        #
        #     # Pad block to fft_size
        #     block_padded = tf.pad(block, [[0,0], [0, fft_size - self.filter_length]])
        #
        #     # FFT, multiply, IFFT
        #     X_block = tf.signal.fft(tf.cast(block_padded, tf.complex64))
        #     Y_block = tf.signal.ifft(X_block * H)
        #     y_block = tf.math.real(Y_block)
        #
        #     # Overlap-add
        #     end_idx = min(i + fft_size, len(y))
        #     y = tf.tensor_scatter_nd_add(y,
        #                                  tf.expand_dims(tf.range(i, end_idx), 1),
        #                                  y_block[:end_idx - i])

        if self.dry_wet:
            return self.dry * inputs + y
        else:
            return y

def create_model(filter_length, cond_dim, mini_batch_size, batch_size, cond_type, dry_wet):

    # Defining inputs
    inputs = tf.keras.layers.Input(batch_shape=(batch_size, mini_batch_size, 1), name='input')
    c = tf.keras.layers.Input(batch_shape=(batch_size, 1, cond_dim), name='c')

    outputs = Conv(filter_length=filter_length, mini_batch_size=mini_batch_size, batch_size=batch_size, cond_type=cond_type, dry_wet=dry_wet)(inputs, c)

    model = tf.keras.models.Model([inputs, c], outputs)
    model.summary()
    return model

