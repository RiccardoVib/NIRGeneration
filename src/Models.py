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

class IR_Conv(tf.keras.layers.Layer):
    def __init__(self, filter_length, mini_batch_size, batch_size, cond_type, dry_wet):
        super(IR_Conv, self).__init__()
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
        # Calculate delay for linear-phase filter
        delay = (self.filter_length - 1) // 2
        inputs_padded = tf.pad(inputs, [[0, 0], [self.filter_length-1, delay], [0, 0]])

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
            outputs_.append(outputs[0, delay:delay + self.mini_batch_size])

        outputs = tf.stack(outputs_)
        if self.dry_wet:
            return self.dry * inputs + outputs
        else:
            return outputs
            
class FFT_Conv(tf.keras.layers.Layer):
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

        if self.dry_wet:
            return self.dry * inputs + y
        else:
            return y

def create_model_FFT(filter_length, cond_dim, mini_batch_size, batch_size, cond_type, dry_wet):

    # Defining inputs
    inputs = tf.keras.layers.Input(batch_shape=(batch_size, mini_batch_size, 1), name='input')
    c = tf.keras.layers.Input(batch_shape=(batch_size, 1, cond_dim), name='c')

    outputs = FFT_Conv(filter_length=filter_length, mini_batch_size=mini_batch_size, batch_size=batch_size, cond_type=cond_type, dry_wet=dry_wet)(inputs, c)

    model = tf.keras.models.Model([inputs, c], outputs)
    model.summary()
    return model


def create_model_IR(filter_length, cond_dim, mini_batch_size, batch_size, cond_type, dry_wet):

    # Defining inputs
    inputs = tf.keras.layers.Input(batch_shape=(batch_size, mini_batch_size, 1), name='input')
    c = tf.keras.layers.Input(batch_shape=(batch_size, 1, cond_dim), name='c')

    outputs = IR_Conv(filter_length=filter_length, mini_batch_size=mini_batch_size, batch_size=batch_size, cond_type=cond_type, dry_wet=dry_wet)(inputs, c)

    model = tf.keras.models.Model([inputs, c], outputs)
    model.summary()
    return model
