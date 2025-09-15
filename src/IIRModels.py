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
import numpy as np
class IIR(tf.keras.layers.Layer):
    def __init__(self, filter_length, mini_batch_size, batch_size, dry_wet):
        super(IIR, self).__init__()
        self.mini_batch_size = mini_batch_size
        self.filter_length = filter_length
        self.batch_size = batch_size
        self.dry_wet = dry_wet

        self.pre0 = tf.keras.layers.Dense(filter_length)  # //2
        self.prelu = tf.keras.layers.PReLU()
        self.film = FiLM(filter_length)
        self.glu = GLU(filter_length)

        self.b0  = tf.Variable(tf.ones(self.batch_size,))
        self.filters = tf.keras.layers.Dense(filter_length-1, activation='softsign')
        # inputs = tf.Variable(tf.ones((self.batch_size, self.mini_batch_size, 1)))
        # c = tf.Variable(tf.ones((self.batch_size, 1, 4)))

    def call(self, inputs, c):
        """
         Differentiable IIR (Infinite Impulse Response) filter implementation.

         This layer implements a general IIR filter with learnable coefficients:
         y[n] = sum(b[i] * x[n-i]) - sum(a[i] * y[n-i])

         Where:
         - b are the feedforward coefficients
         - a are the feedback coefficients
         - x is the input signal
         - y is the output signal
         """
        c_ = self.pre0(c)
        c_ = self.prelu(c_)
        c_ = self.film(c_, c)
        c_ = self.glu(c_)
        filters = self.filters(c_)

        b_coeffs, a_coeffs = tf.split(filters, [self.filter_length // 2 - 1, self.filter_length // 2], axis=-1)
        b_coeffs = tf.concat([self.b0[:, None, None], b_coeffs], axis=-1)

        b_coeffs = b_coeffs[:,0,:]
        a_coeffs = a_coeffs[:,0,:]
        # Apply filter using scan operation for efficiency
        def filter_step(carry, x_t):
            """Single step of IIR filtering."""
            x_buffer = carry[0][0]
            y_buffer = carry[0][1]

            # Update input buffer
            x_buffer = tf.concat([tf.expand_dims(x_t, axis=1), x_buffer[:, :-1, :]], axis=1)

            # Compute feedforward contribution
            ff_contrib = tf.reduce_sum(
                b_coeffs[:, :, tf.newaxis] * x_buffer,
                axis=1
            )

            # Compute feedback contribution
            fb_contrib = tf.reduce_sum(
                a_coeffs[:, :, tf.newaxis] * y_buffer,
                axis=1
            )

            # Current output
            y_t = ff_contrib - fb_contrib

            # Update output buffer
            y_buffer = tf.concat([y_t[:, :, None], y_buffer[:, :-1, : ]], axis=1)
            return (x_buffer, y_buffer), y_t

        # Initialize buffers
        x_buffer = tf.zeros((self.batch_size, self.filter_length // 2, 1))
        y_buffer = tf.zeros((self.batch_size, self.filter_length // 2, 1))

        # Apply filter across time dimension

        (final_carry, outputs) = tf.scan(
            filter_step,
            tf.transpose(inputs, [1, 0, 2]),  # (time, batch, channels)
            initializer=((x_buffer, y_buffer), tf.zeros((self.batch_size, 1)))
        )

        # Transpose back to (batch, time, channels)
        y = tf.transpose(outputs, [1, 0, 2])

        if self.dry_wet:
            return self.dry * inputs + y
        else:
            return y




def create_iir_model(filter_length, cond_dim, mini_batch_size, batch_size, dry_wet):

    # Defining inputs
    inputs = tf.keras.layers.Input(batch_shape=(batch_size, mini_batch_size, 1), name='input')
    c = tf.keras.layers.Input(batch_shape=(batch_size, 1, cond_dim), name='c')

    outputs = IIR(filter_length=filter_length, mini_batch_size=mini_batch_size, batch_size=batch_size, dry_wet=dry_wet)(inputs, c)

    model = tf.keras.models.Model([inputs, c], outputs)
    model.summary()
    return model

