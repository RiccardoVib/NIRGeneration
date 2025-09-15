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
from ConditioningLayer import FiLM

def create_lstm_model(batch_size, mini_batch_size, cond_dim, units, return_sequences=True):

    # Defining inputs
    inputs = tf.keras.layers.Input(
        shape=(mini_batch_size, 1), name='input')

    conds = tf.keras.layers.Input(
        shape=(mini_batch_size, cond_dim), name='cond')

    outputs = tf.keras.layers.LSTM(
                units, stateful=False, return_sequences=return_sequences, name="LSTM")(inputs)
    outputs = FiLM(in_size=units)(outputs, conds)

    outputs = tf.keras.layers.LSTM(
                units, stateful=False, return_sequences=return_sequences, name="LSTM2")(outputs)

    outputs = tf.keras.layers.Dense(1, name='OutLayer')(outputs)
    outputs = outputs + inputs

    model = tf.keras.models.Model([inputs, conds], outputs)
    model.summary()

    return model
