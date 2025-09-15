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