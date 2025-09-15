import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Conv1D, Activation
from ConditioningLayer import FiLM



def residual_block(x, dilation_rate, nb_filters, kernel_size, padding):
    """
    Residual block for the TCN
    """
    # First dilated causal convolution
    prev_x = x

    # Causal dilated convolution
    x = Conv1D(filters=nb_filters,
               kernel_size=kernel_size,
               dilation_rate=dilation_rate,
               padding=padding)(x)

    # Activation and dropout
    x = Activation('relu')(x)

    # Second dilated causal convolution
    x = Conv1D(filters=nb_filters,
               kernel_size=kernel_size,
               dilation_rate=dilation_rate,
               padding=padding)(x)
    x = Activation('relu')(x)

    # Add the residual connection if needed
    if prev_x.shape[-1] != nb_filters:
        prev_x = Conv1D(nb_filters, kernel_size=1, padding='same')(prev_x)

    return layers.add([prev_x, x])


def create_tcn_model(batch_size, mini_batch_size, cond_dim, units=64, kernel_size=9,
                     dilations=[1, 2, 4, 8, 16, 32],  return_sequences=True):
    """
    Creates a Temporal Convolutional Network (TCN) model

    Parameters:
    - input_shape: shape of input data (sequence_length, features)
    - output_units: number of output units
    - nb_filters: number of filters in convolutional layers
    - kernel_size: size of convolutional kernels
    - dilations: list of dilation rates for each layer
    - dropout_rate: dropout rate used in the network
    - return_sequences: whether to return the full sequence or just the last output

    Returns:
    - A compiled TCN model
    """
    receptive_field = 1
    for dilation in dilations:
        # Each layer adds (kernel_size - 1) * dilation to the receptive field
        receptive_field += (kernel_size - 1) * dilation

    # Input layer
    # Defining inputs
    inputs = tf.keras.layers.Input(
        shape=(mini_batch_size, 1), name='input')

    conds = tf.keras.layers.Input(
        shape=(mini_batch_size, cond_dim), name='cond')

    x = inputs

    # Causal padding
    padding = 'causal'  # This ensures the network is causal - no info leakage from future to past

    # Create TCN residual blocks with increasing dilation
    for dilation_rate in dilations:
        x = residual_block(x, dilation_rate, units, kernel_size, padding)
        x = FiLM(in_size=units)(x, conds)

    # Output layer
    if return_sequences:
        outputs = Conv1D(filters=1, kernel_size=1, activation='linear', padding='same')(x)
        outputs = outputs[:, receptive_field:]
    else:
        # Only take the last output for sequence-to-one prediction
        outputs = layers.Lambda(lambda z: z[:, -1, :])(x)
        #outputs = Dense(output_units, activation='linear')(x)

    # Build and compile the model
    model = tf.keras.models.Model([inputs, conds], outputs)
    model.summary()

    return model, receptive_field