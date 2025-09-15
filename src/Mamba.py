import tensorflow as tf
from einops import rearrange, repeat
import numpy as np
import math
from ConditioningLayer import FiLM

def selective_scan(u, delta, A, B, C, D, last_state, stateful):
    # first step of A_bar = exp(ΔA), i.e., ΔA
    dA = tf.einsum('bld,dn->bldn', delta, A)
    dB_u = tf.einsum('bld,bld,bln->bldn', delta, u, B)

    dA_cumsum = tf.pad(dA[:, 1:], [[0, 0], [1, 0], [0, 0], [0, 0]])

    dA_cumsum = tf.math.cumsum(dA_cumsum, axis=1)
    dA_cumsum = tf.exp(dA_cumsum)

    x = dB_u / (dA_cumsum + 1e-12)
    x = tf.math.cumsum(x, axis=1) * dA_cumsum

    if stateful:
        dA_cumsum_l = tf.math.cumsum(dA, axis=1)
        dA_cumsum_l = tf.exp(dA_cumsum_l)
        dA_cumsum_l *= tf.expand_dims(last_state, axis=1)
        x = x + dA_cumsum_l

    last_state = x[:, -1]
    y = tf.einsum('bldn,bln->bld', x, C)

    return y + u * D, last_state

class MambaBlock(tf.keras.layers.Layer):
    def __init__(self, model_states, model_input_dims, model_internal_dim, conv_kernel_size, delta_t_rank, batch_size=22, mini_batch_size=2048, conv_use_bias=True, dense_use_bias=True, stateful=False):
        super(MambaBlock, self).__init__()
        self.model_states = model_states
        self.model_input_dims = model_input_dims
        self.model_internal_dim = model_internal_dim
        self.conv_kernel_size = conv_kernel_size
        self.delta_t_rank = delta_t_rank
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size

        self.conv_use_bias = conv_use_bias
        self.dense_use_bias = dense_use_bias
        self.stateful = stateful

        self.in_projection = tf.keras.layers.Dense(
            self.model_internal_dim * 2,
            input_shape=(self.model_input_dims,), use_bias=False, trainable=True)

        self.conv1d = tf.keras.layers.Conv1D(
            filters=self.model_internal_dim,
            use_bias=self.conv_use_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.model_internal_dim,
            data_format='channels_first',
            padding='causal', trainable=True
        )

        # this layer takes in current token 'x'
        # and outputs the input-specific Δ, B, C (according to S6)
        self.x_projection = tf.keras.layers.Dense(self.delta_t_rank + self.model_states * 2, use_bias=False, trainable=True)

        # this layer projects Δ from delta_t_rank to the mamba internal
        # dimension
        self.delta_t_projection = tf.keras.layers.Dense(self.model_internal_dim,
                                               input_shape=(self.delta_t_rank,), use_bias=True, trainable=True)

        self.A = repeat(
            tf.range(1, self.model_states + 1, dtype=tf.float32),
            'n -> d n', d=self.model_internal_dim)

        self.A_log = tf.Variable(
            tf.math.log(self.A),
            trainable=True, dtype=tf.float32,
            name=f"SSM_A_log")

        self.D = tf.Variable(
            np.ones(self.model_internal_dim),
            trainable=True, dtype=tf.float32,
            name=f"SSM_D")

        self.out_projection = tf.keras.layers.Dense(
            self.model_input_dims,
            input_shape=(self.model_internal_dim,),
            use_bias=self.dense_use_bias, trainable=True)
        #x = tf.Variable(tf.random.normal([self.batch_size, self.model_internal_dim, 1]), dtype='float32')
        self.reset_states()

    def reset_states(self):
        self.state = tf.Variable(
            tf.zeros((self.batch_size, self.model_internal_dim, self.model_states), dtype=tf.float32), name='state',
            trainable=False)

    def call(self, x):

        (batch_size, seq_len, dimension) = x.shape

        x_and_res = self.in_projection(x)  # shape = (batch, seq_len, 2 * model_internal_dimension)
        (x, res) = tf.split(x_and_res,
                            [self.model_internal_dim,
                             self.model_internal_dim], axis=-1)


        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :seq_len]
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = tf.nn.swish(x)

        last_state = self.state[:self.batch_size]
        res_state = self.state[self.batch_size:]

        y, y_state = self.ssm(x, last_state=last_state, stateful=self.stateful)
        if self.stateful:
            self.state.assign(tf.concat([y_state, res_state], axis=0))

        y = y * tf.nn.swish(res)
        return self.out_projection(y)

    def ssm(self, x, last_state, stateful):

        (d_in, n) = self.A_log.shape

        A = -tf.exp(tf.cast(self.A_log, tf.float32))  # shape -> (d_in, n)
        D = tf.cast(self.D, tf.float32)

        x_dbl = self.x_projection(x)  # shape -> (batch, seq_len, delta_t_rank + 2*n)

        (delta, B, C) = tf.split(
            x_dbl,
            num_or_size_splits=[self.delta_t_rank, n, n],
            axis=-1)  # delta.shape -> (batch, seq_len) & B, C shape -> (batch, seq_len, n)

        delta = tf.nn.softplus(self.delta_t_projection(delta))  # shape -> (batch, seq_len, model_input_dim)

        return selective_scan(x, delta, A, B, C, D, last_state, stateful)


class MambaLay(tf.keras.layers.Layer):
    def __init__(self, model_states, projection_expand_factor=2, model_input_dims=2, conv_kernel_size=4, batch_size=8, mini_batch_size=1024, stateful=False, type=tf.float32):
        super(MambaLay, self).__init__()
        self.model_internal_dim = int(projection_expand_factor * model_input_dims)
        self.mini_batch_size = mini_batch_size
        self.model_input_dims = model_input_dims
        self.batch_size = batch_size
        self.stateful = stateful
        self.conv_kernel_size = conv_kernel_size
        self.delta_t_rank = math.ceil(model_input_dims / 2)  # 16
        self.model_states = model_states
        conv_use_bias, dense_use_bias = True, True
        self.block = MambaBlock(model_states=self.model_states,
                                   model_input_dims=self.model_input_dims,
                                   model_internal_dim=self.model_internal_dim,
                                   conv_kernel_size=self.conv_kernel_size,
                                   delta_t_rank=self.delta_t_rank,
                                   batch_size=self.batch_size,
                                   mini_batch_size=self.mini_batch_size,
                                   conv_use_bias=conv_use_bias,
                                   dense_use_bias=dense_use_bias,
                                   stateful=self.stateful)

    def reset_states(self):
        self.block.reset_states()

    def call(self, x):
        x = self.block(x)

        return x


def create_Mamba_model(mini_batch_size, batch_size=20,
                       projection_expand_factor=2, conv_kernel_size=4, model_states=8, cond_dim=3, units=4, stateful=False):
    inp = tf.keras.layers.Input(shape=(mini_batch_size, 1), name='input_ids')

    x = tf.keras.layers.Dense(units)(inp)  #####
    x = MambaLay(model_states=model_states, projection_expand_factor=projection_expand_factor, model_input_dims=units,
                 conv_kernel_size=conv_kernel_size, batch_size=batch_size, mini_batch_size=mini_batch_size, stateful=stateful)(x)

    x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)

    ##########
    # threshold and ratio
    z = tf.keras.layers.Input(shape=(mini_batch_size, cond_dim), name='params_inputs')

    x = FiLM(in_size=units)(x, z)

    x = MambaLay(model_states=model_states, projection_expand_factor=projection_expand_factor,
                 model_input_dims=units,
                 conv_kernel_size=conv_kernel_size, batch_size=batch_size, mini_batch_size=mini_batch_size, stateful=stateful)(x)

    x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)  ####

    x = tf.keras.layers.Dense(1, name='OutLayer')(x)
    x = x + inp
    model = tf.keras.models.Model(inputs=[inp, z], outputs=x, name='Bamba')
    model.summary()

    return model
