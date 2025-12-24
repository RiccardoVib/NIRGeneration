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

import os
import tensorflow as tf
from UtilsForTrainings import plotTraining, writeResults, checkpoints, predictWaves
from Models import create_model_FFT, create_model_IR
from DatasetsClass import DataGeneratorPickles
import numpy as np
import random
from Metrics import ESR, RMSE
from LossFunctions import STFT_loss
import sys
import time
from TCN import create_tcn_model
from LSTM import create_lstm_model
from Mamba import create_Mamba_model
from IIRModels import create_iir_model
import matplotlib.pyplot as plt

def train(**kwargs):

    """
      :param data_dir: the directory in which dataset are stored [string]
      :param save_folder: the directory in which the models are saved [string]
      :param batch_size: the size of each batch [int]
      :param learning_rate: the initial leanring rate [float]
      :param units: the number of model's units [int]
      :param input_dim: the input size [int]
      :param model_save_dir: the directory in which models are stored [string]
      :param save_folder: the directory in which the model will be saved [string]
      :param inference: if True it skip the training and it compute only the inference [bool]
      :param dataset: name of the datset to use [string]
      :param epochs: the number of epochs [int]
      :param teacher: if True it is inferring the training set and store in save_folder [bool]
      :param fs: the sampling rate [int]
      :param conditioning_size: the numeber of parameters to be included [int]
    """

    batch_size = kwargs.get('batch_size', 1)
    mini_batch_size = kwargs.get('mini_batch_size', 1)
    learning_rate = kwargs.get('learning_rate', 1e-1)
    filter_length = kwargs.get('filter_length', 16)
    model_save_dir = kwargs.get('model_save_dir', '../../TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    inference = kwargs.get('inference', False)
    dataset = kwargs.get('dataset', None)
    data_dir = kwargs.get('data_dir', '../../../Files/')
    epochs = kwargs.get('epochs', 60)
    fs = kwargs.get('fs', 48000)
    time_loss = kwargs.get('time_loss', True)
    freq_loss = kwargs.get('freq_loss', False)
    cond_type = kwargs.get('cond_type', None)
    dry_wet = kwargs.get('dry_wet', False)
    units = kwargs.get('units', 4)
    model_t = kwargs.get('model_t', '')

    # set all the seed in case reproducibility is desired
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    # start the timer for all the training script
    global_start = time.time()

    # check if GPUs are available and set the memory growing
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if len(gpu) != 0:
        tf.config.experimental.set_memory_growth(gpu[0], True)
    # tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=18000)])

    # create the DataGenerator object to retrieve the data in the test set
    test_gen = DataGeneratorPickles(data_dir, dataset + '_test.pickle', mini_batch_size=mini_batch_size, model_t=model_t, batch_size=batch_size)

    # create the model
    if model_t == 'tcn':
        model, r = create_tcn_model(units=16, cond_dim=test_gen.z.shape[-1], mini_batch_size=mini_batch_size, batch_size=test_gen.batch_size)
        test_gen = DataGeneratorPickles(data_dir, dataset + '_test.pickle', mini_batch_size=mini_batch_size, model_t=model_t, batch_size=batch_size, r=r)
    elif model_t == 'lstm':
        model = create_lstm_model(units=units*2, cond_dim=test_gen.z.shape[-1], mini_batch_size=mini_batch_size, batch_size=test_gen.batch_size)
    elif model_t == 'ssm':
        model = create_Mamba_model(units=units+16, cond_dim=test_gen.z.shape[-1], mini_batch_size=mini_batch_size, batch_size=test_gen.batch_size, model_states=units)
    elif model_t == 'iir':
        model = create_iir_model(filter_length=filter_length, cond_dim=test_gen.z.shape[-1], mini_batch_size=mini_batch_size,
                                  batch_size=test_gen.batch_size, dry_wet=dry_wet)
    elif model_t == 'fft':
        model = create_model_FFT(filter_length=filter_length, cond_dim=test_gen.z.shape[-1], mini_batch_size=mini_batch_size,
                             batch_size=test_gen.batch_size, cond_type=cond_type, dry_wet=dry_wet)
    else:
        model = create_model_IR(filter_length=filter_length, cond_dim=test_gen.z.shape[-1], mini_batch_size=mini_batch_size,
                     batch_size=test_gen.batch_size, cond_type=cond_type, dry_wet=dry_wet)

    batch_size = test_gen.batch_size
    # define callbacks: where to store the weights
    callbacks = []
    ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(model_save_dir, save_folder)

    # if inference is True, it jump directly to the inference section without train the model
    if not inference:
        callbacks += [ckpt_callback, ckpt_callback_latest]
        # load the weights of the last epoch, if any
        # load the best weights of the model
        weights_path = os.path.join(ckpt_dir, "latest.weights.h5")
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            print(f"Loaded weights from {weights_path}")
        else:
            # if no weights are found,the weights are random generated
            print("Initializing random weights.")

        # create the DataGenerator object to retrieve the data in the training set
        if model_t == 'tcn':
            train_gen = DataGeneratorPickles(data_dir, dataset+ '_train.pickle', mini_batch_size=mini_batch_size, batch_size=test_gen.batch_size, model_t=model_t, r=r)
        else:
            train_gen = DataGeneratorPickles(data_dir, dataset+ '_train.pickle', mini_batch_size=mini_batch_size, batch_size=test_gen.batch_size, model_t=model_t)

        # the number of total training steps
        # define the Adam optimizer with initial learning rate, training steps
        opt = tf.keras.optimizers.AdamW(learning_rate=learning_rate)#, clipnorm=1)

        # compile the model with the optimizer and selected loss function
        if time_loss and not freq_loss:
            #loss = NMSELoss()
            loss = 'mse'
        elif freq_loss and not time_loss:
            loss = STFT_loss(m=[256, 512, 1024, 2048, 4096, 8192], MSE=False)
        elif freq_loss and time_loss:
            loss = STFT_loss(m=[256, 512, 1024, 2048, 4096, 8192], MSE=True)

        model.compile(loss=loss, optimizer=opt)

        # defining the array taking the training and validation losses
        loss_training = np.empty(epochs)
        loss_val = np.empty(epochs)
        best_loss = 1e9
        # counting for early stopping
        count = 0
        count2 = 0
        # training loop
        for i in range(epochs):
            # start the timer for each epoch
            start = time.time()
            print('epochs:', i)

            # reset the model's states
            print(model.optimizer.learning_rate)

            results = model.fit(train_gen, epochs=1, verbose=0, shuffle=True, validation_data=test_gen,
                                callbacks=callbacks)

            # store the training and validation loss
            loss_training[i] = results.history['loss'][-1]
            loss_val[i] = results.history['val_loss'][-1]
            print(results.history['val_loss'][-1])

            # if validation loss is smaller then the best loss, the early stopping counting is reset
            if results.history['val_loss'][-1] < best_loss:
                best_loss = results.history['val_loss'][-1]
                count = 0
                count2 = 0
            # if not count is increased by one and if equal to 20 the training is stopped
            else:
                count = count + 1
                count2 = count2 + 1

                if count2 == 5:
                    model.optimizer.learning_rate = model.optimizer.learning_rate / 2
                    count2 = 0
                if count == 50:
                    break

            avg_time_epoch = (time.time() - start)
            sys.stdout.write(f" Average time/epoch {'{:.3f}'.format(avg_time_epoch / 60)} min")
            sys.stdout.write("\n")

        # write and save results
        writeResults(results, epochs, batch_size, learning_rate, model_save_dir,
                     save_folder, epochs)

        # plot the training and validation loss for all the training
        loss_training = np.array(loss_training)
        loss_val = np.array(loss_val)
        plotTraining(loss_training, loss_val, model_save_dir, save_folder, str(epochs))

        print("Training done")

    avg_time_epoch = (time.time() - global_start)
    sys.stdout.write(f" Average time training{'{:.3f}'.format(avg_time_epoch / 60)} min")
    sys.stdout.write("\n")
    sys.stdout.flush()

    # load the best weights of the model
    weights_path = os.path.join(ckpt_dir, "best.weights.h5")
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")
    else:
        # if no weights are found,there is something wrong
        print("Something is wrong.")

    # reset the states before predicting
    predictions = model.predict(test_gen, verbose=0)

    # predict the test set
    # if batch_size > 1:
    #     predictions = predictions.reshape(-1, mini_batch_size)
    #     preds = []
    #     for j in range(batch_size):
    #         pred = np.concatenate((predictions[0+j], predictions[batch_size+j]), axis=0)
    #         for i in range(2, predictions.shape[0]//batch_size):
    #             pred = np.concatenate((pred, predictions[i*batch_size+j]), axis=0)
    #         preds.append(pred)
    #
    #     predictions = np.array(preds)
    #     y_test = test_gen.y[:, :len(predictions[0])]
    #     x_test = test_gen.x[:, :len(predictions[0])]
    #
    #     y_test = np.array(y_test.reshape(-1), dtype=np.float32)
    #     x_test = np.array((x_test.reshape(-1)), dtype=np.float32)
    #     predictions = np.array((predictions.reshape(-1)), dtype=np.float32)

    #else:
    x_test, y_test = test_gen.getXY()
    predictions = predictions.reshape(-1)
    y_test = y_test.reshape(-1)
    x_test = x_test.reshape(-1)

    predictions = np.array(predictions, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)

    predictWaves(predictions, x_test, y_test, model_save_dir, save_folder, fs, '0')

    # compute the metrics: mse, mae, esr and rmse
    mse = tf.get_static_value(tf.keras.losses.MeanSquaredError()(y_test, predictions))
    mae = tf.get_static_value(tf.keras.losses.MeanAbsoluteError()(y_test, predictions))
    esr = tf.get_static_value(ESR(y_test, predictions))
    rmse = tf.get_static_value(RMSE(y_test, predictions))

    # writhe and store the metrics values
    results_ = {'mse': mse, 'mae': mae, 'esr': esr, 'rmse': rmse}
    with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
        for key, value in results_.items():
            print('\n', key, '  : ', value, file=f)

    return 42
