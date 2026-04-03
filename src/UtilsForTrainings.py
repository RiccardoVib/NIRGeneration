import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from scipy.io import wavfile
import torch


def write_results(results, epochs, batch_size, learning_rate, model_save_dir, save_folder, index):
    """
    Write training results and parameters to text and pickle files.
    """
    summary = {
        'Min_val_loss': np.min(results['val_loss']),
        'Min_train_loss': np.min(results['train_loss']),
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'Val_loss': results['val_loss'],
        'Train_loss': results['train_loss'],
        'epochs': epochs,
    }
    save_path_txt = os.path.join(model_save_dir, save_folder, f'results_{index}.txt')
    save_path_pkl = os.path.join(model_save_dir, save_folder, f'results_{index}.pkl')

    os.makedirs(os.path.dirname(save_path_txt), exist_ok=True)

    with open(save_path_txt, 'w') as f:
        for key, value in summary.items():
            f.write(f'{key} : {value}\n')

    with open(save_path_pkl, 'wb') as f:
        pickle.dump(summary, f)


def plot_result(pred, inp, tar, model_save_dir, save_folder, fs, filename):
    """
    Plot prediction, input, and target waveforms.
    """
    time = np.arange(len(inp)) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(time, inp, label='input', alpha=0.3)
    plt.plot(time, tar, label='target', alpha=0.9)
    plt.plot(time, pred, label='pred', alpha=0.7)
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Model Prediction vs Input/Target')
    out_path = os.path.join(model_save_dir, save_folder, f'plot{filename}.pdf')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_training(loss_training, loss_val, model_save_dir, save_folder, name):
    """
    Plot training and validation loss curves.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(loss_training, label='train')
    plt.plot(loss_val, label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend(loc='upper right')
    out_path = os.path.join(model_save_dir, save_folder, f'{name}loss.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def predict_waves(predictions, x_test, y_test, z_test, p, model_save_dir, save_folder, fs, filename):
    """
    Render prediction, target, and input as wav audio files and plot.
    """
    base_dir = os.path.join(model_save_dir, save_folder, 'WavPredictions')
    os.makedirs(base_dir, exist_ok=True)

    if p is not None:
        z_test = torch.cat([z_test, p], dim=-1)

    for prediction, y, x, z in zip(predictions, y_test, x_test, z_test):
        cond = '_'.join([str(c) for c in z])
        inp_name = f'{cond}_inp.wav'
        pred_name = f'{cond}_pred.wav'
        tar_name = f'{cond}_tar.wav'

        inp_dir = os.path.join(base_dir, inp_name)
        pred_dir = os.path.join(base_dir, pred_name)
        tar_dir = os.path.join(base_dir, tar_name)

        wavfile.write(inp_dir, fs, x.reshape(-1))
        wavfile.write(pred_dir, fs, prediction.reshape(-1))
        wavfile.write(tar_dir, fs, y.reshape(-1))

        plot_result(prediction.reshape(-1), x.reshape(-1), y.reshape(-1), model_save_dir, save_folder, fs, cond + filename)


