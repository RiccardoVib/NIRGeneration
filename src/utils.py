# Copyright (C) 2023 Riccardo Simionato, University of Oslo
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

import math
import numpy as np
import re
from scipy.io.wavfile import write
import os
import matplotlib.pyplot as plt
import json


def find_data_dir(target_folder='Mxr90', max_levels=5):
    """Search for target folder by going up directory levels"""
    for level in range(max_levels):
        prefix = '../' * level
        test_path = os.path.join(prefix, 'Files', target_folder)
        if os.path.exists(test_path):
            return os.path.abspath(test_path)
    return None

def save_audio_files(input_audio, output_audio, prediction_audio, model_path, prefix='0', sample_rate=48000):
    """
    Save audio files in WAV format.

    Parameters:
        input_audio (np.ndarray): Input audio data array.
        output_audio (np.ndarray): Output audio data array (processed).
        prediction_audio: Predicted labels or values (could be additional info to save).
        model_path (str): The path where to save the audio files (should exist).
    """
    # Create the model path directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)

    # Saving input audio
    input_file_path = os.path.join(model_path, prefix + '_input_audio.wav')
    input_audio = np.array(input_audio.squeeze(), dtype=np.float32)
    write(input_file_path, sample_rate, input_audio)  # Scale to int16

    # Saving output audio
    output_file_path = os.path.join(model_path, prefix + '_output_audio.wav')
    output_audio = np.array(output_audio.squeeze(), dtype=np.float32)
    write(output_file_path, sample_rate, output_audio)  # Scale to int16

    # Saving output audio
    output_file_path = os.path.join(model_path, prefix + '_prediction_audio.wav')
    prediction_audio = np.array(prediction_audio.squeeze(), dtype=np.float32)
    write(output_file_path, sample_rate, prediction_audio)  # Scale to int16

    plot(input_audio, output_audio, prediction_audio, model_path, prefix=prefix)

    print(f"\nAudio files saved to {model_path}")

def plot(input_audio, output_audio, prediction_audio, model_path, prefix='0'):

    plt.figure(figsize=(10, 6))
    plt.plot(input_audio, 'b-', label='input_audio')
    plt.plot(output_audio, 'r-', label='output_audio')
    plt.plot(prediction_audio, 'g-', label='prediction_audio')
    plt.title(' ', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Amplitude', fontsize=14)
    plt.legend(fontsize=12)

    filename = prefix + '_plot.png'
    # Save plot
    plt.savefig(model_path/filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {filename}")


# Function to save losses to file
def save_losses(train_losses, val_losses, filename='losses.json'):
    losses_dict = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    with open(filename, 'w') as f:
        json.dump(losses_dict, f)
    print(f"Losses saved to {filename}")

# Function to plot losses
def plot_losses(train_losses, val_losses, filename='loss_plot.png'):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss Over Time', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {filename}")


def natural_sort_key(s):
    """
    Function to use as a key for sorting strings in natural order.
    This ensures that strings with numbers are sorted in human-expected order.
    For example: ["file1", "file10", "file2"] -> ["file1", "file2", "file10"]

    Args:
        s: The string to convert to a natural sort key

    Returns:
        A list of string and integer parts that can be used for natural sorting
    """
    # Split the string into text and numeric parts
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]



def compute_lcm(x, y):
    """Compute the least common multiple of two numbers."""
    return (x * y) // math.gcd(x, y)


import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from librosa import display
import librosa.display
from scipy import fft
from scipy.signal import butter, lfilter

def loadFilePickle(data_dir, filename):
    """
    Initializes a data generator object
      :param data_dir: the directory in which data are stored
      :param output_size: output size
      :param batch_size: The size of each batch returned by __getitem__
    """
    file_data = open(os.path.normpath('/'.join([data_dir, filename])), 'rb')
    Z = pickle.load(file_data)
    return Z

def plotTime(x, fs):
    """
    Initializes a data generator object
      :param data_dir: the directory in which data are stored
      :param output_size: output size
      :param batch_size: The size of each batch returned by __getitem__
    """
    fig, ax = plt.subplots(nrows=1, ncols=1)
    display.waveshow(x, sr=fs, ax=ax)

def plotFreq(x, fs, N):
    """
    Initializes a data generator object
      :param data_dir: the directory in which data are stored
      :param output_size: output size
      :param batch_size: The size of each batch returned by __getitem__
    """

    FFT = np.abs(fft.fftshift(fft.fft(x, n=N))[N // 2:])/len(N)
    freqs = fft.fftshift(fft.fftfreq(N) * fs)
    freqs = freqs[N // 2:]

    fig, ax = plt.subplots(1, 1)
    ax.semilogx(freqs, 20 * np.log10(np.abs(FFT)+1))
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Magnitude (dB)')
    ax.axis(xmin=20, xmax=22050)

def plotSpectogram(x, fs, N):
    """
    Initializes a data generator object
      :param data_dir: the directory in which data are stored
      :param output_size: output size
      :param batch_size: The size of each batch returned by __getitem__
    """

    D = librosa.stft(x, n_fft=N)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    librosa.display.specshow(S_db, sr=fs, y_axis='linear', x_axis='time', ax=ax[0])
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.label_outer()

def butter_lowpass(cutoff, fs, order=2):
    """
    Initializes a data generator object
      :param data_dir: the directory in which data are stored
      :param output_size: output size
      :param batch_size: The size of each batch returned by __getitem__
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_highpass(cutoff, fs, order=2):
    """
    Initializes a data generator object
      :param data_dir: the directory in which data are stored
      :param output_size: output size
      :param batch_size: The size of each batch returned by __getitem__
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def filterAudio(x, fs, f_min, f_max):
    """
    Initializes a data generator object
      :param data_dir: the directory in which data are stored
      :param output_size: output size
      :param batch_size: The size of each batch returned by __getitem__
    """
    [b, a] = butter_highpass(f1, fs, order=2)
    [b2, a2] = butter_lowpass(f2, fs, order=2)
    x = lfilter(b, a, x)
    x = lfilter(b2, a2, x)
    return x
