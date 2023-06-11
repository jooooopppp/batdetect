# utils/preprocessing.py
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import numpy as np

def read_wav_file(path):
    samplerate, data = wavfile.read(path)
    return samplerate, data / max(abs(data))

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def transform_data(filtered_data):
    # Normalize the filtered data
    normalized_data = filtered_data / np.max(np.abs(filtered_data))

    # Reshape the data to match the expected input shape for the model
    reshaped_data = normalized_data.reshape(1, -1)  # Assuming a single-channel input

    return reshaped_data