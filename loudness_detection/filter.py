import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft, fftfreq, ifft
import numpy as np
import os
import re
low = 100*3
high = 5000*3

# This function can plot the sound into frequency domain and store the filtered data
def plt_freq(data, samplerate):
    samples = data.shape[0]
    # normalize the data
    m = max(data)
    data = [(ele / m) for ele in data]

    datafft = fft(data)
    # get the absolute value of real and complex component:
    fftabs = abs(datafft)
    freqs = fftfreq(samples, 1 / samplerate)
    plt.xlim( [10, samplerate/2] )
    plt.xscale( 'log' )
    plt.grid( True )
    plt.xlabel( 'Frequency (Hz)' )

    fftabs[0:low] = 0
    fftabs[high:] = 0
    datafft[0:low] = 0
    datafft[high:] = 0
    filtered_data = ifft(datafft).real
    m = max(filtered_data)
    filtered_data = [(ele / m) for ele in filtered_data]
    wavfile.write("filtered.wav", samplerate, np.asarray(filtered_data))
    plt.plot(freqs[:int(freqs.size/2)],fftabs[:int(freqs.size/2)])
    plt.show()
    return fftabs[:int(freqs.size / 2)]

# This function can extract the data bounded by high and low frequency
def get_freq(data):
    datafft = fft(data)
    # get the absolute value of real and complex component:
    fftabs = abs(datafft)
    return fftabs[low:high]

def output_freq(data):
    m = max(data)
    data = [(ele / m) for ele in data]
    datafft = fft(data)
    # get the absolute value of real and complex component:
    datafft[0:low] = 0
    datafft[high:] = 0
    filtered_data = ifft(datafft).real
    return np.asarray(filtered_data)