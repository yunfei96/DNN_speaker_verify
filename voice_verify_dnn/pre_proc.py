import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft, fftfreq, ifft
import numpy as np
import os
import re
from pydub import AudioSegment
low = 85*3
high = 7000*3

# This function can plot the sound into frequency domain and store the filtered data
def plt_freq(file):
    samplerate, data = wavfile.read(file)
    data = data[:3 * samplerate]  # use the first 3 second data
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

# Calculate and plot spectrogram for a wav audio file
def graph_spectrogram(wav_file):
    rate, data = wavfile.read(wav_file)
    data = data[:3 * rate]
    nfft = 200 # Length of each window segment
    fs = rate # Sampling frequencies
    noverlap = 150 # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
        #plt.show()
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
        #plt.show()
    return pxx.T


def build_train_data():
    subjects = os.listdir('./data')
    subjects.sort()
    data = []
    label = []
    c = 0
    # -------------------for each person--------------------
    for i in subjects:
        #-----------------------for different speech----------------
        if i != '.DS_Store':
            speech = os.listdir('./data/' + i)
            for j in speech:
                # -----------------------for different section------------------
                if j != '.DS_Store':
                    section = os.listdir('./data/' + i + '/' +j)
                    for k in section:
                        if k != '.DS_Store':
                            #set the data and label
                            name = './data/' + i + '/' +j+'/'+k
                            data.append(graph_spectrogram(name))
                            label.append(c)
                            print(c)
                            print(i+j+k)

        c = c+1
    return data,label

# This function build the test data
def build_test_data():
    subjects = os.listdir('./data')
    subjects.sort()
    data = []
    label = []
    c = 0
    # -------------------for each person--------------------
    for i in subjects:
        #-----------------------for different speech----------------
        if i != '.DS_Store':
            speech = os.listdir('./data_t/' + i)
            for j in speech:
                # -----------------------for different section------------------
                if j != '.DS_Store':
                    section = os.listdir('./data_t/' + i + '/' +j)
                    for k in section:
                        if k != '.DS_Store':
                            #set the data and label
                            name = './data_t/' + i + '/' +j+'/'+k
                            data.append(graph_spectrogram(name))
                            label.append(c)
                            print(c)
                            print(i+j+k)

        c = c+1
    return data,label



# uncomment to test these functions
#(data, label) = build_train_data()
#x = graph_spectrogram('00008.wav')
#print(x)