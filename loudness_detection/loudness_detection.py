'''

this program will check the loudness of each 0.5 second
if the loudness is larger than threshold, the 0.5 second will be append to a sentence
if there is 0.5 second margin, the sentence will be feed into a text detector

'''

import numpy as np
import pyaudio
from scipy.io import wavfile
from queue import Queue
from threading import Thread
import sys
import time
import os
import speech_recognition as sr
import wave
import filter
import subprocess


def say(text):
    subprocess.call(['say', text])


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
r = sr.Recognizer()

# ----------------------- config py audio streaming ------------------------------
chunk_duration = 0.5  # Each read length in seconds from mic.
fs = 44100  # sampling rate for mic
chunk_samples = int(fs * chunk_duration)  # Each read length in number of samples.


def get_audio_input_stream(callback):
    stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=1,
        rate=fs,
        input=True,
        frames_per_buffer=chunk_samples,
        input_device_index=0,
        stream_callback=callback)
    return stream


# Queue to communicate between the audio callback and main thread
# ---------------------- detection threshold --------------------------------------
silence_threshold = 20000
q = Queue()
start = False
end = False

data = np.zeros(22050, dtype='int16')


def callback(in_data, frame_count, time_info, status):
    global data, silence_threshold, start, end
    data0 = np.frombuffer(in_data, dtype='int16')
    loudness = filter.get_freq(data0)
    print(np.abs(loudness).mean())
    # smaller than threshold, ignore
    if np.abs(loudness).mean() < silence_threshold:
        # print('-', end = '', flush=True)
        if not start and not end:
            start = True
        elif start and end:
            q.put(data)
            print("s", flush=True)
            start = False
            end = False

        return (in_data, pyaudio.paContinue)
    else:
        # print('.', end = '',flush=True)
        if start:
            data = np.append(data, data0)
            end = True
        # data = np.append(data, data0)
        # q.put(data0)

    return (in_data, pyaudio.paContinue)


stream = get_audio_input_stream(callback)
stream.start_stream()

active = False

try:
    while 1:
        if not active:
            data = q.get()
            data = filter.output_freq(data)
            # wavfile.write("filtered.wav", fs, data)
            data = data * 32767
            data = data.astype(np.int16)
            # fs, data = wavfile.read("filtered.wav")
            # print(data.shape, flush=True)
            print("d", flush=True)
            audio = sr.AudioData(data.tobytes(), fs, 2)
            command = ""
            try:
                command = r.recognize_google(audio, key=None, language="en-US", show_all=False)
                print("command: " + command, flush=True)
            except sr.UnknownValueError:
                print("command: void", flush=True)
            if command == "active" or command == "activation":
                say("Hi, Yunfei")
                active = True
                stream.stop_stream()
        else:
            with sr.Microphone() as source:
                print("Please input command:")
                audio = r.listen(source)
            # recognize speech using google
            try:
                print("command is: " + r.recognize_google(audio, key=None, language="en-US", show_all=False))
            except sr.UnknownValueError:
                print("google could not understand audio")
            active = False
            stream.start_stream()



except (KeyboardInterrupt, SystemExit):
    stream.stop_stream()
    stream.close()
