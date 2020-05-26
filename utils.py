from scipy.io import wavfile
from python_speech_features import mfcc, sigproc
from vad import VoiceActivityDetection
import numpy as np
import sys

def get_feature(fs, signal):
    """
    Extracts 13 MFCC features from 'signal' with a window length of 0.03 seconds and a np.hamming window.

    Args:
        fs (int): Sampling rate
        signal (np.array([], dtype=np.int16)): Signal that MFCC are extracted from

    Returns:
        np.array([len(signal) / winlen]): Contains rows of features, each row holds 13 MFCC features.
    """
    mfcc_feature = mfcc(signal, fs, winlen=0.03, winfunc=np.hamming)
    if len(mfcc_feature) == 0 :
        print(sys.stderr, "ERROR.. failed to extract mfcc feature:", len(signal))
    return mfcc_feature

def read_wav(fname):
    """
    Reads the given 'fname' .wav file. Does the conversion from stereo to mono.

    Args:
        fname (string): Path to the .wav file

    Returns:
        int, np.array: Sample rate, Data read from .wav file
    """
    fs, signal = wavfile.read(fname)
    if len(signal.shape) != 1:
        print("Convert stereo to mono")
        signal = signal[:,0]
    return fs, signal

def VAD_process(signal):
    """
    Removes the silence from the audio 'signal'.

    Args:
        signal (np.array): Audio data to remove silence from

    Returns:
        np.array([], dtype=np.int16): 'Silence free' audio data
    """
    vad = VoiceActivityDetection()
    vad.process(signal)
    return vad.get_voice_samples()
