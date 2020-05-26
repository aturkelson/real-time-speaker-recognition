import librosa
import numpy as np
from scipy.signal import lfilter, butter

import sigproc
import constants as c


def load_wav(filename, sample_rate):
	audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
	audio = audio.flatten()
	return audio


def normalize_frames(m,epsilon=1e-12):
	return np.array([(v - np.mean(v)) / max(np.std(v),epsilon) for v in m])


# https://github.com/christianvazquez7/ivector/blob/master/MSRIT/rm_dc_n_dither.m
def remove_dc_and_dither(sin, sample_rate):
	if sample_rate == 16e3:
		alpha = 0.99
	elif sample_rate == 8e3:
		alpha = 0.999
	else:
		print("Sample rate must be 16kHz or 8kHz only")
		exit(1)
	sin = lfilter([1,-1], [1,-alpha], sin)
	dither = np.random.random_sample(len(sin)) + np.random.random_sample(len(sin)) - 1
	spow = np.std(dither)
	sout = sin + 1e-6 * spow * dither
	return sout


def get_fft_spectrum(filename, buckets):
	signal = load_wav(filename,c.SAMPLE_RATE)
	signal *= 2**15

	# print(filename)
	print(buckets)

	while (len(signal)/(c.FRAME_STEP*c.SAMPLE_RATE) < 101):
		signal = np.append(signal, 0)

	# get FFT spectrum
	signal = remove_dc_and_dither(signal, c.SAMPLE_RATE)
	signal = sigproc.preemphasis(signal, coeff=c.PREEMPHASIS_ALPHA)
	frames = sigproc.framesig(signal, frame_len=c.FRAME_LEN*c.SAMPLE_RATE, frame_step=c.FRAME_STEP*c.SAMPLE_RATE, winfunc=np.hamming)
	fft = abs(np.fft.fft(frames,n=c.NUM_FFT))
	# print(len(fft))
	fft_norm = normalize_frames(fft.T)
	# print(len(fft_norm.T))

	# truncate to max bucket sizes
	rsize = max(k for k in buckets if k <= len(fft_norm.T))
	# print(rsize)
	rstart = int((len(fft_norm.T)-rsize)/2)
	# print(rstart)
	out = fft_norm[:,rstart:rstart+rsize]
	# print(len(out))

	return out
