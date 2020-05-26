import os
import numpy as np
import pandas as pd
import sys
import time
from scipy.spatial.distance import cdist, euclidean, cosine
from glob import glob
from collections import defaultdict

import sigproc

from save import load, dump

sys.path.append("..")
from buffer import AudioBuffer

from model import vggvox_model
from wav_reader import get_fft_spectrum, remove_dc_and_dither, normalize_frames
import constants as c


def build_buckets(max_sec, step_sec, frame_step):
	buckets = {}
	frames_per_sec = int(1/frame_step) # 100
	end_frame = int(max_sec*frames_per_sec) # 1000
	step_frame = int(step_sec*frames_per_sec) # 100
	for i in range(0, end_frame+1, step_frame): 
		s = i
		s = np.floor((s-7+2)/2) + 1  # conv1
		s = np.floor((s-3)/2) + 1  # mpool1
		s = np.floor((s-5+2)/2) + 1  # conv2
		s = np.floor((s-3)/2) + 1  # mpool2
		s = np.floor((s-3+2)/1) + 1  # conv3
		s = np.floor((s-3+2)/1) + 1  # conv4
		s = np.floor((s-3+2)/1) + 1  # conv5
		s = np.floor((s-3)/2) + 1  # mpool5
		s = np.floor((s-1)/1) + 1  # fc6
		if s > 0:
			buckets[i] = int(s)
	return buckets


# def get_embedding(model, wav_file, max_sec):
# 	buckets = build_buckets(max_sec, c.BUCKET_STEP, c.FRAME_STEP)
# 	signal = get_fft_spectrum(wav_file, buckets)
# 	embedding = np.squeeze(model.predict(signal.reshape(1,*signal.shape,1)))
# 	return embedding


# def get_embedding_batch(model, wav_files, max_sec):
# 	return [ get_embedding(model, wav_file, max_sec) for wav_file in wav_files ]


def get_embeddings_from_list_file(model, list_file, max_sec):
	buckets = build_buckets(max_sec, c.BUCKET_STEP, c.FRAME_STEP)
	result = pd.read_csv(list_file, delimiter=",")
	result['features'] = result['filename'].apply(lambda x: get_fft_spectrum(x, buckets)) # Get as much fft as possible from the buckets 'size' for each filename
	result['embedding'] = result['features'].apply(lambda x: np.squeeze(model.predict(x.reshape(1,*x.shape,1)))) # Squeeze function removes all single-dimensionnal entries from the shape of an array
	return result[['filename','speaker','embedding']]


def get_id_result():
	print("Loading model weights from [{}]....".format(c.WEIGHTS_FILE))
	model = vggvox_model() # Creates a VGGVox model
	model.load_weights(c.WEIGHTS_FILE) # Load the weights of the trained models
	model.summary() # Print a summary of the loaded model

	print("Processing enroll samples....")
	enroll_result = get_embeddings_from_list_file(model, c.ENROLL_LIST_FILE, c.MAX_SEC) # Extracts information from fft using the VGGVox model
	enroll_embs = np.array([emb.tolist() for emb in enroll_result['embedding']])
	speakers = enroll_result['speaker']

	toSave = defaultdict(list)
	for i in range(len(speakers)):
		toSave[speakers[i]].append(enroll_embs[i])
	dump(toSave, "data/model/RTSP_CNN.out")

	start_time = time.time()
	print("Processing test samples....")
	test_result = get_embeddings_from_list_file(model, c.TEST_LIST_FILE, c.MAX_SEC)
	test_embs = np.array([emb.tolist() for emb in test_result['embedding']])

	print("Comparing test samples against enroll samples....")
	distances = pd.DataFrame(cdist(test_embs, enroll_embs, metric=c.COST_METRIC), columns=speakers) # Compute the distance between each test and enroll data

	scores = pd.read_csv(c.TEST_LIST_FILE, delimiter=",",header=0,names=['test_file','test_speaker'])
	scores = pd.concat([scores, distances],axis=1)
	scores['result'] = scores[speakers].idxmin(axis=1)

	print(time.time() - start_time, " seconds")

	index = scores[speakers].index
	result = scores[speakers].idxmin(axis=1)
	for idx in index:
		if(min(scores[speakers].values[idx]) > 0.16):
			result[idx] = "Unknown"

	scores['result_threshold'] = result

	scores['correct'] = (scores['result'] == scores['test_speaker'])*1. # bool to int
	scores['correct_threshold'] = (scores['result_threshold'] == scores['test_speaker'])*1. # bool to int

	# print("Writing outputs to [{}]....".format(c.RESULT_FILE))
	# result_dir = os.path.dirname(c.RESULT_FILE)
	# if not os.path.exists(result_dir):
	#     os.makedirs(result_dir)
	# with open(c.RESULT_FILE, 'w', newline="") as f:
	# 	scores.to_csv(f, index=False)

def RT_CNN():
	print("Loading model weights from [{}]....".format(c.WEIGHTS_FILE))
	model = vggvox_model() # Creates a VGGVox model
	model.load_weights(c.WEIGHTS_FILE) # Load the weights of the trained models
	model.summary() # Print a summary of the loaded model

	print("Loading embeddings from enroll")
	toLoad = load("data/model/RTSP_CNN.out")
	enroll_embs = []
	speakers = []
	for spk, embs in toLoad.items():
		for e in embs:
			enroll_embs.append(e)
			speakers.append(spk)
		print(spk)

	count = 0
	buffer = AudioBuffer()
	
	start_time = time.time()
	while count < 3:
		count += 1
		buffer.record(chunk_size = c.SAMPLE_RATE)
		data = buffer.get_data()
		data = np.frombuffer(data, 'int16')
	buckets = build_buckets(c.MAX_SEC, c.BUCKET_STEP, c.FRAME_STEP)

	data *= 2**15

	while (len(data)/(c.FRAME_STEP*c.SAMPLE_RATE) < 101):
		data = np.append(data, 0)

	# get FFT spectrum
	data = remove_dc_and_dither(data, c.SAMPLE_RATE)
	data = sigproc.preemphasis(data, coeff=c.PREEMPHASIS_ALPHA)
	frames = sigproc.framesig(data, frame_len=c.FRAME_LEN*c.SAMPLE_RATE, frame_step=c.FRAME_STEP*c.SAMPLE_RATE, winfunc=np.hamming)
	fft = abs(np.fft.fft(frames,n=c.NUM_FFT))
	fft_norm = normalize_frames(fft.T)

	# truncate to max bucket sizes
	rsize = max(k for k in buckets if k <= len(fft_norm.T))
	rstart = int((len(fft_norm.T)-rsize)/2)
	x = fft_norm[:,rstart:rstart+rsize]

	test_embs = np.squeeze(model.predict(x.reshape(1,*x.shape,1)))
	distances = []
	
	for embs in enroll_embs:
		distances.append(euclidean(test_embs, embs))

	print(len(speakers))

	idx = np.argmin(distances)
	
	print(speakers[idx])
	print("Ok, ", time.time() - start_time - 3, " seconds")


if __name__ == '__main__':
	task = input("test or realTime ? ")
	if task == "test":
		get_id_result()
	elif task == "realTime":
		RT_CNN()
