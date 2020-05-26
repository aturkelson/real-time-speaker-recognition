import pyaudio
import numpy as np
import wave
import time

g_DURATION = 5 #Seconds
g_CHUNK_DURATION = 1 #Seconds

FORMAT = pyaudio.paInt16
g_RATE = 16000
CHANNELS = 1
INDEX = 1

HT = 0 #NEW
BUFFER = [] #NEW

def record(duration, chunk_duration, rate):
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.
    """
    global HT, BUFFER
    chunk_size = int(chunk_duration*rate)
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=rate,
        input=True, input_device_index=INDEX,
        frames_per_buffer=chunk_size,   stream_callback = callback) #NEW

    print("Say a something to the microphone : ")
    
    while stream.is_active(): #NEW
        HT += chunk_duration
        time.sleep(chunk_duration) #NEW

    stream.stop_stream() #NEW
    stream.close() #NEW

    buffer = BUFFER #NEW
    sample_width = p.get_sample_size(FORMAT) #NEW
    
    return sample_width, buffer

def callback(in_data, frame_count, time_info, flag): #NEW
    global BUFFER, HT
    BUFFER.append(in_data)
    if HT >= 5 :
        HT = 0
        return(None, pyaudio.paComplete)
    return(None, pyaudio.paContinue)

def save(data, width, rate):
    for i in range(len(data)):
        wf = wave.open('../Files/test_' + str(i) + '.wav', 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.writeframes(data[i])
        wf.close()

if __name__ == '__main__':
    tmp = time.time()
    duration = g_DURATION; chunk_duration = g_CHUNK_DURATION; rate = g_RATE
    sample_width, audio = record(duration, chunk_duration, rate)
    fin = time.time()
    save(audio, sample_width, rate)
    print("done : ", fin-tmp-duration)