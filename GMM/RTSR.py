import numpy as np
import time
import os
import wave
import sys

sys.path.append("..")

from buffer import AudioBuffer
from utils import VAD_process
from interface import ModelInterface
from evaluation import Evaluation

def predict(signal, m, ev, speaker):
    """
    Predicts the speaker of a signal from the model's label list.

    Args:
        signal (np.array): MFCC features array
        m (ModelInterface): Model object
    """
    rate = 16000

    n_label = m.get_n_label()
    threshold = 1 / n_label

    signal = signal/max(abs(signal))

    label, score = m.predict(rate, VAD_process(signal))

    recog = score > threshold

    if not(recog): 
        print('Not recognize. (', label, 'Score->', score, ')')

    else :
        print(label, ', score->', score)

    ev.new(speaker, label, recog)

def save_RT(name, data, width, rate):
    """
    Saves the recorded audio stream for an unknown speaker.

    Args:
        name (string): Speaker name
        data (np.array): Audio stream data
        width (int): Number of bytes in the format used
        rate (int): Sampling rate
    """
    i = 0
    base = path = "../../Files/RT_DB/" + str(name)
    
    if not (os.path.exists(base)):
        os.makedirs(base)

    while True:
        path = base + "/fly_" + str(i) + ".wav"
        if os.path.exists(path) :
            i += 1
        else :
            wf = wave.open(path, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(width)
            wf.setframerate(rate)
            wf.writeframes(data)
            wf.close()
            break


if __name__ == "__main__":


    # Initialisation #

    count = 0
    tmp = 0
    sampling_rate = 16000
    stop = False
    buffer = AudioBuffer()
    ev = Evaluation()


    # Retrieving the .out model file #

    input_model = input("Enter relative path of the model (ex.: ./model/static/Personal_DB.out) : ")
    for i in range(0, 3, 1):
        if not(os.path.exists(input_model)):
           input_model = input("Model does not exists ! " + str(3 - i) + " attempts left. Enter relative path of the model (ex.: ./model/Pers_DB.out) : ") 
        else:
            break

    if os.path.exists(input_model):


        # Loading the model and starting the while loop #

        print("Model found ! Starting real-time recognition process")
        m = ModelInterface.load(input_model)
        speaker = input("Write the name of the speaker (for evaluation purposes) :")
        
        start_time = time.time()
        while tmp < 5:
            count += 1

            buffer.record(chunk_size = sampling_rate) # 1 second of record
            data = buffer.get_data()
            data = np.frombuffer(data, 'int16')

            # Predicting every 3 loop #
            # Recording at 16000 Hz as sampling rate, (1 * 3) sec as buffer size and converting data in int16 type #

            if count >= 3:
                predict(data, m, ev, speaker)

                # save_RT(speaker, data, width =2, rate=sampling_rate)

                count = 0
                tmp += 1
        
        
        print("Ok, ", time.time() - start_time - 15, " seconds")
        # Stops the recording and closes the audio stream #

        print('Accuracy : ', ev.accuracy(), '\n')
        ev.save("Real-Time_Speaker_Recognition", tmp, "RTSP/RTSP_"+speaker, (time.time() - start_time - 15))

        buffer.stop_record()