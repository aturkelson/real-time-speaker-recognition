import numpy as np
import wave
import time
import argparse
import os

from buffer import AudioBuffer

def get_args():
    desc = "Audio acquisition Command Line Tool"
    epilog = """
    Saves audio into .wav files, at the output path.
    Example :
        Record test.wav in "../TRAIN/person" directory as "take_0.wav" (each of the wav file lasts 2 seconds for a total of 10 seconds):
        ./audio-acquisition.py -u TRAIN -n person -f take -d 10 -c 2
    """
    parser = argparse.ArgumentParser(description=desc, epilog=epilog,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-o', '--output',
                        help='Output folder "../TRAIN/Users/person_speaking"', default='../TRAIN/Users/Default',
                        required=False)

    parser.add_argument('-f', '--file',
                        help='Name of the saved wav file', default='test',
                        required=False)

    parser.add_argument('-d', '--duration',
                        help='Duration of the wav file (in seconds)', default=5,
                        type=float, required=False)

    parser.add_argument('-c', '--chunk',
                        help='Chunk size (in seconds)', default=1,
                        type=float, required=False)

    return parser.parse_args()

def save(path, data, width, rate=16000):
    """
    Saves the data recorded in a .wav file
    
    Args:
        path (string): Path where the files are saved
        data (np.array([], dtype=int)): Data saved
        width (int): Number of bytes in the format used
        rate (int, optional): Sampling rate of the saved file. Defaults to 16000.
    """
    for i in range(len(data)):
        wf = wave.open(path + '_' + str(i) + '.wav', 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.writeframes(data[i])
        wf.close()

if __name__ == '__main__':
    global args
    args = get_args()


    # Check if the output exists, else creates it #

    if not(os.path.exists(path)):
        try:
            os.makedirs(path)
        except OSError:
            print ("Creation of the directory %s failed" %path)


    # Sets all the parameters

    path = path + "/" + str(args.file)
    start_time = time.time()
    duration = args.duration
    chunk_duration = args.chunk
    rate = 16000
    chunk_size = int(chunk_duration*rate)


    # Creates the buffer and starts the recoring #

    count = 0
    buffer = AudioBuffer()
    print("Say something to the microphone : ")
    while count < duration:
        buffer.record(chunk_size, real_time = False)
        count += chunk_duration
    

    # Stops the recording and saves it to 'path' #

    buffer.stop_record()
    final_time = time.time()
    save(path, buffer.get_data(), 2, rate)
    print("Done : ", final_time - start_time - duration)