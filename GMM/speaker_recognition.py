#Not real-time speaker recognition -- Uses wav file to train and predict

import os
import sys
import itertools
import glob
import argparse
import time

sys.path.append('..')

from utils import VAD_process, read_wav, get_feature
from evaluation import Evaluation
from vad import VoiceActivityDetection
from interface import ModelInterface

def get_args():
    desc = "Speaker Recognition Command Line Tool"
    epilog = """
    Wav files in each input directory will be labeled as the basename of the directory.
    Note that wildcard inputs should be *quoted*, and they will be sent to glob.glob module.
    Examples:
        Train (enroll a list of person named person*, and mary, with wav files under corresponding directories):
        ./speaker-recognition.py -t enroll -i "/tmp/person* ./mary" -m model.out
        Predict (predict the speaker of all wav files):
        ./speaker-recognition.py -t predict -i "./*.wav" -m model.out
    """
    parser = argparse.ArgumentParser(description=desc, epilog=epilog,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-t', '--task',
                        help='Task to do. Either "enroll" or "predict"',
                        required=True)
    
    parser.add_argument('-i', '--input',
                        help='Input files (to predict) or Directories (to enroll)',
                        required=True)

    parser.add_argument('-m', '--model',
                        help='Model file to save (in enroll) or use (in predict)',
                        required=True)
    
    parser.add_argument('-d', '--dynamic',
                        help='Dynamic Threshold Mode. Either "true" or "false" (default). Warning : computation takes a lot of time ',
                        required=False, default=False)

    args = parser.parse_args()

    return args

def task_enroll(input_dirs, output_model, isDynamic): # Possible improvement : Store VAD signal of each .wav file instead of calculate it twice for dynamic threshold (useless for static)
    """
    Enroll the speaker inside the GMM model.
    
    Args:
        input_dirs (string): directory of the database
        output_model (string): path of the output model
    """

    # Creates the model object #

    m = ModelInterface()


    # Extracts the absolute path from 'input_dirs' #
    # If input_dirs is an array of directory, it is adapted by the second line #
    
    print(input_dirs)
    input_dirs = [os.path.expanduser(k) for k in input_dirs.strip().split()]
    dirs = itertools.chain(*(glob.glob(d) for d in input_dirs))
    dirs = [d for d in dirs if os.path.isdir(d)]

    if len(dirs) == 0:
        print('No valid directory found!')
        sys.exit(1)


    # Starts the enrollment of the valid directories #
    start_time = time.time()
    print('Starting enrollment')
    for d in dirs:
        print(d)
        # Retrieves the label of the current directory name and loads .wav files are stored #

        label = os.path.basename(d.rstrip('/'))
        print(label)
        wavs = glob.glob(d + '/*.wav')

        if len(wavs) == 0:
            print('No wav file found in %s'%(d))
            continue

        for wav in wavs:

            # Audio processing of the .wav file #
            # Retrieves sampling rate (fs), signal values #
            # VAD removes silence inside the signal #
            # Enrolls the cleared signal and its label inside the model #

            try:
                fs, signal = read_wav(wav)
                signal = signal/max(abs(signal))

                m.enroll(label, fs, VAD_process(signal))

            except Exception as e:
                print(wav + ' error %s'%(e))


    # Starts the training of the model using the enrolled signals #

    print('Enrollment finished\nTraining started')
    m.train()

    print('Training finished')
    

    # Starts a dynamic threshold computation #
    # /!\ Warning : computation takes a lot of time /!\ #

    if(isDynamic):
        print('Dynamic Threshold started')

        i = 0

        for d in dirs:

            # Loads the .wav files #
            # Each .wav file will be used to compute a score for each existing label after training #

            if len(wavs) == 0:
                print('No wav file found in %s'%(d))
                continue

            wavs = glob.glob(d + '/*.wav')
            for wav in wavs:
                try: 
                    fs, signal = read_wav(wav)
                    signal = signal/max(abs(signal))

                    m.dynamic_threshold(fs, VAD_process(signal))
            
                except Exception as e:
                    print(wav + ' error %s'%(e))

            i += 1

            for j in range(0, 100, 10):
                if(i == int(len(dirs) * j * 0.01)):
                    print('%i percent done.'%(j))
            

        # Keeps only the mean from all scores for a given label as dynamic threshold #

        try:
            m.dynamic_mean()
        except Exception as e:
            print('Error for dyanmic threshold : error %e'%(e))
        print('Dynamic threshold finished')

    print(time.time() - start_time, " seconds")
    
    # Saves the model at the specified path 'output_model' #

    m.dump(output_model)

def task_predict(input_files, input_model, isDynamic):
    """
    Predict the speaker from the given file(s)
    
    Args:
        input_files (string): full path to the speaker file
        input_model (string): model trained to give the solution
    """
    # Loads the model object and retrieve the number of speaker #

    m = ModelInterface.load(input_model)
    n_label = m.get_n_label()


    # Computes the threshold (dynamic or static) #

    if(isDynamic):
        dyn_thrsh = m.get_dyn_threshold()
    else:
        threshold = 1 / n_label


    # Creates an Evaluation object to save the results #

    ev = Evaluation()


    # Starts the prediction process #

    print(input_files)
    for f in glob.glob(os.path.expanduser(input_files)):
        try:
            start_time = time.time()
            fs, signal = read_wav(f)
            signal = signal/max(abs(signal))


            # Extracts the features and predicts the label using the higher score within all possible speaker #

            label, score = m.predict(fs, VAD_process(signal)) 

        except Exception as e:
            print(f + ' error %s'%(e))


        # Retrieves the expected label from the directory (evaluation not real time only) #

        root = os.path.split(f)
        if(input_files[-9:] == "*/*/*.wav"):
            root = os.path.split(root[0])
        speaker = os.path.basename(root[0])


        # Recognition process : If the given score is higher than the threshold, the label is correct #
        # Else the speaker is not recognize #

        if(isDynamic):
            threshold = dyn_thrsh[label]
        recog = (score > threshold)

        # recog = True

        if not(recog) : 
            print(speaker, ' not recognize. ->', label, 'Score->', score)

        else :
            print(speaker, '->', label, ', score->', score)
        

        # Adds the speaker and its results to the evaluation object #

        ev.new(speaker, label, recog)


    # Retrieves the Database label used and prints the accuracy #

    path = os.path.split(root[0])[0]
    DB_name = os.path.split(path)[0]
    DB_name = os.path.basename(os.path.split(DB_name)[1])
    print('Accuracy : ', ev.accuracy(), '\n')
    ev.save(os.path.basename(path), n_label, DB_name, (time.time() - start_time))

if __name__ == "__main__":
    global args
    args = get_args()

    task = args.task
    if task == 'enroll':
        task_enroll(args.input, args.model, args.dynamic)
    elif task == 'predict':
        task_predict(args.input, args.model, args.dynamic)