import time
import pickle
import sys
from collections import defaultdict
from skgmm import GMMSet
import numpy as np

sys.path.append("..")

from utils import get_feature

class ModelInterface:
    """
    Model object to create and save models.
    """

    def __init__(self):
        self.features = defaultdict(list)
        self.dyn_threshold = defaultdict(list)
        self.gmmset = GMMSet()
        self.n_label = 0

    def enroll(self, label, fs, signal):
        """
        Enrolls 'signal' to the features list related to 'name'.

        Args:
            label (string): Label of the speaker
            fs (int): Sampling rate of the signal
            signal (np.array([], dtype = np.int16): Audio signal (processed or not) from which the features are extracted
        """
        feat = get_feature(fs, signal)
        self.features[label].extend(feat)

    def train(self):
        """
        Trains the model object and saves it into a GMMSet object.
        """
        self.gmmset = GMMSet()
        start_time = time.time()
        for name, feats in self.features.items():
            try:
                self.gmmset.fit_new(feats, name)
                self.n_label += 1
            except Exception:
                print("%s failed"%(name))
        print(time.time() - start_time, " seconds")

    def predict(self, fs, signal):
        """
        Returns a speaker label for the given 'signal'.

        Args:
            fs (int): Sampling rate of the signal
            signal (np.array([], dtype = np.int16): Audio signal (processed or not) from which the features are extracted

        Returns:
            (string, float): Association of the predicted name label and its score
        """
        try: 
            feat = get_feature(fs, signal)
        except Exception as e:
            print(e)
        return self.gmmset.predict_one(feat)
            
    def dump(self, fname):
        """
        Dumps the model object to file.

        Args:
            fname (int): Name of the model file
        """
        with open(fname, 'wb') as f:
            pickle.dump(self, f, -1)

    def dynamic_threshold(self, fs, signal):
        """
        Computes the dynamic threshold for each speaker in the current model object.
        Saves the score of the signal going through each GMM.

        Args:
            fs (int): Sampling rate
            signal (np.array([], dtype = np.int16): Audio signal (processed or not) from which the features are extracted
        """
        feat = get_feature(fs, signal)
        for name in self.features.keys():
            self.dyn_threshold[name].append(self.gmmset.threshold_score(name, feat))

    def dynamic_mean(self):
        """
        Computes and saves the mean of all scores for each speaker.
        """
        for name in self.features.keys():
            self.dyn_threshold[name] = np.mean(self.dyn_threshold[name])

    def get_n_label(self):
        """
        Returns the number of labels inside the model object.

        Returns:
            int: Number of labels
        """
        return self.n_label

    def get_dyn_threshold(self):
        """
        Returns the dynamic threshold array (one threshold per speaker).

        Returns:
            dict: Dictionary gathering every dynamic threshold
        """
        return self.dyn_threshold

    @staticmethod
    def load(fname):
        """
        Loads a model object from a dump file.

        Args:
            fname (string): Dump file of the model

        Returns:
            ModelInterface: Model object
        """
        with open(fname, 'rb') as f:
            R = pickle.load(f)
            return R