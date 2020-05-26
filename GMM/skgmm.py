from sklearn.mixture import GaussianMixture
import operator
import numpy as np
import math

class GMMSet :
    """
    Gathers every GMM extracted from the database (one GMM per speaker)
    """
    def __init__(self, gmm_order = 32):
        self.gmms = []
        self.gmm_order = gmm_order
        self.y = []

    def fit_new(self, x, label):
        """
        Appends a new GMM to the gmmset, so that there is one GMM per label.
        Adapts the gmm parameters to fit the given features.

        Args:
            x (np.array): MFCC features array
            label (string): Speaker label
        """
        if self.y.count(label) == 0:
            self.y.append(label)
            gmm = GaussianMixture(self.gmm_order, random_state=2)
            gmm.fit(x)
            self.gmms.append(gmm)
        else :
            index = self.y.index(label)
            self.gmms[index].fit(x)

    def gmm_score(self, gmm, x):
        """
        Computes the score (log-likelihood).

        Args:
            gmm (GaussianMixture): GMM which will be used to compute the score from
            x (np.array): MFCC features array
        """
        return np.sum(gmm.score(x))
    
    @staticmethod
    def softmax(s, scores_sum):
        """
        Computes the softmax of a given array.

        Args:
            s (np.array): Score array
            scores_sum (float): Sum of all the scores in the gmmset

        Returns:
            float: Rounded float softmax
        """
        score_max = math.exp(max(s))
        return round(score_max / scores_sum, 5)

    def threshold_score(self, label, x):
        """
        Computes the softmax score for the dynamic threshold, given a features array.

        Args:
            label (string): Label of the speaker whose dynamic threshold is computed
            x (np.array): MFCC features array

        Returns:
            float: Rounded float softmax
        """
        scores = [self.gmm_score(gmm, x) / len(x) for gmm in self.gmms] # Normalized score per gmm (normalized by the size of the features array)
        scores_sum = sum([math.exp(i) for i in scores])
        index = self.y.index(label)
        score = math.exp(scores[index])
        return round(score / scores_sum, 5)

    def predict_one(self, x):
        """
        Computes the softmax score for the final prediction, given a features array.

        Args:
            x (np.array): MFCC features array

        Returns:
            string, float: Predicted label and its score
        """
        scores = [self.gmm_score(gmm, x) / len(x) for gmm in self.gmms] # Normalized score per gmm (normalized by the size of the features array)
        result = [(self.y[index], value) for (index, value) in enumerate(scores)]
        p = max(result, key=operator.itemgetter(1))
        scores_sum = sum([math.exp(i) for i in scores])
        softmax_score = self.softmax(scores, scores_sum)
        return p[0], softmax_score