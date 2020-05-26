import os
import csv

class Evaluation:
    """
    Evalution class that allows a follow up of the result obtained during the speaker-recognition process
    """

    def __init__(self):
        self.__True = [0, 0] # TP and TN
        self.__False = [0, 0] # FP and FN
        self.__total = 0
        self.__error = []
    
    def new(self, speaker, label, recognized):
        """
        Adds a new result by comparing the actual speaker's name vs the label obtained.
        'Recognized' stands for the confidence of the algorithm about the label it gave.
        Ex.: if the speaker's name is the same as the label, but 'recognized' is false, then it is False Negative.

        Args:
            speaker (string): True speaker label
            label (string): Label given as a result of the computation
            recognized (bool): Confidence of the algorithm about the label it gave (true for confident)
        """
        self.__total += 1
        success = (speaker == label)
        if (success and recognized):
            self.__True[0] += 1 # True Positive
        elif (not(success) and not(recognized)):
            self.__True[1] += 1 # True Negative
        elif (not(success) and recognized):
            self.__False[0] += 1 # False Positive
            self.__error.append([speaker, label])
        elif (success and not(recognized)):
            self.__False[1] += 1 # False Negative
            self.__error.append([speaker, label])

    def accuracy(self):
        """
        Returns the accuracy of the prediction.

        Returns:
            float: Accuracy of the prediction
        """
        return (self.__True[0]+self.__True[1]) / self.__total

    def save(self, directory, n_label, DBname, delta_time):
        """
        Saves the results in a .csv file. If the file already exists, the result is append at the end.

        Args:
            directory (path): Directory which contains the database used for the prediction
            n_label (int): Number of label inside the model
            DBname (string): Name of the database, used as the .csv name file
        """
        if not(os.path.exists("res/"+ DBname + ".csv")):
            with open("res/"+ DBname + ".csv", 'w', newline="") as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['directory', 'Number of Labels', 'Success Rate', 'True Postive / True Negative', 'False Postive / False Negative', 'Processing time'])
                writer.writerow([directory, n_label, str(self.accuracy()), str(self.__True), str(self.__False), str(delta_time)])
        else :
            with open("res/"+ DBname + ".csv", 'a+', newline="") as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
                writer.writerow([directory, n_label, str(self.accuracy()), str(self.__True), str(self.__False), str(delta_time)])