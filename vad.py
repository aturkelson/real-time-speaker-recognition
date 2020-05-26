import numpy as np

class VoiceActivityDetection:
    """
    Voice Activity Detection, a silence removal simple process.
    """

    def __init__(self):
        self.__step = 160 # = 16000 Hz * 0.01 ms
        self.__buffer_size = 160 # = 16000 Hz * 0.01 ms
        self.__buffer = np.array([], dtype=np.int16)
        self.__out_buffer = np.array([], dtype=np.int16)
        self.__VADthd = 0
        self.__VADn = 0
        self.__silence_counter = 0

    def vad(self, _frame):
        """
        Removes the silence by computing the power of the '_frame', setting a threshold from the minimal value to 10% of the peak-to-peak value.
        Updates the adaptative '__VADthd' threshold such as it takes into account (number of frame read previously * actual adaptative threshold).
        At every new call of this function, the threshold is adapted with the new threshold computed (ex.: lower __VADthd if lower threshold).
        If the mean of the current frame power if less than the value of the adaptative threshold, it is counted as a silent frame else the counter of silence is reset.
        If the counter of silence goes up to 20 or if (there is less than 50 frames since the beginning AND the current frame ptp value is less than 1e-5),
        the '_frame' is set as silent and remove from the final array.

        Args:
            _frame (np.array([], dtype = uint16)): Raw audio array

        Returns:
            bool: True if the _frame is not silent, false otherwise
        """
        frame = np.array(_frame)**2
        isVoiced = True
        threshold = np.min(frame) + np.ptp(frame) * 0.1
        self.__VADthd = (self.__VADn * self.__VADthd + threshold) / float(self.__VADn + 1)
        self.__VADn += 1

        if np.mean(frame) <= self.__VADthd:
            self.__silence_counter += 1
        else :
            self.__silence_counter = 0

        if (self.__silence_counter > 20) or (self.__VADn < 50 and np.ptp(frame) < 1e-5):    #if peak-to-peak < 1e-7 at the beginning, it is removed 
            #(50__VADn corresponds to 16,000 samples/s * 0.01 s since we slide 10 ms by 10 ms)
            isVoiced = False
        
        return isVoiced

    def add_samples(self, data):
        """
        Adds samples the the input buffer.

        Args:
            data (np.array([], dtype = int16)): Audio array

        Returns:
            bool: True if the size the input buffer is higher than the max buffer size
        """
        self.__buffer = np.append(self.__buffer, data)
        isBufferFull = len(self.__buffer) >= self.__buffer_size
        return isBufferFull

    def get_frame(self):
        """
        Returns the data inside the input buffer.

        Returns:
            np.array([buffer_size], dtype=int16): Data gathered inside the input buffer
        """
        window = self.__buffer[:self.__buffer_size]
        self.__buffer = self.__buffer[self.__step:]
        return window

    def process(self, data):
        """
        Main function of the VAD object. Adds 'data' to the input buffer, applies the VAD on it when full and appends it to the ouput buffer.

        Args:
            data (np.array([], dtype=int16)): Raw audio data array
        """
        if self.add_samples(data):
            while len(self.__buffer) >= self.__buffer_size:
                window = self.get_frame()
                if self.vad(window):
                    self.__out_buffer = np.append(self.__out_buffer, window)

    def get_voice_samples(self):
        """
        Returns the output buffer.

        Returns:
            np.array([], dtype=int16): Output buffer
        """
        return self.__out_buffer