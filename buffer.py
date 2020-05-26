import numpy as np
import pyaudio
import struct

class AudioBuffer:
    """
    Buffer that saves audio stream. Does the recording, adds the data in a np.array of int.
    """

    def __init__(self):
        self.__buffer = np.array([], dtype=int)
        self.__stream = 0

    def record(self, chunk_size = 160, real_time = True):
        """
        Starts the audio stream as initialization and reads the data from the default microphone.
        If the record is for real time application, it uses a ring buffer instead of keeping the whole record.

        Args:
            chunk_size (int, optional): Size of the pyaudio buffer. Defaults to 160.
            real_time (bool, optional): Selection of real time option. Defaults to True.
        """
        if not(self.__stream):
            p = pyaudio.PyAudio()
            rformat=pyaudio.paInt16
            rchannels=1
            rindex=1
            self.__stream = p.open(format=rformat, channels=rchannels, rate=16000,
                    input=True, input_device_index=rindex,
                    frames_per_buffer=chunk_size)
        
        data = self.__stream.read(chunk_size)
        if not real_time :
            self.add_data(data)
        else :
            self.ring(data)

    def stop_record(self):
        """
        Closes the stream.
        """
        self.__stream.stop_stream()
        self.__stream.close()
        self.__stream = 0

    def add_data(self, data):
        """
        Add data for non real time application.

        Args:
            data (np.array([], dtype = int16)): Data gathered by the audio stream
        """
        self.__buffer = np.append(self.__buffer, data)

    def ring(self, data, buflen = 3):
        """
        Add data to a ring buffer of size ('buflen' * number of frames recorded).

        Args:
            data (np.array([], dtype = int16)): Data gathered by the audio stream
            buflen (int, optional): Size of the ring buffer. Defaults to 3.
        """
        if len(self.__buffer) >= buflen:
            self.__buffer[:buflen] = np.append(self.__buffer[1:], data)    
        else:
            self.__buffer = np.append(self.__buffer, data)

    def get_data(self):
        """
        Returns the values gathered in the buffer.

        Returns:
            np.array(): Values recorded in the buffer
        """
        return np.asarray(self.__buffer)
