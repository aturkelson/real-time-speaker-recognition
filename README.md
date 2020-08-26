# Real-time Speaker Recognition
This repository contains algorithms for real-time speaker recognition applications. It is implemented using either _Gaussian Mixture Model_ or _Convolutional Neural Network_. For the _GMM_ part, a dynamic threshold can be used to improve the recognition efficiency, but sharply increases the training time.

## Usage (_GMM_)
Enroll wav files into a _model.out_ and then launch the python script _RTSP.py_:
```bash
cd ./GMM
python3 speaker_recognition.py -t enroll -i ./path/to/wav_files_folder/* -m ./your-output-models/model.out
python3 RTSP.py
```
A prediction is made every three seconds once the model is loaded, for 15 seconds in total. You can modify the duration by changing the _while_ loop, line 103 _(tmp < 5)_.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
