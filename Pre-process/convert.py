from sphfile import SPHFile
import os
import glob
import itertools

dialects_path = "./TIMIT/TEST/*"

dialects_path = [os.path.expanduser(k) for k in dialects_path.strip().split()]
dirs = itertools.chain(*(glob.glob(d) for d in dialects_path))
dirs = [d for d in dirs if os.path.isdir(d)]

print(dirs)

for dialect in dirs:
    print(dialect)
    speakers = os.listdir(path = dialect)
    for speaker in speakers:
        speaker_path =  os.path.join(dialect,speaker)        
        speaker_recordings = os.listdir(path = speaker_path)

        wav_files = glob.glob(speaker_path + '/*.WAV')

        for wav_file in wav_files:
            sph = SPHFile(wav_file)
            txt_file = ""
            txt_file = wav_file[:-3] + "TXT"

            f = open(txt_file,'r')
            for line in f:
                words = line.split(" ")
                start_time = (int(words[0])/16000)
                end_time = (int(words[1])/16000)
            print("writing file ", wav_file)
            sph.write_wav(wav_file.replace(".WAV",".wav"),start_time,end_time)