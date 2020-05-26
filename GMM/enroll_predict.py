import os

task = input("Which task ? e for enroll, p for predict : ")

for i in range(1, 2):
    if task == 'e':
        os.system("python speaker_recognition.py -t enroll -i ../../Files/TIMIT_SR/ENROLL/DR" + str(i) + "/* -m ./model/TIMIT/DR" + str(i) + ".out")
        # os.system("python speaker_recognition.py -t enroll -i ../../Files/VoxCeleb_SR/ENROLL/id100" + str(i) + "* -m ./model/static/VoxCeleb/vox1_model" + str(i) + ".out")
    elif task == 'p':
        os.system("python speaker_recognition.py -t predict -i ../../Files/TIMIT_SR/TEST/DR" + str(i) + "/*/*.wav -m ./model/TIMIT/DR" + str(i) + ".out")
        # os.system("python speaker_recognition.py -t predict -i ../../Files/VoxCeleb_SR/TEST/id100" + str(i) + "*/*/*.wav -m ./model/static/VoxCeleb/vox1_model" + str(i) + ".out")
    else:
        print("Task not found.")
        break
