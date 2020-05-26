import glob
import os
import sys
import argparse
import csv

def get_args():
    desc = "Speaker Recognition Command Line Tool"
    epilog = """
    Create a CSV file regrouping all wav file inside a given folder.
    Example :
        ./WriteCSV.py -t enroll -f TIMIT_SR/DR1 -i ../../Files/TIMIT_SR/ENROLL/DR1/*/*.wav
        ./WriteCSV.py -i ../../Files/VoxCeleb_SR/ENROLL/id1001*/*/*/*.wav
    """
    parser = argparse.ArgumentParser(description=desc, epilog=epilog,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-t', '--task',
                        help='Task (enroll or test)',
                        required=True)

    parser.add_argument('-f', '--folder',
                        help='Folder',
                        required=True)

    parser.add_argument('-i', '--input',
                        help='Input files',
                        required=True)
    
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    global args
    args = get_args()

    folder = 'cfg/' + args.folder

    if not(os.path.exists(folder)):
        try:
            os.makedirs(folder)
        except OSError:
            print ("Creation of the directory %s failed" %folder)

    with open(folder + '/' + args.task + '_list.csv', 'w', newline="") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['filename', 'speaker'])
        for f in glob.glob(os.path.expanduser(args.input)):
            root = os.path.split(f)
            if(args.input[-9:] == "*/*/*.wav"):
                root = os.path.split(root[0])
            user = os.path.basename(root[0])
            f = f.replace("\\",'/')
            f = f.replace("../../Files/", "")
            data = [f, user]
            writer.writerow(data)
