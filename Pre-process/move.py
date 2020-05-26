import os
import glob
import itertools

src = "../Files/TIMIT_SR/TRAIN/*/*/SA*.*"
dst = "../Files/TIMIT_SR/TEST"

src = [os.path.expanduser(k) for k in src.strip().split()]
dirs = itertools.chain(*(glob.glob(d) for d in src))

for d in dirs:
    f = os.path.split(d)
    g = os.path.split(f[0])
    h = os.path.split(g[0])
    
    if not(os.path.exists(dst + "/" + h[1] + "/" + g [1])):
        os.makedirs(dst + "/" + h[1] + "/" + g [1])
    dest = dst + "/" + h[1] + "/" + g [1] + "/" + f[1]
    if os.path.exists(dest):
        continue
    
    print(dest, " moved !")
    os.rename(d, dest)
