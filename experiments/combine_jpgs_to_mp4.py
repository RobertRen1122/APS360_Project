import cv2
import numpy as np
import glob, os

import time
import sys
from stat import S_ISREG, ST_CTIME, ST_MODE
# /Users/angelayang/APS360/Video Sampling/combine_jpgs_to_mp4.py

img_array = []

#Relative or absolute path to the directory
dir_path = sys.argv[1] if len(sys.argv) == 2 else r'.'

#all entries in the directory w/ stats
data = (os.path.join(dir_path, fn) for fn in os.listdir(dir_path))
data = ((os.stat(path), path) for path in data)

# regular files, insert creation date
data = ((stat[ST_CTIME], path)
           for stat, path in data if S_ISREG(stat[ST_MODE]))


for cdate, path in sorted(data):
    filename=os.path.basename(path)
    print(filename)
    if '.jpg' in filename:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        img_array.append(img)


out = cv2.VideoWriter('new_video.mp4',cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 20, (1920 ,1080))
for i in range(len(img_array)):
    out.write(img_array[i])

out.release()

