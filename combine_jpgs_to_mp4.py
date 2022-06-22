import cv2
import numpy as np
import glob, os

# /Users/angelayang/APS360/Video Sampling/combine_jpgs_to_mp4.py

img_array = []
path = "/Users/angelayang/APS360/Video Sampling/"
# arr = os.listdir("/Users/angelayang/APS360/Video Sampling/")
# print(arr)

filenames = next(os.walk("/Users/angelayang/APS360/Video Sampling/"), (None, None, []))[2]
filenames = [filename for filename in filenames if '.jpg' in filename]
print(len(filenames))

for filename in filenames:
    # print(filename)

    img = cv2.imread(path+filename)
    height, width, layers = img.shape
    size = (width,height)
    print(width, height)

    img_array.append(img)

out = cv2.VideoWriter('new_video.mp4',cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 20, (1920 ,1080))
for i in range(len(img_array)):
    out.write(img_array[i])

out.release()

