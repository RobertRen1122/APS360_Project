import cv2
import numpy as np
import glob

img_array = []
for filename in glob.glob('.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('new_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, (3584, 2016))

for i in range(len(img_array)):
    out.write(img_array[i])

out.release()
