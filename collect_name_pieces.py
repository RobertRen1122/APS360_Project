import os
import cv2
import numpy as np

if __name__ == '__main__':
    geshi = '.png'
    framegta_list = []
    camid_list = []
    pid_list = []
    filenames = next(os.walk('.'))[2]
    filenames = [filename for filename in filenames if geshi in filename]
    i = 0
    for filename in filenames:
        name = filenames[i].split('.')[0]
        name = name.split('_')
        framegta_list.append(name[1])
        camid_list.append(name[3])
        pid_list.append(name[5])
        i = i+1
    print('framegta_list:\n', framegta_list)
    print('\ncamid_lis:\n', camid_list)
    print('\npid_list:\n', pid_list)

    k=input("\n\npress ENTER to exit..") 
