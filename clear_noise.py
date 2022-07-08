import os, cv2, itertools, math, time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#clear = number of total occluded joints allowed
clear = 1
cam_id = 2


calibration_path = "C:\\Users\\Shadow\\Downloads\\MTA_ext_short_coords\\MTA_ext_short_coords\\train\\cam_{}".format(cam_id)
cal_csv_path = os.path.join(calibration_path, "coords_cam_{}.csv".format(cam_id))
annotated_csv = pd.read_csv(cal_csv_path)

#clear empty cells in dataframe
annotated_csv.replace('', np.nan, inplace=True)
annotated_csv.dropna(inplace=True)

#remove data of occluded people in dataframe 
#noise = number of occluded joints
def clear_noise():
    x = 0
    while(x < (len(annotated_csv)-22)):
        noise = 0
        for y in range(22):

            laji = annotated_csv.iloc[y+x]["joint_occluded"]
            if(annotated_csv.iloc[y+x]["joint_occluded"] == 1):
                noise += 1
        if(noise > clear):
            annotated_csv.drop(annotated_csv.index[x:21], axis=0, inplace=True)
            x -= 22
        x += 22
    annotated_csv.to_csv(os.path.join(calibration_path,"cleared_coords_cam_{}.csv".format(cam_id)))

clear_noise()
print("cleared_coords_cam_{}.csv".format(cam_id) + " file is generated")
