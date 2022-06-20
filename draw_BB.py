import os, cv2, itertools, math
import pandas as pd

# import numpy as np
# import tqdm

# ================== initialization ===================================
# testing params
cam_id = 1
sampling_rate = 5

# path is hard coded since the data is stored in my external drive :)
path = "D://university//aps//MTA_ext_short//MTA_ext_short//train//cam_{}".format(cam_id)
calibration_path = "D://university//aps//MTA_ext_short_coords//train//cam_{}".format(cam_id)
# get the csv bounding box data as well as the video data
csv_path = os.path.join(path, "coords_fib_cam_{}.csv".format(cam_id))
cal_csv_path = os.path.join(calibration_path, "coords_cam_{}.csv".format(cam_id))
video_path = os.path.join(path, "cam_{}.mp4".format(cam_id))

# create dataframe from csv file and read video
cam_coords = pd.read_csv(csv_path)
cal_csv = pd.read_csv(cal_csv_path, nrows=1)
video_capture = cv2.VideoCapture(video_path)

# BB default color
color = (0, 255, 0)

# dis line color
dis_color = (0, 0, 255)

# config the camera
x_cam = cal_csv.iloc[0]["x_3D_cam"]
y_cam = cal_csv.iloc[0]["y_3D_cam"]
z_cam = cal_csv.iloc[0]["z_3D_cam"]

index_p = cal_csv.iloc[0]["person_id"]
x_p = cal_csv.iloc[0]["x_3D_person"]
y_p = cal_csv.iloc[0]["y_3D_person"]

approx_h = 170
dis = math.sqrt((x_cam-x_p)**2 + (y_cam-y_p)**2)
pixel_h_top = cam_coords[cam_coords.frame_no_cam == 0][cam_coords.person_id == index_p]["y_top_left_BB"].to_list()[0]
pixel_h_bottom = cam_coords[cam_coords.frame_no_cam == 0][cam_coords.person_id == index_p]["y_bottom_right_BB"].to_list()[0]
pixel_h = abs(pixel_h_top-pixel_h_bottom)

# calculate the focus length
focus_len = pixel_h*dis/approx_h

#approximating the horizontal dis
cam_w = 50
# ================== helper fcn =======================================
def drawBoundingBox(image, BB):
    cv2.line(image, (BB.xT, BB.yT), (BB.xT, BB.yB), color, thickness=1)
    cv2.line(image, (BB.xB, BB.yT), (BB.xB, BB.yB), color, thickness=1)
    cv2.line(image, (BB.xT, BB.yB), (BB.xB, BB.yB), color, thickness=1)
    cv2.line(image, (BB.xT, BB.yT), (BB.xB, BB.yT), color, thickness=1)

# ================== BB class =========================================

class BB:
    xT = 0
    yT = 0
    xB = 0
    yB = 0
    xC = 0
    yC = 0

    def __init__(self, xt, yt, xb, yb):
        self.xT = xt
        self.yT = yt
        self.xB = xb
        self.yB = yb
        self.xC = (xb + xt) // 2
        # self.yC = (yb + yt) // 2
        self.yC = yb


# ================== creating video data ==============================

count = 0
cap = cv2.VideoCapture(video_path)
while cap.isOpened():

    ret, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print(count)
    if count % sampling_rate == 0:
        xT = cam_coords[cam_coords.frame_no_cam == count]["x_top_left_BB"].to_list()
        yT = cam_coords[cam_coords.frame_no_cam == count]["y_top_left_BB"].to_list()
        xB = cam_coords[cam_coords.frame_no_cam == count]["x_bottom_right_BB"].to_list()
        yB = cam_coords[cam_coords.frame_no_cam == count]["y_bottom_right_BB"].to_list()
        bbox_list = []
        for i in range(len(xT)):
            bbox = BB(xT[i], yT[i], xB[i], yB[i])
            drawBoundingBox(frame, bbox)
            bbox_list.append(bbox)

        if len(bbox_list) > 1:
            tar_pair = list(itertools.combinations(bbox_list, 2))

            for i in range(len(tar_pair)):

                # check distance here, if too close draw the distance line
                ydis1 = approx_h * focus_len / (tar_pair[i][0].yT - tar_pair[i][0].yB)
                ydis2 = approx_h * focus_len / (tar_pair[i][1].yT - tar_pair[i][1].yB)
                xdis = abs(tar_pair[i][0].xC-tar_pair[i][1].xC)*cam_w/1920
                dis_pair = math.sqrt(abs(ydis1-ydis2)**2+xdis**2)
                # print(dis_pair)
                # D’ = (W x F) / P
                # if dis_pair<10:
                #     cv2.line(frame, (tar_pair[i][0].xC, tar_pair[i][0].yC), (tar_pair[i][1].xC, tar_pair[i][1].yC),
                #              dis_color, thickness=1)
                cv2.line(frame, (tar_pair[i][0].xC, tar_pair[i][0].yC), (tar_pair[i][1].xC, tar_pair[i][1].yC),
                         dis_color, thickness=1)

        cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    count += 1
cap.release()
cv2.destroyAllWindows()
