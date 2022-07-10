import os, cv2, itertools, math, time
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
# import tqdm

# ================== initialization ===================================
# testing params
cam_id = 5
sampling_rate = 5  # fps
clear = 5 #people with this numer of occluded joints will be removed 

# path is hard coded since the data is stored in my external drive :)
path = "C:\\Users\\Shadow\\Downloads\\MTA_ext_short\\MTA_ext_short\\train\\cam_{}".format(cam_id)
calibration_path = "C:\\Users\\Shadow\\Downloads\\MTA_ext_short_coords\\MTA_ext_short_coords\\train\\cam_{}".format(cam_id)
# get the csv bounding box data as well as the video data
csv_path = os.path.join(path, "coords_fib_cam_{}.csv".format(cam_id))
cal_csv_path = os.path.join(calibration_path, "clear_{}_coords_cam_{}.csv".format(clear, cam_id))
video_path = os.path.join(path, "cam_{}.mp4".format(cam_id))

# create dataframe from csv file and read video
start_time = time.time()
cam_coords = pd.read_csv(csv_path)
print("Bounding Box Data loading finished in %s seconds ---" % (time.time() - start_time))
cal_csv = pd.read_csv(cal_csv_path, nrows=1)
annotated_csv = pd.read_csv(cal_csv_path)
print("All position labelling data loading finished in %s seconds ---" % (time.time() - start_time))

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

dis = math.sqrt((x_cam - x_p) ** 2 + (y_cam - y_p) ** 2)
pixel_h_top = cam_coords[(cam_coords.frame_no_cam == 0) &
                         (cam_coords.person_id == index_p)]["y_top_left_BB"].to_list()[0]
pixel_h_bottom = cam_coords[(cam_coords.frame_no_cam == 0) &
                            (cam_coords.person_id == index_p)]["y_bottom_right_BB"].to_list()[0]
pixel_h = abs(pixel_h_top - pixel_h_bottom)

# calculate the focus length
focus_len = pixel_h * dis

# approximating the horizontal dis
cam_w = 50


# ================== helper fcn =======================================
def drawboundingbox(image, boundingbox):
    cv2.line(image, (boundingbox.xT, boundingbox.yT), (boundingbox.xT, boundingbox.yB), color, thickness=1)
    cv2.line(image, (boundingbox.xB, boundingbox.yT), (boundingbox.xB, boundingbox.yB), color, thickness=1)
    cv2.line(image, (boundingbox.xT, boundingbox.yB), (boundingbox.xB, boundingbox.yB), color, thickness=1)
    cv2.line(image, (boundingbox.xT, boundingbox.yT), (boundingbox.xB, boundingbox.yT), color, thickness=1)


# ================== BB class =========================================

class BB:
    xT = 0
    yT = 0
    xB = 0
    yB = 0
    xC = 0
    yC = 0
    id = -1

    def __init__(self, xt, yt, xb, yb, pid):
        self.xT = xt
        self.yT = yt
        self.xB = xb
        self.yB = yb
        self.xC = (xb + xt) // 2
        # self.yC = (yb + yt) // 2
        self.yC = yb
        self.id = pid


# ================== accuracy graph initialization ====================
observed_d = []
accurate_d = []
observed_w = []
accurate_w = []
# ================== creating video data ==============================

count = 0
cap = cv2.VideoCapture(video_path)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("total frame num: ", length)
while cap.isOpened():
    # to avoid bug for size of -1 at the last frame
    if count == length - 2:
        break

    ret, frame = cap.read()
    if count % sampling_rate == 0:
        bbox_list = []
        accurate_dict = {}
        xT = cam_coords[cam_coords.frame_no_cam == count]["x_top_left_BB"].to_list()
        yT = cam_coords[cam_coords.frame_no_cam == count]["y_top_left_BB"].to_list()
        xB = cam_coords[cam_coords.frame_no_cam == count]["x_bottom_right_BB"].to_list()
        yB = cam_coords[cam_coords.frame_no_cam == count]["y_bottom_right_BB"].to_list()
        p_ID = cam_coords[cam_coords.frame_no_cam == count]["person_id"].to_list()

        # for depth precision checking
        xP = annotated_csv[(annotated_csv.frame_no_cam == count) &
                           (annotated_csv.joint_type == 0)]["x_3D_person"].to_list()
        yP = annotated_csv[(annotated_csv.frame_no_cam == count) &
                           (annotated_csv.joint_type == 0)]["y_3D_person"].to_list()
        p_id_annotated = annotated_csv[(annotated_csv.frame_no_cam == count) &
                                       (annotated_csv.joint_type == 0)]["person_id"].to_list()

        for i in range(len(p_id_annotated)):
            accurate_dict[p_id_annotated[i]] = [xP[i], yP[i]]

        for i in range(len(xT)):
            bbox = BB(xT[i], yT[i], xB[i], yB[i], p_ID[i])
            drawboundingbox(frame, bbox)
            bbox_list.append(bbox)

        if len(bbox_list) > 1:
            tar_pair = list(itertools.combinations(bbox_list, 2))

            for i in range(len(tar_pair)):
                # check distance here, if too close draw the distance line

                try:
                    label_1 = accurate_dict[tar_pair[i][0].id]
                    label_2 = accurate_dict[tar_pair[i][1].id]
                except:
                    break

                # depth accuracy testing =====================================================
                ydis1 = focus_len / (tar_pair[i][0].yB - tar_pair[i][0].yT)
                ydis2 = focus_len / (tar_pair[i][1].yB - tar_pair[i][1].yT)
                # y bottom > top
                # x bottom > top
                # print(tar_pair[i][0].xB, tar_pair[i][0].xT)
                depth = abs(ydis1 - ydis2)
                actual_dis1 = math.sqrt((label_1[0] - x_cam) ** 2 + (label_1[1] - y_cam) ** 2)
                actual_dis2 = math.sqrt((label_2[0] - x_cam) ** 2 + (label_2[1] - y_cam) ** 2)

                observed_d.append(ydis1)
                observed_d.append(ydis2)
                accurate_d.append(actual_dis1)
                accurate_d.append(actual_dis2)

                # width accuracy testing ====================================================
                x_center = (1920+1)/2
                w_to_cam_1 = tar_pair[i][0].xC - x_center
                h_to_cam_1 = 1080 - tar_pair[i][0].yB
                w_to_cam_2 = tar_pair[i][1].xC - x_center
                h_to_cam_2 = 1080 - tar_pair[i][1].yB

                theta_1 = math.atan(h_to_cam_1/w_to_cam_1)
                theta_2 = math.atan(h_to_cam_2/w_to_cam_2)

                D = math.sqrt(ydis1**2+ydis2**2-2*ydis2*ydis1*math.cos(theta_1-theta_2))
                # print(w_to_cam_1, w_to_cam_2)
                # print(h_to_cam_1, h_to_cam_2)

                xdis = abs(tar_pair[i][0].xC - tar_pair[i][1].xC) * cam_w / 1920
                # actual_w = math.sqrt((label_1[0] - label_2[0]) ** 2 + (label_1[1] - label_2[1]) ** 2)

                dis_pair = math.sqrt(depth ** 2 + xdis ** 2)
                actual_w = math.sqrt((label_1[0] - label_2[0]) ** 2 + (label_1[1] - label_2[1]) ** 2)

                observed_w.append(D)
                accurate_w.append(actual_w)
                # print(dis_pair)
                # if dis_pair<10:
                #     cv2.line(frame, (tar_pair[i][0].xC, tar_pair[i][0].yC), (tar_pair[i][1].xC, tar_pair[i][1].yC),
                #              dis_color, thickness=1)
##                cv2.line(frame, (tar_pair[i][0].xC, tar_pair[i][0].yC), (tar_pair[i][1].xC, tar_pair[i][1].yC),
##                         dis_color, thickness=1)

        cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
    count += 1

cap.release()
cv2.destroyAllWindows()


# depth plot
plt.scatter(observed_d, accurate_d, color='r')
xpoints = np.array([0, 100])
ypoints = np.array([0, 100])
plt.plot(xpoints, ypoints)
plt.xlabel('Extrapolated depth')
plt.ylabel('Actual depth')

# width plot
plt.figure()
plt.scatter(observed_w, accurate_w, color='b')
plt.plot(xpoints, ypoints)
plt.xlabel('Distance approximation')
plt.ylabel('Actual distance')


plt.show()
