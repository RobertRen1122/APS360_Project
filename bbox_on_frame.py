import pandas as pd
import os
import cv2

CSV_DIR = 'sample_coords.csv'
FRAMES_DIR = 'temp_frames_resized'
FRAMES_WITH_BB_DIR = 'resized_temp_frames_with_draw_bounding_box'
WIDTH = 1920
HEIGHT = 1080
RESIZED_DIM = 448
X_RATIO = RESIZED_DIM / WIDTH
Y_RATIO = RESIZED_DIM / HEIGHT

def parse_coords(csv_dir, frame_no):
    df = pd.read_csv(csv_dir)
    coord_list = []
    print(f'Frame {frame_no}:')
    bboxes_at_frame = df.loc[df['frame_no_cam'] == frame_no]
    for i in range(len(bboxes_at_frame)):
        x_top_left, y_top_left = int(list(bboxes_at_frame['x_top_left_BB'])[i] * X_RATIO), int(list(bboxes_at_frame['y_top_left_BB'])[i] * Y_RATIO)
        x_bottom_right, y_bottom_right = int(list(bboxes_at_frame['x_bottom_right_BB'])[i] * X_RATIO), int(list(bboxes_at_frame['y_bottom_right_BB'])[i] * Y_RATIO)
        person_id = list(bboxes_at_frame['person_id'])[i]
        # print(f'Coords set {i}: {x_top_left, y_top_left, x_bottom_right, y_bottom_right}')
        coord_list.append([x_top_left, y_top_left, x_bottom_right, y_bottom_right, person_id])
    print(f'There are {len(coord_list)} bounding boxes in frame {frame_no}')
    return coord_list

def draw_bounding_box(frame_no, coords):
    img_dir = os.path.join(FRAMES_DIR, str(frame_no)+'.jpg')
    img = cv2.imread(img_dir)
    for coord_list in coords:
        cv2.rectangle(img, (coord_list[0], coord_list[1]), (coord_list[2], coord_list[3]), (255, 0, 0), 2)
        cv2.putText(img, f'ID: {coord_list[4]}', org=(coord_list[0], coord_list[1]-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0))
    img_with_bb = os.path.join(FRAMES_WITH_BB_DIR, str(frame_no)+'_bb.jpg')
    cv2.imwrite(img_with_bb, img)
    print(f"Frame {frame_no} written...")

if __name__ == '__main__':
    df = pd.read_csv(CSV_DIR)
    frame_list = list(set(list(df['frame_no_cam'])))
    for frame in frame_list:
        coords_set = parse_coords(csv_dir=CSV_DIR, frame_no=frame)
        draw_bounding_box(frame_no=frame, coords=coords_set)