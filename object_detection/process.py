# Working with cam_0
# Steps for preprocessing
# 1. Separate all frames into images, resized to 448 by 448, store to images folder
# 2. Work on coords_fib_cam_0.csv, turn into the form of [is_person, box_x_top_left,
#    box_y_top_left, box_x_bottom_right, box_y_bottom right]
# 3. Construct frame_num.jpg, frame_num.txt

import cv2
import pandas as pd

WIDTH = 1920
HEIGHT = 1080
RESIZED_DIM = 448
X_RATIO = RESIZED_DIM / WIDTH
Y_RATIO = RESIZED_DIM / HEIGHT

def vid_to_frame(vid_dir):
    vid = cv2.VideoCapture(vid_dir)
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    frame_num = 0
    while(True):
        success, frame = vid.read()
        if success:
            frame = cv2.resize(frame, (448, 448))
            cv2.imwrite(f'./images/{frame_num}.jpg', frame)
            print(f'Wrote {frame_num}...')
        else:
            break
        frame_num += 1
    print(f'{length} frames with a frame rate of {fps}')

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
        coord_list.append([x_top_left/448, y_top_left/448, x_bottom_right/448, y_bottom_right/448, person_id])
    print(f'There are {len(coord_list)} bounding boxes in frame {frame_no}')
    return coord_list


if __name__ == '__main__':
    vid_dir = 'raw_data/train/cam_0/cam_0.mp4'
    vid_to_frame(vid_dir=vid_dir)
    coords_dir = 'raw_data/train/cam_0/coords_fib_cam_0.csv'
    df = pd.read_csv(coords_dir)
    frame_list = list(set(list(df['frame_no_cam'])))
    for frame in frame_list:
        coords_set = parse_coords(csv_dir=coords_dir, frame_no=frame)
        with open(f'data/labels/{frame}.txt', 'w') as f:
            for coords in coords_set:
                coords = coords[:-1]
                coords.insert(0, 1)
                coords = [str(coord) for coord in coords]
                line = ' '.join(coords)
                f.write(line)
                f.write('\n')
                print(f"Wrote {frame}.txt")
            f.close()
