import pandas as pd
import os
import cv2
from pathlib import Path

RAW_DIR = 'data/raw/cam_0'
IMAGE_DIR = 'data/images'
LABEL_DIR = 'data/labels'
WIDTH = 1920
HEIGHT = 1080
NUM_IMAGES = 4922


def coords_conversion(coords_list):
    # https://christianbernecker.medium.com/convert-bounding-boxes-from-coco-to-pascal-voc-to-yolo-and-back-660dc6178742
    x1, y1, x2, y2 = coords_list[0], coords_list[1], coords_list[2], coords_list[3]
    x_mid = (x2 + x1) / (2 * WIDTH)
    y_mid = (y2 + y1) / (2 * HEIGHT)
    width = (x2 - x1) / WIDTH
    height = (y2 - y1) / HEIGHT
    converted_list = [x_mid, y_mid, width, height]
    return converted_list


def vid_to_frame(vid_dir):
    vid = cv2.VideoCapture(vid_dir)
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    frame_num = 0
    while True:
        success, frame = vid.read()
        if success:
            img_dir = os.path.join(IMAGE_DIR, f'{frame_num}.jpg')
            cv2.imwrite(img_dir, frame)
            print(f'Wrote {frame_num} to {img_dir}')
        else:
            break
        frame_num += 1
    print(f'Obtained {length} frames with a frame rate of {fps}')


def create_train_test_csv(train_ratio=0.8):
    num_train_imgs = int(train_ratio * NUM_IMAGES)
    num_test_imgs = NUM_IMAGES - num_train_imgs


def parse_coords(csv_dir, frame_no):
    df = pd.read_csv(csv_dir)
    coord_list = []
    # print(f'Frame {frame_no}:')
    bboxes_at_frame = df.loc[df['frame_no_cam'] == frame_no]
    for i in range(len(bboxes_at_frame)):
        x_top_left, y_top_left = int(list(bboxes_at_frame['x_top_left_BB'])[i]), int(
            list(bboxes_at_frame['y_top_left_BB'])[i])
        x_bottom_right, y_bottom_right = int(list(bboxes_at_frame['x_bottom_right_BB'])[i]), int(
            list(bboxes_at_frame['y_bottom_right_BB'])[i])
        # print(f'Coords set {i}: {x_top_left, y_top_left, x_bottom_right, y_bottom_right}')
        coordinates = [x_top_left, y_top_left, x_bottom_right, y_bottom_right]
        scaled_coordinates = coords_conversion(coordinates)
        coord_list.append(scaled_coordinates)
    # print(f'There are {len(coord_list)} bounding boxes in frame {frame_no}')
    return coord_list


def generate():
    image_exists = os.path.exists(IMAGE_DIR)
    label_exists = os.path.exists(LABEL_DIR)
    if not image_exists:
        os.makedirs(IMAGE_DIR)
    if not label_exists:
        os.makedirs(LABEL_DIR)

    vid_file, csv_file = '', ''
    for file in os.listdir(RAW_DIR):
        if file.endswith('.mp4'):
            vid_file = os.path.join(RAW_DIR, file)
        if file.endswith('.csv'):
            csv_file = os.path.join(RAW_DIR, file)

    vid_to_frame(vid_dir=vid_file)
    for i in range(4922):
        coords = parse_coords(csv_file, i)
        filename = os.path.join(LABEL_DIR, f'{i}.txt')
        num_lines = 0
        with open(filename, 'w') as f:
            for coords_set in coords:
                coords_set.insert(0, 1)
                coords_string = ' '.join(str(x) for x in coords_set)
                num_lines += 1
                f.write(coords_string + '\n')
        print(f'Finished writing {num_lines} lines for frame {i}')


if __name__ == '__main__':
    generate()
