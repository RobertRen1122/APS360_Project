import cv2
import os

TEMP_VID_DIR = 'temp_video'
TEMP_FRAMES_DIR = 'resized_temp_frames_with_draw_bounding_box'
VID_NAME = 'combined_resized.mp4'

def frames_to_vid(frames_dir, frame_rate):
    img_files = os.listdir(frames_dir)
    for f in img_files:
        if '.jpg' not in f:
            img_files.remove(f)
    img_files_sorted = sorted(img_files, key=lambda x: int(list(x.split('_'))[0]))
    video_path = os.path.join(TEMP_VID_DIR, VID_NAME)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), frame_rate, (448, 448))
    for img in img_files_sorted:
        print(f'Writing {img}...')
        img = os.path.join(TEMP_FRAMES_DIR, img)
        img = cv2.imread(img)
        out.write(img)


if __name__ == '__main__':
    frames_to_vid(frames_dir=TEMP_FRAMES_DIR, frame_rate=41)