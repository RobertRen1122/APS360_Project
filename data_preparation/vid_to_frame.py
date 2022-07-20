import cv2
from pathlib import Path
import os


def vid_to_frame(cam_id):
    os.makedirs(f'cam_{cam_id}_resized', exist_ok=True)
    cam_id = str(cam_id)
    vid_dir = f'cam_{cam_id}/cam_{cam_id}.mp4'
    vid_dir = str(Path(vid_dir))
    vid = cv2.VideoCapture(vid_dir)
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    frame_num = 0
    while True:
        success, frame = vid.read()
        if success:
            frame = cv2.resize(frame, (448, 448))
            target_dir = f'cam_{cam_id}_resized/{frame_num}.jpg'
            target_dir = str(Path(target_dir))
            cv2.imwrite(target_dir, frame)
            print(f'Wrote {frame_num} to {target_dir}')
        else:
            break
        frame_num += 1
    print(f'Obtained {length} frames with a frame rate of {fps}')
