import cv2
import os
from pathlib import Path



def frames_to_vid(cam_id, frame_rate):
    cam_id = str(cam_id)
    frames_dir = f'cam_{cam_id}_bboxed'
    frames_dir = str(Path(frames_dir))
    img_files = os.listdir(frames_dir)
    videos_dir = 'bboxed_videos'
    exists = os.path.exists(videos_dir)
    if exists:
        os.makedirs(videos_dir, exist_ok=False)
    for f in img_files:
        if '.jpg' not in f:
            img_files.remove(f)
    img_files_sorted = sorted(img_files, key=lambda x: int(list(x.split('_'))[0]))
    video_path = f'{videos_dir}/cam_{cam_id}.mp4'
    video_path = str(Path(video_path))
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), frame_rate, (448, 448))
    for img in img_files_sorted:
        print(f'Writing {img}...')
        img = f'{frames_dir}/{img}'
        img = str(Path(img))
        img = cv2.imread(img)
        out.write(img)
