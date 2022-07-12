import cv2

VIDEO_DIR = 'cam_5.mp4'

def vid_to_frame(vid_dir):
    vid = cv2.VideoCapture(vid_dir)
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    frame_num = 0
    while(True):
        success, frame = vid.read()
        if success:
            frame = cv2.resize(frame, (448, 448))
            cv2.imwrite(f'./temp_frames_resized/{frame_num}.jpg', frame)
            print(f'Wrote {frame_num}...')
        else:
            break
        frame_num += 1
    print(f'{length} frames with a frame rate of {fps}')

if __name__ == '__main__':
    vid_to_frame(VIDEO_DIR)