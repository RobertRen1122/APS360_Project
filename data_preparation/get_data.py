from bbox_on_frame import bbox_on_frame
from frame_to_vid import  frames_to_vid
from vid_to_frame import vid_to_frame
import sys

if __name__ == '__main__':
    cam_id = sys.argv[1]
    print(f'Getting the data for cam_{sys.argv[1]}')
    vid_to_frame(cam_id=cam_id)
    print("\n")
    print("========================================")
    print('Converted video to frames')
    print("========================================")
    print("\n")
    bbox_on_frame(cam_id=cam_id)
    print("\n")
    print("========================================")
    print('Drew bounding boxes on all frames')
    print("========================================")
    print("\n")
    frames_to_vid(cam_id=cam_id, frame_rate=41)
    print("\n")
    print("========================================")
    print('Reconstructed all frames back to video')
    print("========================================")
    print("\n")