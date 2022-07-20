import cv2
import os
import sys
from stat import S_ISREG, ST_CTIME, ST_MODE


def getFrame(sec, path, video):
    video.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = video.read()
    if hasFrames:
        cv2.imwrite(os.path.join(path, "frame_" + str(sec) + "_sec.jpg"), image)  # save frame as JPG file
    return hasFrames


def sampling_recombining_frame_for_all_mp4(frameRate):
    filenames = next(os.walk('.'), (None, None, []))[2]
    filenames = [filename for filename in filenames if '.mp4' in filename]
    for filename in filenames:
        filename = os.path.join(os.path.dirname(__file__), filename)
        assert os.path.exists(filename)
        video = cv2.VideoCapture(filename)

        mp4Name = filename.replace("/", " ")
        mp4Name = mp4Name.replace(".", " ")
        mp4Name = mp4Name.split()
        parentPath = os.getcwd()
        folderName = "sampled_frames_" + mp4Name[-2]
        dir_path = os.path.join(parentPath, folderName)
        os.mkdir(dir_path)

        sec = 0

        success = getFrame(sec, dir_path, video)
        while success:
            sec = sec + frameRate
            sec = round(sec, 2)
            success = getFrame(sec, dir_path, video)

        img_array = []
        data = (os.path.join(dir_path, fn) for fn in os.listdir(dir_path))
        data = ((os.stat(path), path) for path in data)
        data = ((stat[ST_CTIME], path)
                for stat, path in data if S_ISREG(stat[ST_MODE]))

        for cdate, path in sorted(data):
            filename_in_folder = os.path.basename(path)
            if '.jpg' in filename_in_folder:
                jpg_dir = os.path.join(dir_path, filename_in_folder)
                img = cv2.imread(jpg_dir)
                height, width, layers = img.shape
                size = (width, height)
                print(filename_in_folder)
                img_array.append(img)

        new_mp4_dir = os.path.join(dir_path, "recombined.mp4")

        out = cv2.VideoWriter(new_mp4_dir, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 1 / frameRate, (1920, 1080))
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

    video.release()


if __name__ == '__main__':
    sampling_recombining_frame_for_all_mp4(10)
