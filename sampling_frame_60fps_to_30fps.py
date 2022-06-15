import cv2
import os


if __name__ == '__main__' :
    filenames = next(os.walk('.'), (None, None, []))[2]
    filenames = [filename for filename in filenames if '.mp4' in filename]
    i = 0
    for filename in filenames:
        filename = os.path.join(os.path.dirname(__file__), filename)
        assert os.path.exists(filename)
        video = cv2.VideoCapture(filename);
        def getFrame(sec): 
            video.set(cv2.CAP_PROP_POS_MSEC,sec*1000) 
            hasFrames,image = video.read() 
            if hasFrames: 
                cv2.imwrite("frame "+str(sec)+" sec.jpg", image)     # save frame as JPG file 
            return hasFrames 

    sec = 0 
    frameRate = 0.05
    success = getFrame(sec) 
    while success: 
        sec = sec + frameRate 
        sec = round(sec, 2) 
        success = getFrame(sec)

      
    video.release()