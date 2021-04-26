import cv2, time, imutils
import numpy as np

# Capturing video
# video = cv2.VideoCapture(0)
# video = cv2.VideoCapture('cap.mp4')
video = cv2.VideoCapture('vtest.avi')
# video = cv2.VideoCapture('rtsp://root:root@27.72.56.161:554/axis-media/media.amp')

frame_width = int(video.get(3))
frame_height = int(video.get(4))

isResize = True
width_resized = 1000

min_object_area = 300

print('video resolution: ',(frame_width, frame_height))

# Saving video
# videoWriter = cv2.VideoWriter('detect.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (320,240))

def getBackgound():
    frames = []
    # #Randomly select 25 frames
    # frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
    # #Store selected frames in an array
    # for fid in frameIds:
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    #     ret, frame = cap.read()
    #     frames.append(frame)

    #select 20 first frames

    for i in range(0,30):
        ret, frame = video.read()
        if (isResize):
            frame = imutils.resize(frame, width_resized)
        frames.append(frame)

    # Calculate the median along the time axis
    medianFrame = np.mean(frames, axis=0).astype(dtype=np.uint8)
    cv2.imshow('background', medianFrame)
    gray_background = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.GaussianBlur(gray_background, (15, 15), 0)

    return gray_background

# Assigning our static_back to None
static_back = getBackgound()
# cv2.imshow('background', static_back)


# Infinite while loop to treat stack of image as video
# Back to first frame
video.set(cv2.CAP_PROP_POS_FRAMES, 0)
pre_frame = None
while True:
    # Reading frame(image) from video
    check, frame = video.read()
    # frame = cv2.resize(frame, (640,480))
    if (isResize):
        frame = imutils.resize(frame, width_resized)

    # Converting color image to gray_scale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Converting gray scale image to GaussianBlur
    # so that change can be find easily
    gray = cv2.GaussianBlur(gray, (15, 15), 0)

    # Difference between static background
    # and current frame(which is GaussianBlur)
    diff_frame = cv2.absdiff(static_back, gray)
    # if pre_frame is not None:
    #     diff_frame = cv2.absdiff(pre_frame, gray)
    # else:
    #     diff_frame = gray.copy()
    # pre_frame = gray.copy()

    # If change in between static background and
    # current frame is greater than 30 it will show white color(255)
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]

    # remove small white noises
    # img_erosion = thresh_frame.copy()
    img_erosion = cv2.erode(thresh_frame, np.ones((7,7), np.uint8), iterations=1)

    # increases object area (interest objects)
    img_dilation = cv2.dilate(img_erosion, np.ones((7,7), np.uint8), iterations=1)
    img_dilation = cv2.dilate(img_dilation, np.ones((7,7), np.uint8), iterations=1)

    # Finding contour of moving object
    cnts,_ = cv2.findContours(img_dilation.copy(),
                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < min_object_area:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        # making green rectangle arround the moving object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)


    # Displaying the difference in currentframe to the staticframe
    cv2.imshow("Difference Frame", diff_frame)

    # Displaying the black and white image in which if
    # intensity difference greater than 30 it will appear white
    cv2.imshow("Threshold Frame", thresh_frame)

    # Displaying the final black and white image after remove noises
    cv2.imshow("Remove noises", img_dilation)

    # Displaying color frame with contour of motion of object
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)
    # if q entered whole process will stop

    #save video

    if key == ord('q'):
        #Stop
        break
    if key == ord('p'):
        #PAUSE
        cv2.waitKey(0)

    # frame = cv2.resize(frame, (320,240))
    # videoWriter.write(frame)

video.release()
# videoWriter.release()

# Destroying all the windows
cv2.destroyAllWindows()
