import mediapipe as mp 
import cv2 as cv 
import time 
import PoseModule as pm
import numpy as np



cap = cv.VideoCapture(0)
pTime = 0
detector = pm.poseDetector()
count = 0 
dir = 0

while True:
    success, img = cap.read()
    img = detector.findPose(img, draw = False)
    lmList = detector.findPosition(img, draw = False)
    # print(lmList)
    if len(lmList) != 0:
        # right arm 
        # detector.findAngle(img, 12, 14, 16)
        # left arm
        angle = detector.findAngle(img, 11, 13, 15)
        # left leg
        # detector.findAngle(img, 23, 25, 27)
        percentage  = np.interp(angle, (30, 200), (0,100))
        bar = np.interp(angle, (30, 200), (650, 100))
        # print(angle, percentage)

        # chck for the curls 
        if percentage < 7:
            if dir == 0:
                count += 0.5
                dir = 1
        if percentage > 90:
            if dir == 1:
                count += 0.5
                dir = 0
        # print(count)
        cv.rectangle(img, (20, 280), (40, 300), (0,0,0), cv.FILLED)
        cv.putText(img, f"{int(count)}", (20, 300), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # cv.rectangle(img, (500, 100), (575, 350), (0,255,0), cv.FILLED)
        # cv.rectangle(img, (300, int(bar)), (375, 350), (0,255,0), cv.FILLED)
        # cv.putText(img, f"{int(percentage)}%", (400, 75), cv.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 4)



    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv.putText(img, f"{int(fps)}", (20, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    
    cv.imshow('Image', img)
    cv.waitKey(1)