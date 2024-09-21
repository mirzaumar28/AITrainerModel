import cv2 as cv 
import mediapipe as mp 
import time 
import math

class poseDetector():
    # def __init__(self, mode = False, upper_body = False,
    #             smooth = True, detection_confidence = 1, 
    #             tracking_confidence = 0.5):
        
       
        # self.pose = self.mpPose.Pose(static_image_mode = self.mode,
        #                             upper_body_only = self.upper_body, 
        #                             smooth_landmarks = self.smooth, 
        #                             min_detection_confidence = self.detection_confidence,
        #                             min_tracking_confidence = self.tracking_confidence)
        # self.pose = self.mpPose.Pose(self.mode,
        #                             self.upper_body, 
        #                             self.smooth, 
        #                             self.detection_confidence,
        #                             self.tracking_confidence)
    def __init__(self,
        static_image_mode=False,
        # model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5):

        self.mode = static_image_mode
        # self.model_complexity = model_complexity
        # self.smooth = smooth_landmarks
        # self.enable_segmentation = enable_segmentation
        # self.smooth_segmentation = smooth_segmentation
        self.dmin_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode
                                       ,min_detection_confidence = 0.5
                                        ,min_tracking_confidence = 0.5
                                         ) 
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw = True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # print(self.results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                                self.mpPose.POSE_CONNECTIONS)
                    
        return img           
                
    def findPosition(self, img, draw = True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, landmarks in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, landmarks)
                cx, cy = int(landmarks.x * w), int(landmarks.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx,cy), 10, (255,0,0), cv.FILLED)

        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw = True):
        # get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # calculate the angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        angle = abs(angle)
        # print(angle)
        if draw:

            cv.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv.line(img, (x2, y2), (x3, y3), (255, 255, 255), 3)

            cv.circle(img, (x1, y1), 10, (0,255,0), cv.FILLED)
            cv.circle(img, (x1, y1), 15, (255,0,0), 2)
            cv.circle(img, (x2, y2), 10, (0,255,0), cv.FILLED)
            cv.circle(img, (x2, y2), 15, (255,0,0), 2)
            cv.circle(img, (x3, y3), 10, (0,255,0), cv.FILLED)
            cv.circle(img, (x3, y3), 15, (255,0,0), 2)

            # cv.putText(img, f"{int(angle)}",(x2 - 20, y2 + 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        return angle





def main():
    cap = cv.VideoCapture(0)
    pTime = 0
    detector = poseDetector()


    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        if len(lmList)!= 0:
            # print(lmList[14])
            cv.circle(img, (lmList[14][1], lmList[14][2]), 20, (0,0,255), cv.FILLED)


        width = int(img.shape[1] )
        height = int(img.shape[0] )
        dim = (width, height)
        img = cv.resize(img, dim, interpolation = cv.INTER_AREA)

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (40, 80), cv.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255))
        cv.imshow('Image', img)
        cv.waitKey(10)


if __name__ == "__main__":
    main()