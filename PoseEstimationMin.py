import cv2.cv2 as cv
import mediapipe as mp
import time


mpDraw= mp.solutions.drawing_utils

mpPose = mp.solutions.pose
pose = mpPose.Pose()
'''
static_image_mode---->if False it will try to detect and will track only when the confidence is high
if true it will always try to find new detections

def __init__(self,
               static_image_mode=False,
               model_complexity=1,
               smooth_landmarks=True,
               enable_segmentation=False,
               smooth_segmentation=True,
               min_detection_confidence=0.5,---->if detection confidence is more than 50% it will move on to tracking
               min_tracking_confidence=0.5)---->if more that 50%it will continue to track else it will come back to 
                                                detection

'''

cap = cv.VideoCapture('PoseVideos/1.mp4')
cTime=0
pTime=0
while True:
    success, image = cap.read()

    imageRGB = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(image, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id,land in enumerate(results.pose_landmarks.landmark):
            h,w,c= image.shape
            print(id,land)
            cx,cy=int(land.x*w),int(land.y*h)
            cv.circle(image, (cx, cy), 6, (255, 0, 0), cv.FILLED)





    cTime= time.time()
    fps= 1/(cTime-pTime)
    pTime= cTime

    cv.putText(image,str(int(fps)), (70,100), cv.FONT_HERSHEY_TRIPLEX, 3, (255,0,100),3)
    cv.imshow("Image", image)

    cv.waitKey(1)

