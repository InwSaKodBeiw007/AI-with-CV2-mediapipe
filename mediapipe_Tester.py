import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

## important wtf is this ima breakdown for this after.
BaseOption = mp.tasks.BaseOptions
HandlandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandlandmarkerOptions(
    base_options=BaseOption(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

cam = cv2.VideoCapture(0)
cam.set(3,1280)
cam.set(4,720)

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while cam.isOpened():
        success, frame = cam.read()
        if not success:
            break

        converted_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=converted_rgb)
        import time
        timestamp = int(time.time() * 1000) # ms
        
        result = landmarker.detect_for_video(mp_image,timestamp)

        if result.hand_landmarks:
            for hand_landmark in result.hand_landmarks: ## result ต้องเป็นตัวเดียวกับที่ใช้เช็ค if เราเช็คว่ามันมีค่าไหมเราก็ต้องใช้มันสิ กุคิดไรของกุเนี้ย
                for lm in hand_landmark:
                    ''' ##กว่าจะวาดได้ this is hard than its look.
                # for lm in hand_landmark.landmarks:    ไม่มี .landmarks จะน้ะ
                #     h,w,_ = frame.shape
                #     cx, cy = int(lm.x * w), int(lm.y * h)
                #     cv2.circle(frame,(cx,cy),5,(0,255,0),-1)
                    '''
                    pos = (int(lm.x * frame.shape[1]), 
                    int(lm.y * frame.shape[0]))
                    cv2.circle(frame, pos, 2, (0, 255, 0), -1)

        cv2.imshow("Have Landmark",frame)
        if cv2.waitKey(1) == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()