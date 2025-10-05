import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import collections
import numpy as np
import time

## important wtf is this ima breakdown for this after.
BaseOption = mp.tasks.BaseOptions

drawing_points = collections.deque(maxlen=512)
drawing_color = (0, 0, 255) # Red color for drawing
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

def capture_second_camera():
    cam2 = cv2.VideoCapture(1)
    cam2.set(3, 1280)
    cam2.set(4, 720)
    return cam2

def test_cameras():
    print("Testing camera availability...")
    found_cameras = []
    for i in range(5): # Test indices 0 to 4
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera index {i} is available.")
            found_cameras.append(i)
            cap.release()
        else:
            print(f"Camera index {i} is NOT available.")
    return found_cameras

available_cameras = test_cameras()

if not available_cameras:
    print("No cameras found. Please ensure cameras are connected and not in use by other applications.")
    exit()

# If cameras are found, proceed with the original logic using the first two available cameras
cam_index_0 = available_cameras[0] if len(available_cameras) > 0 else -1
cam_index_1 = available_cameras[1] if len(available_cameras) > 1 else -1

if cam_index_0 != -1:
    cam = cv2.VideoCapture(cam_index_0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
else:
    print("Primary camera (index 0) not available.")
    exit()

if cam_index_1 != -1:
    cam2 = cv2.VideoCapture(cam_index_1)
    cam2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
else:
    print("Second camera (index 1) not available. Proceeding with only one camera.")
    cam2 = None # Set cam2 to None if not available

with vision.HandLandmarker.create_from_options(options) as landmarker:
    while cam.isOpened() and (cam2 is None or cam2.isOpened()):
        success, frame = cam.read()
        success2 = False
        frame2 = None
        if cam2:
            success2, frame2 = cam2.read()

        if not success or (cam2 and not success2):
            break

        converted_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=converted_rgb)
        timestamp = int(time.time() * 1000) # ms
        
        result = landmarker.detect_for_video(mp_image,timestamp)

        # # Create a blank canvas for drawing, same size as frame
        h, w, _ = frame.shape
        frame2_drawing_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        if result.hand_landmarks:
            print(result.hand_landmarks)
            
        '''
        drawing_threshold = 0.05 # Adjust this value as needed (normalized coordinates)
        if result.hand_landmarks:
            for hand_landmark in result.hand_landmarks:
                index_finger_tip = hand_landmark[8]
                middle_finger_tip = hand_landmark[12]

                # Calculate Euclidean distance between index and middle finger tips
                distance = ((index_finger_tip.x - middle_finger_tip.x)**2 + \
                            (index_finger_tip.y - middle_finger_tip.y)**2)**0.5

                if distance < drawing_threshold:
                    h, w, _ = frame.shape
                    current_drawing_pos = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))
                    drawing_points.appendleft(current_drawing_pos)

                for lm in hand_landmark:
                    pos = (int(lm.x * w), int(lm.y * h))
                    cv2.circle(frame2_drawing_canvas, pos, 2, (0, 255, 0), -1) # Draw on blank canvas
        
        # Draw lines from drawing_points on the blank canvas
        for i in range(1, len(drawing_points)):
            if drawing_points[i - 1] is None or drawing_points[i] is None:
                continue
            cv2.line(frame2_drawing_canvas, drawing_points[i - 1], drawing_points[i], drawing_color, 5)
        '''

        cv2.imshow("Have Landmark",frame)
        cv2.imshow("Second Camera", frame2_drawing_canvas)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        if key == ord("c"):
            drawing_points.clear()
    cam.release()
    if cam2:
        cam2.release()
    cv2.destroyAllWindows()