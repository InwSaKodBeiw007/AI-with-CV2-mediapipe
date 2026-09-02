# AI-with-CV2-mediapipe
Draw image on openCV to AI model mediapipe. with Python script i create blank screen for drawing then send the blank drawed screen to LLM by n8n webhook just when LLM finished the question, n8n send the answer back to server that opening blank screen through api (they share same address by DOCKER local_ip) 

___
I have looking deep for this mediapipe and found just little thing!

.ref
>https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python#model
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models
https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb
https://www.daydev.com/computer-vision/opencv-computer-vision/hand-tracking-webcam-mediapipe.html
https://www.youtube.com/watch?v=zSa-fOGh8es&list=PLlD0XVjVhLaLVZWgJuOBrv4JBsWK99DGV&index=1
https://github.com/samwestby/OpenCV-Python-Tutorial/blob/main/4_video.py
https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb

Copilot is the best partner for me, She help some install and tracking my python version to match with mediapipe-task version
