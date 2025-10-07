from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
import requests, time, os, threading

from flask import Flask, jsonify,request

def getHandinfo(img):
    
    # Find hands in the current frame
    # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
    # The 'flipType' parameter flips the image, making it easier for some detections
    hands, img = detector.findHands(img, draw=True, flipType=True)

    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand = hands[0]  # Get the first hand detected
        lmList = hand["lmList"]  # List of 21 landmarks for the first hand
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    else:
        return None
    
def draw(info,previous_pos,canvas):
    fingers, lmlist = info
    current_pos = None

    if fingers == [0,1,0,0,0] or fingers == [1,1,0,0,0]:
        current_pos =  lmlist[8][0:2]

        if previous_pos is None: previous_pos = previous_pos = current_pos
        cv2.line(img=canvas,pt1=current_pos,pt2=previous_pos,color=(255,0,255),thickness=10)
    return current_pos,canvas


def _send_file_async(filepath, filename, url):
    try:
        with open(filepath, 'rb') as screenshot:
            files = {'file': (filename, screenshot, 'image/png')}
            requests.post(url, files=files, timeout=10)
            # print(f'sended! with :{response.json()}')
            # print(f"  URL : {response.json().get('url', 'N/A')}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending file: {e}")
    finally:
        # Clean up the temporary file after sending
        if os.path.exists(filepath):
            os.remove(filepath)
def sendto(fingers, canvas, last_send_time, cooldown=3):
    """
        last_send_time: เวลาที่ส่งล่าสุด
        cooldown: ระยะเวลารอระหว่างการส่ง (วินาที)
    
        Returns:
            เวลาปัจจุบัน ถ้าส่งไปแล้ว, มิฉะนั้นคืนค่า last_send_time
    """
    global latest_answer_from_thread ##  main loop จะเห็น latest_answer_from_thread ทันที
    if fingers == [1,0,0,0,1]:
        current_time = time.time()
            
            # เช็คว่าผ่าน cooldown period แล้วหรือยัง
        if current_time - last_send_time >= cooldown:
            filename = f"drawing_{int(current_time)}.png"
            filepath = os.path.join("temp", filename)
              
            # บันทึกภาพ canvas
            cv2.imwrite(filepath, canvas)

            threading.Thread(target=_send_file_async, args=(filepath, filename, url), daemon=True).start()  ## send post requests

        return current_time
    return last_send_time

def main():
    previous_pos = None
    canvas = None
    last_send_time = 0
    answer = None
    while True:
        success, img = cap.read()
        img = cv2.flip(img,1)

        if not success:
            break
        if canvas is None: canvas = np.zeros_like(img)

        #ML handdetection
        result = getHandinfo(img)
        if result:
            fingers,lmlist = result
            previous_pos,canvas = draw(result,previous_pos,canvas)
            last_send_time = sendto(fingers,canvas,last_send_time,cooldown=3) ## this cooldown for delay requests

        combine_cans = cv2.addWeighted(src1=img,alpha=0.7,src2=canvas,beta=0.4,gamma=0)

        ''''''
        # อัพเดท answer จาก thread
        if latest_answer_from_thread:
            answer = latest_answer_from_thread

        # Fixed: Check if answer is a string (not None and not a numpy array)
        if answer is not None and isinstance(answer, str):
            text_size = cv2.getTextSize(answer, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = combine_cans.shape[1] - text_size[0] - 10
            text_y = text_size[1] + 10
            cv2.putText(combine_cans, answer, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the image in a window
        cv2.imshow("Image", img)
        cv2.imshow("draw screen",canvas)
        cv2.imshow("combined",combine_cans)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        if key == ord('c'):
            canvas = np.zeros_like(img)

    cap.release()
    cv2.destroyAllWindows()


cap = cv2.VideoCapture(0)
# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)
cap.set(3,1280)
cap.set(4,720)

url = "http://localhost:8000/uploads"

''''''
app = Flask(__name__)

@app.route('/new-answer', methods=['POST'])
def new_answer():
    data = request.json or {}

    def answerPoint():
        global latest_answer_from_thread
        answer = data.get("answer")
        latest_answer_from_thread = answer
        
    threading.Thread(target=answerPoint, daemon=True).start()
    return jsonify({"status": "received"})

latest_answer_from_thread = None

def run_listener():
    app.run(port=9000, debug=False)
threading.Thread(target=run_listener, daemon=True).start()

if __name__ == "__main__":
    main()