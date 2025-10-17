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
    finally:    ## begin call last after try or except can be useful
        # handle_cooldown = False ## ไม่ได้โดน set เป็น False

        global handle_cooldown  ## main loop จะเห็นการเปลี่ยนแปลงนี้
        handle_cooldown = False

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
    if fingers == [1,0,0,0,1] and handle_cooldown:
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
        if result and handle_cooldown:  ## add handle_cooldown here
            fingers,lmlist = result
            previous_pos,canvas = draw(result,previous_pos,canvas)
            last_send_time = sendto(fingers,canvas,last_send_time,cooldown=3) ## this cooldown for delay requests

        combine_cans = cv2.addWeighted(src1=img,alpha=0.7,src2=canvas,beta=0.4,gamma=0)

        ''''''
        # อัพเดท answer จาก thread
        if latest_answer_from_thread:
            answer = latest_answer_from_thread



        # Display the answer text on the combined image
        base_x = combine_cans.shape[1] - 10
        base_y = 60
        line_height = 40
        # Fixed: Check if answer is a string (not None and not a numpy array)
        if answer is not None and isinstance(answer, str):
            text_size = cv2.getTextSize(answer, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            # text_x = combine_cans.shape[1] - text_size[0] - 10

            text_x = base_x - text_size[0]
            text_y = base_y
            cv2.putText(combine_cans, answer, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            base_y += line_height  ## เลื่อนตำแหน่ง Y ลงมาเพื่อไม่ให้ทับกัน ของ cooldown_text

        cooldown_text = "Cooldown ready" if handle_cooldown else "On Cooldown"  ## pseudo code
        cooldown_size = cv2.getTextSize(cooldown_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cooldown_x = base_x - cooldown_size[0]
        cooldown_y = base_y  ## ให้เป็น base_y ที่โดน ปรับปรุ่งเมื่อมี answer มา
        cooldown_color = (0, 255, 0) if handle_cooldown else (0, 0, 255) ## cooldown_color เล่นท่าหน่อยแบบ pseudo code
        cv2.putText(combine_cans, cooldown_text, (cooldown_x, cooldown_y), cv2.FONT_HERSHEY_SIMPLEX, 1, cooldown_color, 2, cv2.LINE_AA)

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

handle_cooldown = True  ## to control sending requests

'''
ใช้ได้เลยเพราะ server flask เรารันใน container โดย host="0.0.0.0"  
Script นี้รันใน host machine และมีการตั้งค่าใน docker-compose.yml ของ server flask กับ n8n ให้
container flask-web สามารถเข้าถึงได้ที่ localhost:8000 ตรง ports: 8000:8000
'''

url = "http://localhost:8000/uploads"

''''''
app = Flask(__name__)

@app.route('/new-answer', methods=['POST'])
def new_answer():
    data = request.json or {}

    def answerPoint(data):
        global latest_answer_from_thread
        answer = data.get("answer")
        latest_answer_from_thread = answer

        global handle_cooldown
        handle_cooldown = True  ## allow sending requests again
        
    threading.Thread(target=answerPoint,args=(data,), daemon=True).start()  ## edited
    return jsonify({"status": "received"})

latest_answer_from_thread = None

@app.get("/status/test")
def handle_status():
    return jsonify({"status": "AI-with-CV2-mediapipe is running"}), 200 ## ตรงนี้ถ้าจะยิงมาต้องใช้ ip ของ host machine
## เพื่อให้ container ใน docker-compose มองเห็น server นี้ได้

def run_listener():
    app.run(host="0.0.0.0", port=9000, debug=False)
threading.Thread(target=run_listener, daemon=True).start()

if __name__ == "__main__":
    main()