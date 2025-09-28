import cv2

myvideos = cv2.VideoCapture(0)
myvideos.set(3,1280)
myvideos.set(4,720)

while True:
    checkme,frame = myvideos.read()
    # print(frame)      ## matrix
    # print(checkme)    ## boolean
    if not checkme:
        break
    cv2.imshow("gamekung",frame)
    if cv2.waitKey(1) == ord("q"):
        break





myvideos.release()
cv2.destroyAllWindows()