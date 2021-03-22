import cv2
from collections import deque
import numpy as np
from keras.models import load_model
from 
mlp = load_model('mlp_model.h5')
cnn = load_model('cnn_model.h5')

kernel = np.ones((3,3), np.uint16)
index = 0

bb = np.zeros((480,640,3), dtype=np.uint8)
alp = np.zeros((200, 200, 3), dtype=np.uint8)

points = deque(maxlen=512)

pr1 = 26
pr2 = 26
bll= np.array([120, 80, 80])
blu= np.array([140, 255, 255])


letters = { 1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J',
11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R', 19: 'S', 20: 'T',
21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z', 27: '-'}


camera = cv2.VideoCapture(0)

while True:
    
    (gr, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bM = cv2.inRange(hsv, bll,blu)
    bM = cv2.erode(bM, kernel, iterations=2)
    bM = cv2.morphologyEx(bM, cv2.MORPH_OPEN, kernel)
    bM = cv2.dilate(bM, kernel, iterations=2)
    cnts, _ = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    center = None
    if len(cnts) > 0:
        
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
       
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        points.appendleft(center)

    elif len(cnts) == 0:
        if len(points) != 0:
            blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_BGR2GRAY)
            blur1 = cv2.medianBlur(blackboard_gray, 15)
            blur2 = cv2.GaussianBlur(blur1, (5, 5), 0)
            thresh1 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            blackboard_cnts = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
            if len(blackboard_cnts) >= 1:
                cnt = sorted(blackboard_cnts, key = cv2.contourArea, reverse = True)[0]

                if cv2.contourArea(cnt) > 1000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    alp = blackboard_gray[y-10:y + h + 10, x-10:x + w + 10]
                    newImage = cv2.resize(alphabet, (28, 28))
                    newImage = np.array(newImage)
                    newImage = newImage.astype('float32')/255

                    pr1 = mlp_model.predict(newImage.reshape(1,28,28))[0]
                    pr1 = np.argmax(pr1)

                    pr2 = cnn_model.predict(newImage.reshape(1,28,28,1))[0]
                    pr2 = np.argmax(pr2)
            points = deque(maxlen=512)
            bb = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                    continue
            cv2.line(frame, points[i - 1], points[i], (0, 255, 255), 2)
            cv2.line(bb, points[i - 1], points[i], (255, 255, 255), 8)

  
    cv2.putText(frame, "Multilayer Perceptron : " + str(letters[int(pr1)+1]), (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 255), 2)
    cv2.putText(frame, "Convolution Neural Network:  " + str(letters[int(pr2)+1]), (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


    cv2.imshow("Alphabets Recognition Real Time", frame)

   
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()