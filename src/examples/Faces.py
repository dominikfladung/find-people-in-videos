# https://www.youtube.com/watch?v=PmZ29Vta7Vc

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('../cascades/data/haarcascade_frontalface_alt.xml')

cap = cv2.VideoCapture(0)

while (True):
    retm, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        print(x, y, w, h)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        rectangleColor = (255, 0, 0)  # BGR 0-255
        rectangleStroke = 2
        startCoordinates = (x, y)
        endCoordinates = (x + w, y + h)
        cv2.rectangle(frame, startCoordinates, endCoordinates, rectangleColor, rectangleStroke)

    cv2.imshow('frame', frame)

    if cv2.waitKey(29) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
