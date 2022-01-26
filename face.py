from asyncore import read
import cv2
from cv2 import rectangle
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# img =cv2.imread('robert.jpg')
webcam= cv2.VideoCapture(0)
while True:
    successful_frame_read , frame= webcam.read()

    greyscale_image=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_coordinate = trained_face_data.detectMultiScale(greyscale_image)

    for (x,y,w,h) in face_coordinate:
        cv2,rectangle(frame,(x,y) , (x+w , y+h) , (0,255,0),2)

    cv2.imshow('param_image', frame)
    cv2.waitKey(1)

# 
# print(face_coordinate)





