import cv2
from tracker import *


cap = cv2.VideoCapture(r"C:\Users\Admin\Desktop\MyFile\0. DATASCIENCE PROJECT\11. Object tracking\object_tracking\highway.mp4")

while True:
    ret, frame = cap.read()
    
    cv2.imshow('Frame', frame)
    
    key = cv2.waitKey(30)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()