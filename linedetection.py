import cv2
import numpy as np

image = cv2.imread('/Users/alkinkabul/Desktop/eurotruck2.png')
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
thresh = cv2.GaussianBlur(gray,(5,5),0)
canny = cv2.Canny(thresh,threshold1=200,threshold2=120)
lines = cv2.HoughLinesP(canny,1,np.pi/180,50,None,50,10)
if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            cv2.line(image, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
            print(l[0],l[1],l[2],l[3])
cv2.imshow('img', image)
cv2.waitKey(0)

