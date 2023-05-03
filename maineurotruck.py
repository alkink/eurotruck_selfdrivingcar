import cv2
import numpy as np
from mss import mss
from PIL import Image
import calibration as cb
import edgedetectioneurotruck as edg
import time
import matplotlib.pylab as plt

# cap = cv2.VideoCapture('/Users/alkinkabul/Desktop/roadlinedetection/videos/drive2-1.mp4')

# i = 1;
# pTime = 0


# while cap.isOpened():
#     ret, image = cap.read()
#     image = edg.control(image,storage)
#
#     ctime = time.time()
#     fps = 1 / (ctime - pTime)
#     pTime = ctime
#     cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
#
#     cv2.imshow('normal',image)
#
#
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
#     elif cv2.waitKeyEx(1) & 0xFF == ord('s'):
#         out = cv2.imwrite('photo'+str(i)+'.png',image)
#         i += 1
#


# cap.release()
# image2 = edg.control(image,storage)
# cv2.imshow('test',image2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# image = cv2.imread('/Users/alkinkabul/Desktop/eurotruck2-2.png')
# image2 = edg.control(image,storage)
# cv2.imshow('test2', image)
# cv2.imshow('test',image2)
# cv2.waitKey(0)

# (775,467,(482,467),(667,34011),(578,340))





bounding_box = {'top': 160, 'left': 400, 'width': 640, 'height': 390}

sct = mss()

storage = []
isFirst = True
current_lanes = []
curves = []

while True:
    sct_img = sct.grab(bounding_box)
    sct_img =  np.asarray(sct_img)
    img2 = sct_img[:,:,:3]
    storage,img3,isFirst,current_lanes,curves = edg.pipeline(img2,storage,isFirst,current_lanes,curves)
    img3 = cv2.resize(img3,(400,300))
    # print('storage =')
    # print(storage)
    cv2.imshow('screen', np.asarray(img3))
    cv2.imwrite('ss1.jpg',img2)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break
