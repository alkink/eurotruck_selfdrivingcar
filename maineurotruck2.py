import cv2
import numpy as np
from mss import mss
import edgedetectioneurotruck2 as edg

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
    img3,current_lanes,curves = edg.pipeline(img2,current_lanes,curves)
    img3 = cv2.resize(img3,(400,300))
    cv2.imshow('screen', np.asarray(img3))

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break
