import numpy as np
import cv2
from mss import mss
from PIL import Image

def grab_screen():

    sct = mss()

    while 1:
        w, h = 800, 640
        monitor = {'top': 160, 'left': 400, 'width': 640, 'height': 390}
        img = Image.frombytes('RGB', (w,h), sct.grab(monitor).rgb)
        cv2.imshow('test', cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
