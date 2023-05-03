import cv2
import numpy as np
import math
from mss import mss
import calibration as clb
import time
import edgedetectioneurotruck2 as edg
import keyboard_control as kb

def roi(image, polygons):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked = cv2.bitwise_and(image, mask)
    return masked

def proceesed_img(original_image):
    proceesed_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    proceesed_img = cv2.GaussianBlur(proceesed_img, (5, 5), 0)
    proceesed_img = cv2.Canny(proceesed_img, threshold1=150, threshold2=163)
    # these polygon repressent the data point within with the pixel data are selected for lane detection
    # polygons = np.array([[500, 345], [750, 345], [750, 470], [500, 480]])
    polygons = np.array([[200, 345], [650, 345], [650, 380], [200, 380]])
    proceesed_img = roi(proceesed_img, [polygons])
    return proceesed_img



def display_line(image, line):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (10, 100, 255), 12)
            # cv2.line(line_image, (x_1, y), (x_2, y), (0, 255, 0), 3)
            # cv2.line(line_image, (int(l_x + line_center), y + 25), (int(l_x + line_center), y - 25), (100, 25, 50), 5)
            cv2.circle(line_image, (477, 360), 5, [150, 10, 25], 10)
    return line_image


def fill_image(img,inverse_M,lines,text):
    mtx,dst = clb.calibration_main()
    undistorted_image = clb.undistort_image(img,mtx,dst)

    empty_img = np.zeros_like(undistorted_image)
    ysize,xsize=empty_img.shape[:2]

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(empty_img, (x1, y1), (x2, y2), (10, 100, 255), 12)


    warped_image = cv2.warpPerspective(empty_img,inverse_M,(xsize, ysize))
    weighted_img = cv2.addWeighted(undistorted_image,0.7,warped_image,0.3,0,dtype=cv2.CV_8U)
    weighted_img = cv2.putText(weighted_img, text, (50, 90), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    return weighted_img


for i in list(range(4))[::-1]:
    time.sleep(1)


sct = mss()


bounding_box = {'top': 160, 'left': 400, 'width': 640, 'height': 390}
while (True):
    sct_img = sct.grab(bounding_box)
    sct_img = np.asarray(sct_img)
    img2 = sct_img[:, :, :3]
    new_image = proceesed_img(img2)
    lines = cv2.HoughLinesP(new_image, 1, np.pi / 180, 100, np.array([]), minLineLength=50, maxLineGap=15)
    left_coordinate = []
    right_coordinate = []
    left_lane = []
    right_lane = []


    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            left_lane.append((x1,y1))
            right_lane.append((x2,y2))
            slope = (x2 - x1) / (y2 - y1)
            if slope < 0.1:
                left_coordinate.append([x1, y1, x2, y2])
                text = 'turn left'
                # kb.left()
            elif slope > 0.1:
                right_coordinate.append([x1, y1, x2, y2])
                text = 'turn right'
                # kb.right()
        l_avg = np.average(left_coordinate, axis=0)
        r_avg = np.average(right_coordinate, axis=0)
        l = l_avg.tolist()
        r = r_avg.tolist()
        try:
            # with the finded slope and intercept, this is used to find the value of point x on both left and right line
            # the center point is denoted by finding center distance between two lines

            c1, d1, c2, d2 = r
            a1, b1, a2, b2 = l
            l_slope = (b2 - b1) / (a2 - a1)
            r_slope = (d2 - d1) / (c2 - c1)
            l_intercept = b1 - (l_slope * a1)
            r_intercept = d1 - (r_slope * c1)
            y = 360
            l_x = (y - l_intercept) / l_slope
            r_x = (y - r_intercept) / r_slope
            distance = math.sqrt((r_x - l_x) ** 2 + (y - y) ** 2)
            # line_center repressent the center point on the line
            line_center = distance / 2
            print(line_center)

            center_pt = [(l_x + line_center)]
            f_r = [(l_x + (line_center * 0.25))]
            f_l = [(l_x + (line_center * 1.75))]
            # create a center point which is fixed
            center_fixed = [477]
            x_1 = int(l_x)
            x_2 = int(r_x)


        except:
            pass
            # print('slow')

    line_image = display_line(img2, lines)
    combo_image = cv2.addWeighted(img2, 0.8, line_image, 1.2, 2)
    # line_image2 = fill_image(img2,inverse_M,lines,text)
    # combo_image2 = cv2.resize(combo_image, (400, 300))
    combo_image3 = cv2.resize(combo_image,(400,300))
    cv2.imshow('test',cv2.resize(new_image,(400,300)))
    print(left_coordinate)

    if cv2.waitKey(25) & 0xff == ord('q'):
        cv2.destroyAllWindows()
        break