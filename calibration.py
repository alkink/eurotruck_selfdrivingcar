import cv2
import numpy as np
import glob
import matplotlib.pylab as plt
import os
import pickle


def camera_calibrate():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('/Users/alkinkabul/Desktop/roadlinedetection/calibration/*.jpg')

    for fname in images:
        print(fname)
        image = cv2.imread(fname)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print('sa')
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret == True:
            print(ret)
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            cv2.drawChessboardCorners(image, (9, 6), corners2, ret)


    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist

def calibration_main():
    if os.path.exists('camera_calib.p'):
        with open('camera_calib.p', mode='rb') as f:
            data = pickle.load(f)
            mtx, dist = data['mtx'], data['dist']
    else:
        mtx, dist = camera_calibrate()
        with open('camera_calib.p', mode='wb') as f:
            pickle.dump({'mtx': mtx, 'dist': dist}, f)

    return mtx,dist





def undistort_image(img,mtx,dst):
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dst, (w,h), 1, (w,h))
    dst = cv2.undistort(img, mtx, dst, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    #cv2.imwrite('calibresult.png', dst)
    return dst

def perspective_Transform(image):
    ysize = image.shape[0]
    xsize = image.shape[1]
    # dst = np.float32([(350, 0), (xsize-350, 0), (xsize-350, ysize), (350, ysize)])
    # src = np.float32([(587,455),(696,455),(xsize-150,ysize-15),(250,ysize-15)])

    # src = np.float32([
    # (640,390),
    # (580,390),
    # (370,600),
    # (890,600)
    # ])
    src = np.float32([
    (646, 302),
    (582, 302),
    (520, 398),
    (741, 398)])
    #[('646', '302'), ('582', '303'), ('510', '411'), ('751', '413')]

    dst = np.float32([
    (xsize - 350, 0),
    (350, 0),
    (350, ysize),
    (xsize - 350, ysize)])
    M = cv2.getPerspectiveTransform(src, dst)
    inverse_M = cv2.getPerspectiveTransform(dst,src)
    dst = cv2.warpPerspective(image, M, (xsize, ysize))
    return dst,inverse_M


# img = cv2.imread('photo5.png')
# mtx , dist =calibration_main()
# image2 = (undistort_image(img,mtx,dist))
# print(image2.shape)
# plt.imshow(perspective_Transform(undistort_image(img,mtx,dist)))
# plt.show()





# cv2.imshow('photo',perspective_Transform(undistort_image(img,mtx,dist)))
#cv2.waitKey(0)
