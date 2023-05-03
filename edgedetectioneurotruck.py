import cv2
import numpy as np
import sys
import matplotlib.pylab as plt
import calibration as clb
np.set_printoptions(threshold=sys.maxsize)
global text

diff = np.array([0, 0, 0], dtype='float')


def binary_threshold(img, low, high):
    if len(img.shape) == 2:
        output = np.zeros_like(img)
        mask = (img >= low) & (img <= high)

    elif len(img.shape) == 3:
        output = np.zeros_like(img[:,:,0])
        mask = (img[:,:,0] >= low[0]) & (img[:,:,0] <= high[0]) & (img[:,:,1] >= low[1]) & (img[:,:,1] <= high[1]) & (img[:,:,2] >= low[2]) & (img[:,:,2] <= high[2])
    output[mask] = 1
    return output




def control(combined,prev_lanes,curves,isFirst):
    # current_lanes = []
    #
    # mtx, dst = clb.calibration_main()
    # undistorted_image = clb.undistort_image(image,mtx,dst)
    # perspective_image,inverse_M = clb.perspective_Transform(undistorted_image)
    #
    #
    # combined = np.asarray(hls_converter(perspective_image) + hsv_mask(perspective_image) + lab_mask(perspective_image)
    #                       + adaptive_mask(perspective_image)+r_binary(perspective_image)+Luv_L_binary(perspective_image)
    #                       +S_binary(perspective_image), dtype=np.uint8)
    #
    # combined[combined < 4] = 0
    # combined[combined >= 4] = 1
    #
    #
    #
    # # output_image,current_lanes,text,storage = find_lines(combined,prev_lanes)
    # #
    # # current_lanes,output_image2 = advanced_line_detectionu(combined, current_lanes)


    if isFirst:
        output_image, curves,current_lanes, text, prev_lanes = find_lines(combined, prev_lanes)
        isFirst = False
        print('isFirst = ',isFirst)
    else:
        curves,current_lanes, output_image = advanced_line_detectionu(combined, curves)
        print('pass')
        average_lanes(current_lanes,prev_lanes)

    # if  current_lanes[0] is None or current_lanes[1] is None :
    #     average_lane = average_lanes(current_lanes,prev_lanes)
    #     output_image, current_lanes, text, prev_lanes = find_lines(combined, average_lane)
    # else:
    #     if isFirst == False:
    #         if diff_calculate(current_lanes,prev_lanes):
    #             print('fit diff was too big')
    #             current_lanes = average_lanes(current_lanes,prev_lanes)
    #         else:
    #             current_lanes, output_image = advanced_line_detectionu(combined, current_lanes)
    #     else:
    #         output_image, current_lanes, text, prev_lanes = find_lines(combined, prev_lanes)




    # filled_image = fill_image(image, inverse_M, poly_stack_left, poly_stack_right,text)

    return output_image,curves,current_lanes, text, prev_lanes , isFirst



def pipeline(image,prev_lanes,isFirst,current_lanes,curves):

    mtx, dst = clb.calibration_main()
    undistorted_image = clb.undistort_image(image,mtx,dst)
    perspective_image,inverse_M = clb.perspective_Transform(undistorted_image)


    combined = np.asarray(hls_converter(perspective_image) + hsv_mask(perspective_image) + lab_mask(perspective_image)
                          + adaptive_mask(perspective_image)+r_binary(perspective_image)+Luv_L_binary(perspective_image)
                          +S_binary(perspective_image), dtype=np.uint8)

    combined[combined < 4] = 0
    combined[combined >= 4] = 1

    output_image,curves,current_lanes, text, prev_lanes, isFirst = control(combined,prev_lanes,curves,isFirst)

    # output_image, current_lanes, text, prev_lanes = find_lines(combined, prev_lanes)

    result = fill_image(image,inverse_M,current_lanes[0],current_lanes[1],text)

    return prev_lanes,result , isFirst , current_lanes , curves

#yellow_low_bound = np.array([np.round(45/2),np.round(0.35)])
#yellow_high_bound =
# yellow_lower = np.array([np.round( 40 / 2), np.round(0.00 * 255), np.round(0.35 * 255)])
# yellow_upper = np.array([np.round( 60 / 2), np.round(1.00 * 255), np.round(1.00 * 255)])
# yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)

def hls_converter(image):
    #hls
    hls=cv2.cvtColor(image,cv2.COLOR_BGR2HLS)

    yellow_lower =np.array([10,150,150])
    yellow_upper =np.array([50,255,240])
    yellow_mask = cv2.inRange(hls,yellow_lower,yellow_upper)

    white_lower = np.array([0,160,0])
    white_upper = np.array([255,255,255])
    mask = cv2.inRange(hls,white_lower,white_upper)
    white_mask = binary_threshold(hls,white_lower,white_upper)


    return white_mask

def hsv_mask(image):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    # hsv_low_yellow = np.array((14, 32, 244))
    # hsv_high_yellow = np.array((34, 108, 255))
    # yellow_mask2 = cv2.inRange(hsv, hsv_low_yellow, hsv_high_yellow)

    white_lower = np.array([0, 0, 180])
    white_upper = np.array([255, 40, 255])
    hsv_white = binary_threshold(hsv,white_lower,white_upper)

    return hsv_white

def lab_mask(image):
    lab = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    lab_low_yellow = np.array((171, 114, 126))
    lab_high_yellow = np.array((255, 255, 146))
    yellow_mask = cv2.inRange(lab, lab_low_yellow, lab_high_yellow)
    return yellow_mask

def adaptive_mask(image):
    hls = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    adaptive_white_HSV = cv2.adaptiveThreshold(hsv[:, :, 2], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C\
                                               , cv2.THRESH_BINARY, 121,-25)
    # adaptive_white_HLS = cv2.adaptiveThreshold(hls[:, :, 1], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C\
    #                                            , cv2.THRESH_BINARY, 121,-25)
    adaptive_img = cv2.adaptiveThreshold(image[:, :, 0], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C\
                                         , cv2.THRESH_BINARY, 121, -25)
    # adaptive_white = np.asarray(adaptive_white_HSV  + adaptive_img, dtype=np.uint8)

    adaptive_white = adaptive_white_HSV & adaptive_img
    # adaptive_white[adaptive_white < 2] = 0
    # adaptive_white[adaptive_white >= 2] = 1
    return adaptive_white



def r_binary(image):
    R_channel = image[:,:,0]
    R_binary = binary_threshold(R_channel, 170,255)
    return R_binary

def S_binary(image):
    hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = binary_threshold(s_channel,160,255)
    return  s_binary

def Luv_L_binary(image):
    luv = cv2.cvtColor(image,cv2.COLOR_RGB2LUV)
    L_channel = luv[:,:,0]
    L_binary = binary_threshold(L_channel,185,255)
    return  L_binary




def binaryToImage(combined):
    y,x = combined.shape[:2]
    img2 = np.zeros((y,x,3))
    img2[:,:,0] = combined*255
    img2[:,:,1] = combined*255
    img2[:,:,2] = combined*255
    return img2

def calculate_curveture (img,left_fit,right_fit):
    curve_rad_left = ((1 + (2 * left_fit[0] * 30 + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    curve_rad_right = ((1 + (2 * right_fit[0] * 30 + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
    if(left_fit[1]+right_fit[1]) > 0.2:
        text = 'turn left'
    elif(left_fit[1]+right_fit[1]/2) < -0.2:
        text = 'turn right'
    else:
        text = 'straight'
    return img,text,curve_rad_left,curve_rad_right


def find_lines(image,storage):
    global left_curve,right_curve
    histogram = np.sum(image[int(image.shape[0]/2):,:], axis=0)
    midpoint = np.int32(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    output_image = binaryToImage(image)

    nwindows = 9
    window_height = np.int32(image.shape[0]//nwindows)
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])


    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    margin = 100
    minpix = 50
    for window in range(nwindows):
        window_y_low = image.shape[0] - (1 + window) * window_height
        windows_y_high =image.shape[0]  - (window) * window_height
        windows_x_left_low = leftx_current - margin
        windows_x_left_high = leftx_current + margin
        windows_x_right_low = rightx_current - margin
        windows_x_right_high = rightx_current + margin

        cv2.rectangle(output_image,(windows_x_left_low,window_y_low),(windows_x_left_high,windows_y_high),(0,255,0),2)
        cv2.rectangle(output_image, (windows_x_right_low, window_y_low), (windows_x_right_high, windows_y_high), (0, 255, 0), 2)

        # window_y_low_next = image.shape[0] - (1 + window+1) * window_height
        # windows_y_high_next = image.shape[0] - (window+1) * window_height

        sliding_left = ((nonzeroy >= window_y_low) & (nonzeroy <= windows_y_high)
                        & (nonzerox >= windows_x_left_low) & (nonzerox <= windows_x_left_high)).nonzero()[0]
        sliding_right =((nonzeroy >= window_y_low) & (nonzeroy <= windows_y_high)
                        & (nonzerox >= windows_x_right_low) & (nonzerox <= windows_x_right_high)).nonzero()[0]
        left_lane_inds.append(sliding_left)
        right_lane_inds.append(sliding_right)

        # cv2.circle(output_image,output_image[sliding_right],1,(0,255,0),2)
        if len(sliding_left) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[sliding_left]))
        if len(sliding_right) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[sliding_right]))
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    xsize = output_image.shape[1]
    ysize = output_image.shape[0]
    y_range = np.linspace(0,ysize-1,ysize)

    if(leftx.size == 0 or lefty.size ==0):
        left_curve = None
        print('left curve =',left_curve)
    elif(rightx.size == 0 or righty.size == 0 ):
        right_curve = None
        print('right curve =',left_curve)
    else:
        left_curve = np.polyfit(lefty, leftx, 2)
        right_curve = np.polyfit(righty, rightx, 2)

    left_curve,right_curve,storage = check_lanes(left_curve,right_curve,storage)

    # left_curve = left_right_combined[0]
    # right_curve = left_right_combined[1]
    # storage = add_prev_lanes(left_curve,right_curve,storage)



    predict_left = np.poly1d(left_curve)
    predict_right = np.poly1d(right_curve)

    poly_x_left = predict_left(y_range)
    poly_x_left = poly_x_left[(poly_x_left>0) & (poly_x_left<xsize-1)]

    poly_x_right = predict_right(y_range)
    poly_x_right = poly_x_right[(poly_x_right>0) & (poly_x_right<xsize-1)]

    poly_y_right = np.linspace(ysize - len(poly_x_right),ysize-1,len(poly_x_right))
    poly_y_left = np.linspace(ysize - len(poly_x_left),ysize-1,len(poly_x_left))


    predict_stack_left = np.dstack((np.int32(poly_x_left),np.int32(poly_y_left)))
    predict_stack_right = np.dstack((np.int32(poly_x_right),np.int32(poly_y_right)))


    output_image[[lefty],[leftx]] = [255,0,0]
    output_image[[righty],[rightx]] = [255,0,0]


    cv2.polylines(output_image,predict_stack_left, isClosed= False,color=(201,0,222),thickness=4)
    cv2.polylines(output_image,predict_stack_right, isClosed= False,color=(201,0,222),thickness=4)

    output_image, text ,left_radius, right_radius = calculate_curveture(output_image, predict_left, predict_right)



    current_lanes = [predict_stack_left,predict_stack_right]
    curves = [left_curve,right_curve]
    advanced_line_detectionu(image, curves)
    return output_image,curves,current_lanes,text,storage
    # return output_image


def advanced_line_detectionu(image,curves):
    left_curve = curves[0]
    right_curve = curves[1]
    output_image = np.copy(image)




    margin = 100
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_curve[0]*(nonzeroy**2) + left_curve[1]*nonzeroy + left_curve[2] - margin)) & (nonzerox < (left_curve[0]*(nonzeroy**2) + left_curve[1]*nonzeroy + left_curve[2] + margin)))
    right_lane_inds = ((nonzerox > (right_curve[0]*(nonzeroy**2) + right_curve[1]*nonzeroy + right_curve[2] - margin)) & (nonzerox < (right_curve[0]*(nonzeroy**2) + right_curve[1]*nonzeroy + right_curve[2] + margin)))
    left_x = nonzerox[left_lane_inds]
    left_y = nonzeroy[left_lane_inds]
    right_x = nonzerox[right_lane_inds]
    right_y = nonzeroy[right_lane_inds]

    print(right_lane_inds)

    poly_left_fit = np.polyfit(left_y,left_x,2)
    poly_right_fit = np.polyfit(right_y,right_x,2)


    predict_left = np.poly1d(poly_left_fit)
    predict_right = np.poly1d(poly_right_fit)

    y_range = np.linspace(0, output_image.shape[0] - 1, output_image.shape[0])

    poly_x_left = predict_left(y_range)
    poly_x_right = predict_right(y_range)

    poly_y_right = np.linspace(output_image.shape[0] - len(poly_x_right), output_image.shape[0] - 1, len(poly_x_right))
    poly_y_left = np.linspace(output_image.shape[0] - len(poly_x_left), output_image.shape[0] - 1, len(poly_x_left))

    output_image = binaryToImage(output_image)
    last_image = np.zeros_like(output_image)
    output_image[[left_y], [left_x]] = [0, 0, 255]
    output_image[[right_y], [right_x]] = [255, 0, 0]

    predict_stack_left = np.dstack((np.int32(poly_x_left - margin), np.int32(poly_y_left)))
    predict_stack_left2 = np.array([np.flipud(np.dstack((np.int32(poly_x_left + margin), np.int32(poly_y_left)))[0][:])])
    fill_left = np.hstack((predict_stack_left, predict_stack_left2))
    predict_stack_right = np.dstack((np.int32(poly_x_right - margin), np.int32(poly_y_right)))
    predict_stack_right2 = np.array([np.flipud(np.dstack((np.int32(poly_x_right + margin), np.int32(poly_y_right)))[0][:])])
    fill_right = np.hstack((predict_stack_right, predict_stack_right2))

    cv2.fillPoly(last_image, fill_left, (0, 255, 0))
    cv2.fillPoly(last_image, fill_right, (0, 255, 0))

    # left_line_window1 = np.array([np.transpose(np.vstack([poly_x_left - margin, poly_y_left]))])
    # left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([poly_x_left + margin, poly_y_left])))])
    # left_line_pts = np.hstack((left_line_window1, left_line_window2))
    # right_line_window1 = np.array([np.transpose(np.vstack([poly_x_right - margin, poly_y_right]))])
    # right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([poly_x_right + margin, poly_y_right])))])
    # right_line_pts = np.hstack((predict_stack_right, right_line_window2))


    # cv2.fillPoly(last_image, np.int_([left_line_pts]), (0, 255, 0))
    # cv2.fillPoly(last_image, fill_right, (0, 255, 0))

    result = cv2.addWeighted(output_image, 1, last_image, 0.3, 0)

    current_lanes = [poly_left_fit,poly_right_fit]
    curves = [poly_left_fit,poly_right_fit]
    return curves,current_lanes,result

def check_lanes(left_fit,right_fit,storage):
    if left_fit is None or right_fit is None:
        left_right_combin = np.mean(storage,axis=0)
        left_fit = left_right_combin[0]
        right_fit = left_right_combin[1]

    else:
        left_right_combined = [left_fit,right_fit]
        if diff_calculate(left_right_combined,storage):
            print('diff_detected')
            left_right_combin = np.mean(storage, axis=0)
            left_fit = left_right_combin[0]
            right_fit = left_right_combin[1]
            return left_fit,right_fit,storage
        storage.append(left_right_combined)
        if(len(storage) > 5):
            storage.pop(0)

    return left_fit,right_fit,storage

def average_lanes(current_lanes,prev_lanes):
    prev_lanes.append(current_lanes)
    if (len(prev_lanes) > 5):
        prev_lanes.pop(0)
    new_lanes = np.average(prev_lanes, axis=0)
    return new_lanes

def add_prev_lanes(left_fit,right_fit,prev_lanes):
    if left_fit is  None or right_fit is  None:
        print('cant find')
    else:
        current_lanes = [left_fit,right_fit]
        prev_lanes.append(current_lanes)
        # print(prev_lanes)
        if (len(prev_lanes) > 5):
            prev_lanes.pop(0)
    return prev_lanes


def diff_calculate(current_lanes,prev_lanes):
    if  prev_lanes :
        diff = np.abs(current_lanes - np.mean(prev_lanes, axis=0))
        if diff[0][0] > 0.001:
            return True
        if diff[0][1] > 0.25:
            return True
        if diff[0][2] > 1000.:
            return True
        if diff[1][0] > 0.001:
            return True
        if diff[1][1] > 0.25:
            return True
        if diff[1][2] > 1000.:
            return True
        return False
    else: return False

def fill_image(img,inverse_M,poly_stack_left,poly_stack_right,text):
    mtx,dst = clb.calibration_main()
    undistorted_image = clb.undistort_image(img,mtx,dst)

    empty_img = np.zeros_like(undistorted_image)
    ysize,xsize=empty_img.shape[:2]

    poly_stack_right = poly_stack_right[:,::-1]
    concatenated_stacks = np.concatenate((poly_stack_left,poly_stack_right),axis=1)

    filled_img = cv2.fillPoly(empty_img,concatenated_stacks,(0,255,0))


    warped_image = cv2.warpPerspective(filled_img,inverse_M,(xsize, ysize))
    weighted_img = cv2.addWeighted(undistorted_image,0.7,warped_image,0.3,0,dtype=cv2.CV_8U)
    weighted_img = cv2.putText(weighted_img, text, (50, 90), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    return weighted_img

# storage = []
# image = cv2.imread('ss1.jpg')
#
# storage,output_image = pipeline(image,storage)
# # image2 = binaryToImage(pers_img)
# cv2.imshow('test1',image)
# cv2.imshow('test',output_image)
# cv2.waitKey(0)