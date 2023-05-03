import cv2
import numpy as np
import calibration as clb

def binary_threshold(img, low, high):
    if len(img.shape) == 2:
        output = np.zeros_like(img)
        mask = (img >= low) & (img <= high)

    elif len(img.shape) == 3:
        output = np.zeros_like(img[:,:,0])
        mask = (img[:,:,0] >= low[0]) & (img[:,:,0] <= high[0]) & (img[:,:,1] >= low[1]) & (img[:,:,1] <= high[1]) & (img[:,:,2] >= low[2]) & (img[:,:,2] <= high[2])
    output[mask] = 1
    return output

def binaryToImage(combined):
    y,x = combined.shape[:2]
    img2 = np.zeros((y,x,3))
    img2[:,:,0] = combined*255
    img2[:,:,1] = combined*255
    img2[:,:,2] = combined*255
    return img2






def applyThreshold(channel, thresh):
    # Create an image of all zeros
    binary_output = np.zeros_like(channel)

    # Apply a threshold to the channel with inclusive thresholds
    binary_output[(channel >= thresh[0]) & (channel <= thresh[1])] = 1

    return binary_output

def rgb_rthresh(img, thresh=(125, 255)):
    # Pull out the R channel - assuming that RGB was passed in
    channel = img[:,:,0]
    # Return the applied threshold binary image

    return (applyThreshold(channel, thresh))

def hls_sthresh(img, thresh=(125, 255)):
    # Convert to HLS
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # Pull out the S channel
    channel = hls[:,:,2]
    # Return the applied threshold binary image
    return (applyThreshold(channel, thresh))

def lab_bthresh(img, thresh=(125, 255)):
    # Convert to HLS
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # Pull out the B channel
    channel = lab[:,:,2]
    # Return the applied threshold binary image
    return applyThreshold(channel, thresh)

def luv_lthresh(img, thresh=(125, 255)):
    # Convert to HLS
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    # Pull out the L channel
    channel = luv[:,:,0]
    # Return the applied threshold binary image
    return applyThreshold(channel, thresh)

def r_thresh(img):
    r = img[:,:,0]

def binaryPipeline(img, show_images=False, \
                   sobel_kernel_size=7, sobel_thresh_low=35, sobel_thresh_high=50, \
                   canny_kernel_size=5 , canny_thresh_low=50, canny_thresh_high=150, \
                   r_thresh_low=225, r_thresh_high=255, \
                   s_thresh_low=220, s_thresh_high=250, \
                   b_thresh_low=175, b_thresh_high=255, \
                   l_thresh_low=215, l_thresh_high=255 \
                  ):
    r = rgb_rthresh(img, thresh=(r_thresh_low, r_thresh_high))
    s = hls_sthresh(img, thresh=(s_thresh_low, s_thresh_high))
    b = lab_bthresh(img, thresh=(b_thresh_low, b_thresh_high))
    l = luv_lthresh(img, thresh=(l_thresh_low, l_thresh_high))

    print(l.shape)

    combined_binary = np.zeros_like(r)
    print(combined_binary.shape)
    combined_binary[ (r == 1) | (s == 1) | (b == 1) | (l == 1) ] = 1

    return combined_binary

img = binaryToImage(binaryPipeline(image,show_images=True))
cv2.imshow('test', pers_img)
cv2.imshow('test2', img)
cv2.imshow('hls test',hls_sthresh(pers_img,thresh=(75,255)))
cv2.waitKey(0)

