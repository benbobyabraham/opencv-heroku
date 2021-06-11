import cv2
import numpy as np

def gaussian_blur(image, kernel_size = 9):
    kernel = np.ones((kernel_size , kernel_size))/kernel_size**2
    blurred_image = cv2.filter2D(image, -1, kernel)
    return blurred_image

def median_blur(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

def rgb_to_bgr(image):
    return cv2.cvtColor(image , cv2.COLOR_RGB2BGR)

def diff_channel(image):
    h,w = image.shape[:2]
    image.astype(np.float)
    R = np.zeros((h,w,3), dtype = np.float)
    G = np.zeros((h,w,3), dtype = np.float)
    B = np.zeros((h,w,3), dtype = np.float)

    R[:, :, 0] = image[:, :, 0]/255
    G[:, :, 1] = image[:, :, 1]/255
    B[:, :, 2] = image[:, :, 2]/255
    return (R , G , B)

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def sobel(image):
    sobel_x = np.abs(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = 3))
    sobel_y = np.abs(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = 3))
    return np.clip(sobel_x + sobel_y, 0, 255).astype(np.uint8)

def laplacian(image):
    kernelx = np.array([[1, 1, 1],[1, -8, 1],[1, 1, 1]])
    img = cv2.filter2D(image, -1, kernelx)
    return np.clip(img,0,255).astype(np.uint8)

def scharr(image):
    scharr_x = np.abs(cv2.Scharr(image, cv2.CV_64F, 1, 0))
    scharr_y = np.abs(cv2.Scharr(image, cv2.CV_64F, 0, 1))
    return np.clip(scharr_x + scharr_y, 0, 255).astype(np.uint8)

def perwitt(image):
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    img_prewittx = cv2.filter2D(image, -1, kernelx)
    img_prewitty = cv2.filter2D(image, -1, kernely)
    return np.clip(img_prewittx + img_prewitty,0,255).astype(np.uint8)

def canny(image, minThreshold = 50, maxThreshold = 250):
    return cv2.Canny(image, minThreshold, maxThreshold)

def erosion(image):
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    eroded_image = cv2.erode(image, se,iterations = 2)
    return eroded_image

def dilation(image):
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dialted_image = cv2.dilate(image, se, iterations = 2)
    return dialted_image

def thresholding(image, thresh_value = 145, attr = cv2.THRESH_BINARY):
    _,image =  cv2.threshold(image, thresh_value, 255, attr)
    return image

def boundary_detection(image,  thresh_value = 120):
    return image - thresholding(image, thresh_value)

def gray_inverse(image):
    return 255 - grayscale(image)






