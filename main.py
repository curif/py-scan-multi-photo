
import cv2
import numpy as np
import sys
import math
import argparse
from scipy.signal import find_peaks_cwt


def get_size(img):
    ih, iw = img.shape[:2]
    return iw * ih
    
def process_image(image, size_min_filter, threshold=200):
    photos = []
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blurred, threshold, 255,cv2.THRESH_BINARY_INV)[1]

    # apply morphology to ensure regions are filled and remove extraneous noise
    kernel = np.ones((7,7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((11,11), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    print("detected: {} images".format(len(cnts)))

    # Iterate thorugh contours and filter for ROI
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        
        ROI = original[y:y+h, x:x+w]
        if get_size(ROI) > size_min_filter:
            #paint image for debug
            #cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
            #rotate?
            if ROI.shape[0] > ROI.shape[1]:
                ROI = cv2.rotate(ROI, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        
            photos.append(ROI)
            
    #cv2.imwrite('thresh_{}.png'.format(threshold), thresh)
    
    return photos
    
#https://stackoverflow.com/questions/65791233/auto-selection-of-gamma-value-for-image-brightness
def auto_gama(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    mid = 0.5
    mean = np.mean(val)
    meanLog = math.log(mean)
    midLog = math.log(mid*255)
    gamma = midLog/meanLog
    gamma = 1 / gamma
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv2.LUT(image, lookUpTable)

def autoAdjustments_with_convertScaleAbs(img):
    alow = img.min()
    ahigh = img.max()
    amax = 255
    amin = 0

    # calculate alpha, beta
    alpha = ((amax - amin) / (ahigh - alow))
    beta = amin - alow * alpha
    # perform the operation g(x,y)= α * f(x,y)+ β
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return new_img
    
def morph_close(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel1)
    div = np.float32(gray)/(close)
    return np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX))
    
#https://stackoverflow.com/questions/23195522/opencv-fastest-method-to-check-if-two-images-are-100-same-or-not
def is_similar(image1, image2):
    return image1.shape == image2.shape and not(np.bitwise_xor(image1,image2).any())
    
def find_similar(image, arrimages):
    for img in arrimages:
        if is_similar(image, img):
            return True
    return False

#only for black & white photos.
def AutoAdjustmentBlackWhite(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        peaks = find_peaks_cwt(hist.flatten(), np.arange(1, 50))
        indent_right = peaks[-1]
        indent_left = peaks[0]
        dist = indent_right - indent_left
        alpha = 256.0 / dist
        beta = -(indent_left * alpha)

        result = cv2.convertScaleAbs(gray, result, alpha, beta)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        

def convertScale(img, alpha, beta):
    """Add bias and gain to an image with saturation arithmetics. Unlike
    cv2.convertScaleAbs, it does not take an absolute value, which would lead to
    nonsensical results (e.g., a pixel at 44 with alpha = 3 and beta = -210
    becomes 78 with OpenCV, when in fact it should become 0).
    """

    new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    return new_img.astype(np.uint8)
# https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape/56909036
# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=25):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = convertScale(image, alpha=alpha, beta=beta)
    #return (auto_result, alpha, beta)
    return auto_result

# https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape/56909036
# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrastV2(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return auto_result

def enhance(img):
    return cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

parser = argparse.ArgumentParser(description='photo scan')
parser.add_argument('--image_name', help="image path name to scan", default='1.png')
parser.add_argument('--image_size_min_factor', help="image path name to scan", default=15, type=int)
parser.add_argument('--base_name', help="name prefix for processed photos", default='ROI')
parser.add_argument('--filter_gamma', help="apply filter", action='store_true')
parser.add_argument('--filter_brightness', help="apply filter", action='store_true')
parser.add_argument('--filter_enhance', help="apply filter", action='store_true')
args = parser.parse_args()
print(args)

image = cv2.imread(args.image_name)
size_min = get_size(image) / args.image_size_min_factor


print("first pass")
photos = process_image(image, size_min)
print("first pass: {} photos".format(len(photos)))

# seccond pass 
print("2nd pass")
photos += [ photo
            for threshold in [200, 150, 100]
            for ROI in photos
            for photo in process_image(ROI, size_min, threshold)
            ]
print("2nd pass: {} photos total".format(len(photos)))

if args.filter_gamma:
    print("Adjust Gama")
    photos = [ auto_gama(photo) 
                for photo in photos ]
            
if args.filter_brightness:
    print("Adjust brightness")
    photos = [ automatic_brightness_and_contrast(photo, clip_hist_percent=25) 
            for photo in photos ] 
            #+ [ automatic_brightness_and_contrast(photo) for photo in photos ]

# print("Adjust AutoAdjustmentBlackWhite")
# photos = [ AutoAdjustmentBlackWhite(photo) 
            # for photo in photos ]
if args.filter_enhance:
    print("Enhance")
    photos = [ enhance(photo) 
                for photo in photos ] 
            
#delete similar
difference = []
for i in range(0, len(photos)-1):
    if not find_similar(photos[i], photos[i+1:]):
        print("photo {} is unique".format(i))
        difference.append(photos[i])

photos = difference

#saving            
image_number = 0
for ROI in photos:
    imagename="{}_{}.png".format(args.base_name, image_number)
    cv2.imwrite(imagename, ROI)
    image_number += 1
    print("**** saved: {}".format(imagename))

print("done")
