
import cv2
import numpy as np
import sys
import math


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
            
    cv2.imwrite('thresh_{}.png'.format(threshold), thresh)
    
    return photos
    
#https://stackoverflow.com/questions/65791233/auto-selection-of-gamma-value-for-image-brightness
def auto_gama(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    mid = 0.5
    mean = np.mean(val)
    meanLog = math.log(mean)
    midLog = math.log(mid*255)
    gamma =midLog/meanLog
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

#https://stackoverflow.com/questions/23195522/opencv-fastest-method-to-check-if-two-images-are-100-same-or-not
def is_similar(image1, image2):
    return image1.shape == image2.shape and not(np.bitwise_xor(image1,image2).any())
    
def find_similar(image, arrimages):
    for img in arrimages:
        if is_similar(image, img):
            return True
    return False
    
image = cv2.imread(sys.argv[1])
size_min = get_size(image) / 15

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

print("Adjust Gama")
photos = [ auto_gama(photo) 
            for photo in photos ]
print("Adjust brightness")
photos = [ autoAdjustments_with_convertScaleAbs(photo) 
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
    imagename="ROI_{}.png".format(image_number)
    cv2.imwrite(imagename, ROI)
    image_number += 1
    print("**** saved: {}".format(imagename))

print("done")
