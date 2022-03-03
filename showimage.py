import cv2
import sys

if len(sys.argv) < 2:
    raise Exception("image file not found")
name = sys.argv[1]
percent = 0
if len(sys.argv) > 2:
    percent = int(sys.argv[2])

image = cv2.imread(sys.argv[1])
print("shape {} {}".format(image.shape[0], image.shape[1]))

if percent > 0:
    width = int(image.shape[1] * percent / 100)
    height = int(image.shape[0] * percent / 100)
    dim = (width, height)
  
    # resize image
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
 
    #print('Resized Dimensions : ',image.shape)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
