import cv2
import sys

range_max = int(sys.argv[1])
percent = 0
if len(sys.argv) > 2:
    percent = int(sys.argv[2])
posx=0
posy=40
maxy=0
for n in range(0, range_max+1):
    name = "ROI_{}.png".format(n)
    image = cv2.imread(name)
    if image is None:
        print('{} missing'.format(n))
        continue
        
    print("{} shape ({}, {})".format(name, image.shape[0], image.shape[1]))

    if percent > 0:
        width = int(image.shape[1] * percent / 100)
        height = int(image.shape[0] * percent / 100)
        dim = (width, height)
      
    # resize image
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    cv2.namedWindow(name)
    cv2.moveWindow(name, posx, posy)
    posx += image.shape[1]
    if image.shape[1] > maxy:
        maxy = image.shape[1]
    if posx > 2500:
        posx = 0
        posy += maxy
    
    cv2.imshow(name, image)

    #print('Resized Dimensions : ',image.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()
