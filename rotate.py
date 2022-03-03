import cv2
import sys

if len(sys.argv) < 2:
    raise Exception("image file not found")
    
name = sys.argv[1]
direction = sys.argv[2]
image = cv2.imread(name)
if image is None:
    raise Exception("image not found")
    
if direction == "R":
    dire = cv2.cv2.ROTATE_90_CLOCKWISE
else:
    dire = cv2.cv2.ROTATE_90_COUNTERCLOCKWISE


rotated = cv2.rotate(image,dire)

cv2.imwrite("rotated_{}".format(name), rotated)
print("rotated")
