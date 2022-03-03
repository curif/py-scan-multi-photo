import cv2
import sys

if len(sys.argv) < 2:
    raise Exception("image file not found")
    
name = sys.argv[1]

coords_from = sys.argv[2].split(",")
coords_to = sys.argv[3].split(",")
coords_from = [int(x) for x in coords_from]
coords_to = [int(x) for x in coords_to]
print(name, coords_from, coords_to)

image = cv2.imread(name)
if image is None:
    raise Exception("image not found")
    
cropped = image[coords_from[1]:coords_to[1],
                coords_from[0]:coords_to[0]
                ]
cv2.imwrite("cropped_{}".format(name), cropped)
print("cropped")
