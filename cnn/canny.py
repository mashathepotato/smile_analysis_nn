import cv2
import numpy as np
  
# img = cv2.imread("dataset/smile arc/ideal_cropped/ideal_cropped3.jpg")
img = cv2.imread("raw_data/smile arc/ideal/ideal0.jpg")
  
# Setting parameter values
t_lower = 150  # Lower Threshold
t_upper = 150  # Upper threshold
  
# Applying the Canny Edge filter

edge = cv2.Canny(img, t_lower, t_upper)

# img = cv2.resize(img, (300, 300))
# edge = cv2.resize(edge, (200, 200))

for i in range(edge.shape[0]):
    for j in range(edge.shape[1]):
        print(edge[i, j], end=' ')
    print()

cv2.imshow('edge', edge)
cv2.waitKey(0)
cv2.destroyAllWindows()