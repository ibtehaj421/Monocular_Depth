import cv2
import numpy as np

depth = cv2.imread("data/nyu2_train/basement_0001a_out/1.jpg", cv2.IMREAD_UNCHANGED)
print(depth.dtype, depth.min(), depth.max())
