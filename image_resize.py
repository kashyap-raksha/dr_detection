import os
import cv2

inputpath = "train/"
outputpath = "cropped_train/"
for file in os.listdir(inputpath):
    image = cv2.imread(inputpath+file)
    dim = (256, 256)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    crop = gray_image[:,:].copy()
    cv2.imwrite(outputpath+file, crop)