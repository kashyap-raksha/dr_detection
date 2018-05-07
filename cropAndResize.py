import os
import sys
# from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import io
from skimage.transform import resize
import numpy as np

path = 'train/'
new_path = 'cropped_train/'
cropx = 1800
cropy = 1800
img_size = 256

dirs = [l for l in os.listdir(path) if l != '.DS_Store']
total = 0

for item in dirs:
    img = io.imread(path+item)
    y, x, channel = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    img = img[starty:starty+cropy, startx:startx+cropx]
    img = resize(img, (256, 256))
    io.imsave(str(new_path + item), img)
    total += 1