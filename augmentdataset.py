import pandas as pd
from skimage import io
from skimage.transform import rotate
import cv2


def augmentImage(file_path, degrees_of_rotation, lst_imgs):

    for l in lst_imgs:
        img = io.imread(file_path + str(l) + '.jpeg')
        img = rotate(img, degrees_of_rotation)
        io.imsave(file_path + str(l) + '_' + str(degrees_of_rotation) + '.jpeg', img)


trainLabels = pd.read_csv("trainLabels.csv")

trainLabels['image'] = trainLabels['image'].str.rstrip('.jpeg')
trainLabels_DR = trainLabels[trainLabels['level'] >= 1]

lst_imgs_DR = [i for i in trainLabels_DR['image']]

path = "cropped_train/"

augmentImage(path, 90, lst_imgs_DR)
augmentImage(path, 120, lst_imgs_DR)
augmentImage(path, 180, lst_imgs_DR)
augmentImage(path, 270, lst_imgs_DR)

for l in lst_imgs_DR:
    img = cv2.imread(path + str(l) + '.jpeg')
    img = cv2.flip(img, 1)
    cv2.imwrite(path + str(l) + '_mir' + '.jpeg', img)
