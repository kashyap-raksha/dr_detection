import numpy as np
import pandas as pd
import cv2

labels = pd.read_csv("trainLabels_final.csv")

dim = (128, 128)
file_path = "cropped_train/"
lst_imgs = [l+'.jpeg' for l in labels['image']]
X_train = np.array([cv2.resize(cv2.imread(file_path+img), dim, interpolation = cv2.INTER_AREA) for img in lst_imgs])


np.save("X_train_128x128.npy", X_train)