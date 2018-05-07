
import numpy as np
import pandas as pd
from PIL import Image


trainLabels = pd.read_csv('trainLabels.csv')

trainLabels['image'] = [i + '.jpeg' for i in trainLabels['image']]
trainLabels['black'] = np.nan
file_path = "cropped_train/"

lst_imgs = [l for l in trainLabels['image']]

trainLabels['black'] = [1 if np.mean(np.array(Image.open(file_path + img))) == 0 else 0 for img in lst_imgs]
trainLabels = trainLabels.loc[trainLabels['black'] == 0]
trainLabels.to_csv('trainLabels_master.csv', index=False, header=True)