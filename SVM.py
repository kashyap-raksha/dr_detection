import os
import numpy as np
import cv2
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import svm

path = "cropped_train_64x64"

Files = []
Xlist = []
labels = []
names = []

df_label = pd.DataFrame.from_csv('trainLabels_final.csv')


def get_labels(i):
    l = i.split(os.path.sep)[-1].split(".")[0]
    st1 = df_label.loc[[l]]
    to_int = int(st1.values)
    return to_int


File = []


path = "cropped_train_64x64/"

for file in os.listdir(path):
    Files.append(file)
File = Files[1:]
dim = (128, 128)
img_vect = np.array(
    [cv2.resize(cv2.imread(path + img), dim, interpolation=cv2.INTER_AREA).flatten() for img in File])

Files = []

labels = []
for file in os.listdir(path):
    Files.append(file)
Files = Files[1:]
print(len(Files))
for file in Files:
    s = file.split("_")
    if len(s) == 3:
        a = s[0] + "_" + s[1] + ".jpeg"
        q = get_labels(a)
    else:
        a = file
        q = get_labels(a)
    labels.append(q)

lab = np.array(labels)

(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(img_vect, lab, test_size=0.20, random_state=57)

clf = svm.SVC()
clf = svm.SVC(gamma=0.0001, C=10)
clf.fit(trainFeat, trainLabels)
acc = clf.score(testFeat, testLabels)
print(acc)
print(clf.predict(testFeat))