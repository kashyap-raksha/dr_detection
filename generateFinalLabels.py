import pandas as pd
import os

df = pd.read_csv("trainLabels.csv")

names = []
for l in df['image']:
    names.append(l)

newlines = []
for name in os.listdir("cropped_train"):
    s = name.split("_")
    if len(s) == 3:
        a = s[0]+"_"+s[1]
        s = df.loc[df['image'] == a, "level"].values[0]
        newRow = "%s,%s\n" % (name.split(".")[0], s)
        newlines.append(newRow)

with open("trainLabels_final.csv", "a") as f:
    for newRow in newlines:
        f.write(newRow)