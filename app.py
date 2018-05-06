import os
from flask import Flask, render_template, request, flash, redirect, url_for
from keras.models import load_model
import numpy as np
from PIL import Image
import cv2

app = Flask(__name__)
app.secret_key = 'secret key'

root_path = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=['POST'])
def upload():
    folder = os.path.join(root_path, 'files')
    print(folder)

    if not os.path.isdir(folder):
        os.mkdir(folder)
    destination = ""
    for file in request.files.getlist("file"):
        destination = "/".join([folder, file.filename])
        file.save(destination)
    flash('File uploading successful')

    immatrix = np.array([np.array(Image.open(destination))], 'f')
    print(immatrix.shape)
    immatrix = immatrix.reshape(1, 128, 128, 3).astype('float32')
    model = load_model('dr_cnn_model.h5')
    res = model.predict(immatrix, verbose=1)
    index = res.flatten().argmax(axis=0)
    if index == 0:
        msg = "No DR is Detected"
    else:
        msg = "Level "+index+" DR is Detected. Please consult a Doctor ASAP."
    os.remove(destination)
    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(port=8001, debug=True)
