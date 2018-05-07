# dr_detection


Steps to Run the application:

Pre-requisites:
Make sure all of these libararies are already installed before running the app
1. sklearn
2. pandas
3. numpy
4. keras
5. tensorflow
6. skimage
7. opencv


Option 1: Using already trained model

1. Go to the root directory of the application and run python app.py
2. This will start application on http://localhost:8001

Option 2: Generate the model once again (Takes about 16 hours on 16 GB RAM I5 3rd Gen processor)

1. The images are alredy preprocessed and a .npy file is generated to make it easy for generating the modle.
2. Run the CNN.py file as python CNN.py
3. This will generate dr_cnn_model.h5 model file in the same folder
4. Copy this folder and paste it in python web application's root folder.
5. We are ready to run our application by running python app.py
6. This will start application on http://localhost:8001

