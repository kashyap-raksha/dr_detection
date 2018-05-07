from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.utils import class_weight

np.random.seed(57)

batch_size = 1000
nb_classes = 5
nb_epoch = 30

img_rows, img_cols = 128, 128
channels = 3
nb_filters = 32
kernel_size = (4,4)

# Import data
labels = pd.read_csv("trainLabels_final.csv")
X = np.load("X_train_128x128.npy")
y = np.array(labels['level'])

weights = class_weight.compute_class_weight('balanced', np.unique(y), y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channels)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, channels)

input_shape = (img_rows, img_cols, channels)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255


y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)
print("y_train Shape: ", y_train.shape)
print("y_test Shape: ", y_test.shape)


print("Training Model")

model = Sequential()

model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                 padding='valid',
                 strides=4,
                 input_shape=(img_rows, img_cols, channels)))
model.add(Activation('relu'))

model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
model.add(Activation('relu'))

model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

kernel_size = (8, 8)
model.add(Conv2D(64, (kernel_size[0], kernel_size[1])))
model.add(Activation('relu'))
# model.add(Dropout(0.2))


model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
print("Model flattened out to: ", model.output_shape)

model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dropout(0.25))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

stop = EarlyStopping(monitor='val_acc',
                     min_delta=0.001,
                     patience=2,
                     verbose=0,
                     mode='auto')

tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
          verbose=1,
          validation_split=0.2,
          class_weight=weights,
          callbacks=[stop, tensor_board])

print("Predicting")
y_pred = model.predict(X_test)


score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


y_pred = [np.argmax(y) for y in y_pred]
y_test = [np.argmax(y) for y in y_test]


precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')


print("Precision: ", precision)
print("Recall: ", recall)


model.save("dr_cnn_model.h5")


print("Completed")