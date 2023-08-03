#import the required libraries
import os
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import random
from IPython.display import display
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D,Flatten,MaxPool2D,Dense
from tensorflow.keras import Sequential
from tensorflow.keras.models import save_model
from keras.callbacks import EarlyStopping
%matplotlib inline

data_path="/content/drive/MyDrive/Tomatod"
#listing sub directories
sub_directories = os.listdir(data_path)
labels= pd.read_csv("/content/drive/MyDrive/Tomato diseases.csv")

#Finding number of classes in the data
print("Number of classes:",len(sub_directories))

checking resolution of the images
res = cv2.imread(os.path.join(data_path, sub_directories[0], os.listdir(os.path.join(data_path, sub_directories[0]))[0])).shape
print("Height: ", res[0])
print("Width: ", res[1])
print("Number of Channels: ", res[2])
print("Resolution: {}x{}".format(res[0], res[1]))
from tensorflow.keras.preprocessing.image import ImageDataGenerator

define data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./32)
val_datagen = ImageDataGenerator(rescale=1./32)

train_generator = train_datagen.flow_from_directory('/content/drive/MyDrive/Tomatod split/train',target_size=(256, 256),batch_size=64,class_mode='categorical')

val_generator = val_datagen.flow_from_directory('/content/drive/MyDrive/Tomatod split/val',target_size=(256, 256),batch_size=64,class_mode='categorical')
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow import keras

Define the input shape
inputs = keras.Input(shape=(64, 64, 3))

Add layers
x = Conv2D(filters=28, kernel_size=3, activation="relu")(inputs)
x = MaxPooling2D(pool_size=2)(x)
x = Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = MaxPooling2D(pool_size=2)(x)
x = Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = GlobalMaxPooling2D()(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(4, activation="softmax")(x)

Define the model
base_model = keras.Model(inputs=inputs, outputs=outputs)

Compile the model
base_model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
history = base_model.fit(train_generator, epochs=10, batch_size=32, validation_data=val_generator)

test_loss, test_accuracy = base_model.evaluate(val_generator, batch_size=32)
TF_LITE_MODEL_FILE_NAME = "tfinal_model.tflite"

tf_lite_converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
tflite_model = tf_lite_converter.convert()

tflite_model_name = TF_LITE_MODEL_FILE_NAME
open(tflite_model_name, "wb").write(tflite_model)
