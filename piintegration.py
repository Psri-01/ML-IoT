import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter

#Load the saved TFLite model
model_path = '/content/drive/MyDrive/Mini Project/t_model.tflite'
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

#Define the class labels
class_labels = ['Bacterial_spot', 'Healthy', 'Late_blight', 'Septoria_leaf_spot', 'Tomato_Yellow_Leaf_Curl_Virus']

#Load the input image
img_path = '/content/drive/MyDrive/freshleaf.jpg'
img = cv2.imread(img_path)

#Resize the image to the input shape of the model
img = cv2.resize(img, (64, 64))

#Normalize and convert the image to FLOAT32
img = img.astype('float32') / 255.0

#Add a batch dimension to the image
img = np.expand_dims(img, axis=0)

#Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#Set the input tensor to the image
interpreter.set_tensor(input_details[0]['index'], img)

#Run the inference
interpreter.invoke()

#Get the output tensor
output = interpreter.get_tensor(output_details[0]['index'])

#Get the predicted class label
predicted_label = class_labels[np.argmax(output)]

#Print the predicted class label
print(predicted_label)
