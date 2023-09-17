# Edge-based detection in Tomato leaves using Machine Learning

# Motivation behind this project

It is an individual project integrating the capabilities of ML and IoT, done to have a deep understanding of the connection between the two. I was also interested in building and deploying ML models so this was a good use case to start with.

# Brief description of the project

In this project, an ML model was developed to identify four types of leaf diseases, namely Tomato Bacterial spot, Late blight, Septoria leaf spot and Yellow Leaf Curl Virus, apart from healthy tomato leaves using a light weight CNN model and deploy it on Raspberry Pi 3 (bullseye) after converting it to a tflite(TensorFlow lite) model to test the presence or absence of diseases in the leaves by predictive analysis. Accuracy obtained by this model was almost the same as that obtained by a VGG-19 model (Visual Geometry group, pretrained model). It helped me gain insights into image processing, data manipulation and hands-on machine learning.

# General stages of training Convolutional Neural Networks

![image](https://github.com/Psri-01/ML-IoT/assets/114862496/a2dc12f2-edfa-4f9f-96ea-8196a61ba334)


1. Data collection and preparation
The first step is to collect and prepare data for training the CNN model. Collected images of tomato leaves with and without diseases from the PlantVillage dataset on Kaggle and reduced it to 5 directories with 1k images per directory, and then resizing and cropping the images to a standard size.

2. Model architecture
The next step is to design the architecture of the CNN model. This involves choosing the number of layers, the type of layers, and the activation functions for each layer.

3. Model training
Once the model architecture is designed, the next step is to train the model. This involves feeding the training data into the model and adjusting the weights of the model so that it learns to classify images of tomato leaves with and without diseases. 

4. Model evaluation
Once the model is trained, the next step is to evaluate its performance. This involves testing the model on a dataset of images that it has not seen before, and measuring its accuracy and other metrics. Using ImageDataGenerator and normalizing the images to 1./32 for easier training and validation set splits. Validation accuracy was 96.69% and training accuracy was around 92%. Final loss is 0.103

5. Model deployment
Once the model is evaluated and found to be satisfactory, the next step is to convert it to a TensorFlow lite format and deploy it in a Raspberry Pi for edge detection of tomato leaf diseases and be further used in real-world applications.

# Key Challenges faced and how I addressed them

1. Data imbalance: The dataset of images that I used was imbalanced, meaning that there were more images of tomato leaves with diseases than without. This made it difficult for the model to learn to classify images of tomato leaves with diseases.
To address this challenge, I used a technique called data augmentation to artificially increase the number of images of tomato leaves with diseases. This involved creating new images of tomato leaves with diseases by rotating, flipping, and cropping the existing images.

2. Lack of labeled data: Labeled data is data that has been manually annotated with the correct labels. This is essential for training a machine learning model, but it can be difficult and time-consuming to create.
To address this challenge, I created 5 labels in the detection phase in Raspberry Pi Thonny editor.

# Model Summary 

![image](https://github.com/Psri-01/ML-IoT/assets/114862496/8af56300-152e-4d60-aa06-308cb84a4533)

# Code description (summary)

1 • The code imports necessary libraries such as os, cv2, pandas, seaborn, matplotlib, numpy, and tensorflow for image processing, data manipulation, visualization, and machine learning tasks.


2 • It defines the path to the directory containing the dataset and loads the labels from a CSV file.


3 • It determines the number of classes in the dataset by counting the subdirectories in the data path.


4 • It reads the resolution (height, width, and number of channels) of the first image in the dataset to understand the input dimensions.


5 • It sets up data generators using ImageDataGenerator to preprocess and augment the image data for training and validation.


6 • It constructs a convolutional neural network (CNN) model using Keras. The model consists of convolutional layers, max pooling layers, a global max pooling layer, dense layers, and dropout regularization.


7 • It compiles the model with the Adam optimizer, categorical cross-entropy loss function, and accuracy as the evaluation metric.


8 • It trains the model on the training data using the fit function, specifying the number of epochs, batch size, and validation data.


9 • It evaluates the trained model on the validation data to obtain the test loss and accuracy.


10 • It converts the trained model to a TensorFlow Lite model using the tf.lite.TFLiteConverter and saves it as a .tflite file.


11 • Loading the model: The code starts by importing necessary libraries such as numpy, cv2 (OpenCV), and Interpreter from tflite_runtime. It then specifies the path to the saved TFLite model file (model_path) and creates an instance of the Interpreter class using the model path. The interpreter is allocated tensors to prepare for inference.


12 • Defining class labels: The code defines a list of class labels (class_labels) corresponding to the different categories the model can predict. The order of the labels should match the output of the model.


13 • Loading and preprocessing the input image: The code specifies the path to the input image (img_path) and reads it using OpenCV's cv2.imread function. The image is then resized to the input shape required by the model (64x64 pixels). The pixel values of the image are normalized to the range of [0, 1] by dividing by 255.0, converting the image to float32. Finally, a batch dimension is added to the image by expanding its dimensions using np.expand_dims.


14 • Running inference with the model: The code retrieves the input and output details of the model using the interpreter.get_input_details() and interpreter.get_output_details() methods. The input tensor is set to the preprocessed image using interpreter.set_tensor. The inference is performed by invoking the interpreter with interpreter.invoke(). The output tensor is obtained using interpreter.get_tensor.


15 • Predicting and printing the class label: The code uses np.argmax to find the index of the highest probability in the output tensor. This index is used to retrieve the corresponding class label from class_labels. Finally, the predicted class label is printed using print(predicted_label).

# Why VGG?

VGG architectures have a simple and uniform structure, which makes them easy to understand and implement. They are proficient in extracting intricate features from images, making them effective for various computer vision tasks.
Transfer Learning: Due to their uniform architecture they are used for transfer learning where pre-trained weights can be fine-tuned for specific tasks.

Cons:
VGG networks can be computationally expensive to train and require substantial computing resources.
Memory Consumption: The uniform structure of VGG leads to a large number of parameters, resulting in higher memory consumption. This leads to difficulty when trying to deploy it in a memory constrained device, which is why I went for the Sequential model instead.

# Future Scope

The model could suggest appropriate fertilizers based on the disease, obtain model accuracy, use transfer learning approaches to get improved results, reduce model size further to enable deployment in an edge device (ESP S3 or Arduino BLE) and lastly, we could get real time images of a tomato leaf using a Pi Camera and detect the disease for enabling smart agricultural decisions, which would inform farmers of early crop failure and enable them to take proactive measures to prevent it.
