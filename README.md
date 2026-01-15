# NN
Deep Learning Neural Networks Basics

Inputs & Outputs 

Depending on the nature of the given task, the outputs of neural networks can either be in the form of classes (if it is a classification problem) or numeric (if it is a regression problem). 
One of the commonly used output functions is the softmax function for classification.
There are various problems you will face while trying to recognise handwritten text using an algorithm, including:
1. Noise in the image
2. The orientation of the text
3. Non-uniformity in the spacing of text
4. Non-uniformity in handwriting 
The MNIST data set takes care of some of these problems, as the digits are written in a box. Now the only problem the network needs to handle is the non-uniformity in handwriting. Since the images in the MNIST data set are 28 X 28 pixels, the input layer has 784 neurons (each neuron takes 1 pixel as an input) and the output layer has 10 neurons (each giving the probability of the input image belonging to any of the 10 classes). The image is classified into the class with the highest probability in the output layer.

To summarise, the weights are applied to the inputs respectively, and along with the bias, the cumulative input is fed into the neuron. An activation function is then applied on the cumulative input to obtain the output of the neuron. We have seen some of the activation functions such as softmax and sigmoid in the previous segment. We will explore other types of activation functions in the next segment. These functions apply non-linearity to the cumulative input to enable the neural network to identify complex non-linear patterns present in the data set.

The activation functions introduce non-linearity in the network, thereby enabling the network to solve highly complex problems. Problems that take the help of neural networks require the ANN to recognise complex patterns and trends in the given data set. If we do not introduce non-linearity, the output will be a linear function of the input vector. This will not help us in understanding more complex patterns present in the data set. 
