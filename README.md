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

###The features of these activation functions are as follows:

Sigmoid: When this type of function is applied, the output from the activation function is bound between 0 and 1 and is not centred around zero. A sigmoid activation function is usually used when we want to regularise the magnitude of the outputs we get from a neural network and ensure that this magnitude does not blow up.
Tanh (Hyperbolic Tangent): When this type of function is applied, the output is centred around 0 and bound between -1 and 1, unlike a sigmoid function in which case, it is centred around 0.5 and will give only positive outputs. Hence, the output is centred around zero for tanh. 
ReLU (Rectified Linear Unit): The output of this activation function is linear in nature when the input is positive and the output is zero when the input is negative. This activation function allows the network to converge very quickly, and hence, its usage is computationally efficient. However, its use in neural networks does not help the network to learn when the values are negative.
Leaky ReLU (Leaky Rectified Linear Unit): This activation function is similar to ReLU. However, it enables the neural network to learn even when the values are negative. When the input to the function is negative, it dampens the magnitude, i.e., the input is multiplied with an epsilon factor that is usually a number less than one. On the other hand, when the input is positive, the function is linear and gives the input value as the output. We can control the parameter to allow how much ‘learning emphasis’ should be given to the negative value.


Recall that models such as linear regression and logistic regression are trained on their coefficients, i.e., the task is to find the optimal values of the coefficients to minimize a cost function.
Neural networks are no different; they are trained on weights and biases.

During training, the neural network learning algorithm fits various models to the training data and selects the best prediction model. The learning algorithm is trained with a fixed set of hyperparameters associated with the network structure. Some of the important hyperparameters to consider to decide the network structure are given below:

1. Number of layers
2. Number of neurons in the input, hidden and output layers
3. Learning rate (the step size taken each time we update the weights and biases of an ANN)

   The notations that you will come across going forward are as follows:

W represents the weight matrix.
b stands for bias.
x represents the input.
y represents the ground truth label.
p represents the probability vector of the predicted output for the classification problem 
