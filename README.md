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

## Feedforward Neural Network
To summarise, the steps involved in computing the output of the Equation neuron in layer Equation is as follows:
Multiply each row of the weight matrix with the output from the previous layer to obtain the weighted sum of inputs from the previous layer.
Convert the weighted sum into the cumulative input by adding the bias vector.
Apply the activation function Equation to the cumulative input to obtain the output vector Equation. 
The pseudocode for a feedforward pass is given below:

We initialise the variable Equation as the input: Equation
We loop through each of the layers computing the corresponding output for each layer, i.e., Equation. 
For l in [1,2,......,L]: Equation
We compute the prediction p by applying an activation function to the output from the previous layer, i.e., we apply a function to Equation, as shown below.  Equation
Neural networks minimise the error in the prediction by optimising the loss function with respect to the parameters in the network. In other words, this optimisation is done by adjusting the weights and biases. We will see how this adjustment is done in subsequent sessions. For now, we will concentrate on how to compute the loss. 
In the case of regression, the most commonly used loss function is MSE/RSS.
In the case of classification, the most commonly used loss function is Cross Entropy/Log Loss.

The task of training neural networks is similar to that of other ML models such as linear regression and logistic regression. The predicted output (output from the last layer) minus the actual output is the cost (or the loss), and we have to tune the parameters Equation and Equation such that the total cost is minimised.  
One important point to note here is that we minimise the average of the total loss and not the total loss that you will get to see shortly. Minimising the average loss implies that the total loss is getting minimised.This can be done using any optimisation routine such as gradient descent. 

### Multiclass Classification using Perceptrons
Until now, you have seen how a perceptron performs binary classification. But if that were the only task a perceptron (or a collection of them) could do, we wouldn’t have cared much about them. It turns out that they can do much more complex things, such as multiclass classification. 

## Back Propagation
For a neural network, you will learn how the loss function is minimized using the gradient descent function by finding the optimum values of weights and biases using backpropagation.
Once the gradients of all the weights and biases are computed, the gradient descent update equation can be used to obtain the updated values of the weights and biases.

Forward Pass with updated parameters

Now, let’s perform another forward pass and check if performing backpropagation and updating the weights and biases once has helped in reducing the loss

You can see that the loss function computed on the updated weights and biases is lower than earlier, which is what we want. By repeatedly performing backpropagation to get optimum values of weights and biases, we can continue reducing the loss. This, eventually, will help us obtain the predicted output that is as close as possible to the actual expected output. This is how a neural network learns using backpropagation.

Since this is a simple neural network, you could compute these values manually. But as the number of hidden layers and of neurons per hidden layer increase, computing these values manually will not be possible. The machine will perform these computations. The aim of considering this example is to demonstrate how a basic neural network behaves so that you can extrapolate the ideas learnt in this simple example to larger networks.

We took the following steps when passing an input through the network: 
1. Forward propagation of the input through the network with random initial values for weights and biases
2. Making a prediction and computing the overall loss
3. Updating model parameters using backpropagation i.e., updating the weights and biases in the network, using gradient descent
4. Forward propagation of the input through the network with updated parameters leading to a decrease in the overall loss 

Repeat the process until the optimum values of weights and biases are obtained such that the model makes acceptable predictions

#### The pseudocode/pseudo-algorithm is given as follows:
1: Initialise with the input 
Forward Propagation
2: For each layer, compute the cumulative input and apply the non-linear activation function on the cumulative input of each neuron of each layer to get the output.
3: For classification, get the probabilities of the observation belonging to a class, and for regression, compute the numeric output.
4: Assess the performance of the neural network through a loss function, for example, a cross-entropy loss function for classification and RMSE for regression.
Backpropagation
5: From the last layer to the first layer, for each layer, compute the gradient of the loss function with respect to the weights at each layer and all the intermediate gradients.
6: Once all the gradients of the loss with respect to the weights (and biases) are obtained, use an optimisation technique like gradient descent to update the values of the weights and biases.
Repeat this process until the model gives acceptable predictions:
7: Repeat the process for a specified number of iterations or until the predictions made by the model are acceptable. 

Please note that we are using Tensorflow primarily to understand what is happening behind the scenes. You are expected to be comfortable with is the high-level API Keras. In the next session, you will learn about Keras. 
The platform has all the tools needed to build a solution and deploy it on different platforms. One part of the complete TensorFlow environment is the TensorFlow machine learning library, which will be covered in this session. TensorFlow is a deep learning library developed by Google. It is used widely in the industry for several different applications.
A tensor is the fundamental data structure used in TensorFlow. It is a multidimensional array with a uniform data type. The data type for an entire tensor is the same. 

So, what impact does this have on the ML process? 
In the case of data frames, all the raw data, such as integers, strings and floats, can be loaded into a single data frame. So, you could load raw data into a data frame and then process the data to convert it into a numerical form for ML. In the case of tensors, data would need to be loaded into another data structure and processed first. And when you are ready to learn from the data, you can load it into a tensor. 

In most use cases in ML, you will use either 2D or 3D tensors. A 2D tensor is equivalent to a matrix. It can be used to represent a feature matrix, with each column being a feature and each row being a data point. A 2D tensor would suffice most ML needs. You might want to convert a higher-dimension tensor to a 2D tensor for learning tasks. Recall the ML algorithms covered so far in this course; the data sets for all of them were in a matrix form, where each row represented a data point and each column represented a feature. This is how all algorithms are designed to work. 
## Tensor 
tensorflow- https://www.tensorflow.org/api_docs/python/tf
Now, let’s summarise the differences between these two types of tensors:
The values of constant tensors cannot be changed once they are declared but those of variable tensors can be.
 Constant tensors need to be initialised with a value while they are being declared, whereas variable tensors can be declared later using operations.
Differentiation is calculated for variable tensors only, and the gradient operation ignores constants while differentiating. 

 You explored how to regularise a neural network using dropouts. You were also introduced to batch normalisation and understood how it helps in training a model. 

#### Regularizing usinh drop outs and Batch normalization

## Module 2:
The objectives of this session include the following: 

Reshaping and Broadcasting
Computational Graphs and Gradients
Minimise Functions
TensorFlow Architecture 
TensorFlow Playground

### Reshaping and Broadcasting:

#### Reshaping:
As the name suggests, reshaping changes the dimension of a tensor. Nevertheless, the reshape function has certain limitations. It can reshape a tensor but cannot remove or add new elements. For instance, consider a tensor of shape (3, 4); it will have 12 elements. The reshape function can change the shape of this tensor to (4, 3) or (6, 2), or even (12, 1), but not (4, 4). In case the shape is changed to (4, 4), you will need 16 elements, but you have only 12; hence, the reshape operation will throw an error.

#### Broadcasting: 
There is a way you can perform an operation on tensors with mismatching dimensions as well. You can broadcast the elements to more dimensions. For instance, you can use a matrix with shape (1) and multiply it by a matrix of shape (3, 3). Even though the shapes are not compatible, this multiplication is executed because the smaller matrix is repeated over all the elements of the larger matrix. This repetition of one matrix is called broadcasting.
In PySpark, the broadcast operation includes storing a given data frame in the cache of all the machines in the network. That way all the machines can obtain the data from the broadcasted data frame faster. The PySpark API has a special command for this operation. In TensorFlow, you do not need to write any extra command to specify the matrix that is to be broadcasted. TensorFlow can determine this independently. So, before moving ahead, let’s watch the next video, which shows how broadcasting is implemented. 

### Computational graphs have a few benefits, which include the following:

#### Visualisation: 
Computational graphs help with visualising algorithms, which in turn makes it easier to develop and maintain complex algorithms. This is especially helpful in the case of neural networks since neural network models are quite complex. 
#### Gradient calculations:
An important ability of TensorFlow is to calculate gradients. Computational graphs are used to trace the dependencies of variables on each other. As you saw in the video, a path is traced from a dependent variable to an independent variable first, and then all the intermediate gradients are calculated and used to compute the expected gradient using the chain rule of differentiation.
#### Distributed architecture:
Computational graphs also help with distributing the training process on a cluster of machines. In one of the upcoming segments, you will learn how this distributed architecture works exactly.

The gradient of any function can be calculated by following these steps: 

1. Initialise the independent variables. The dependent and independent variables need to be tf.Variable-type tensors for the gradient to work.
2. Create a context of GradientTape() and record the equations that relate to the different variables inside the context.
3. To find the derivative of an equation that is recorded in the gradient tape context, use .gradient() outside the context and pass in the variable to differentiate and the variable with respect to which the differentiation will occur.



## Loss Function
Now, let's look at another issue related to the loss function that you have been using until now. This problem arises specifically involving classification in discrete classes if you used the simple class wise comparison loss that can be given by the following formula:
Using this formula would not help us in understanding how the output of the neural network change with delta change in the input. Also, since you know that neural networks train using backpropagation, you need a differential loss function and the function given above is not differentiable

Hence, you use a surrogate loss function like the cross-entropy loss for a classification scenario defined as Equation. Though Equations are not discrete values, you use them to find the discrete predicted class. In other words, you have defined a surrogate function that helps us satisfy both conditions:
The outcomes are discrete values.The loss function is differentiable

## Cross Entropy Loss Function-
The cross-entropy loss solves the above problem to some extent. It helps in achieving the generalisability of the model with respect to the true loss function. In other words, the cross-entropy loss helps us in the prediction of the points which are outside the training data as it predicts the probability of the data point rather than declaring the point as correctly(0) or incorrectly(1) classified as in the case of the true loss function.

However, overfitting can also happen while using the cross-entropy loss
The issue of overfitting can be addressed while using the cross-entropy loss using the early stopping criteria. It states that you should stop the training at the point where the validation error starts increasing as you know that the training or the empirical error will keep decreasing as you increase the number of epochs or training 


## gradient descent
### Momentum Based Methods
Exponential
ADAgrad
RMS

### vanishing gradients
Let us first consider the case of vanishing gradients. You have computed gradients for weights and biases during backpropagation. These gradients usually consist of multiple terms multiplied by each other. Now consider, if all these terms are very small (less than 1), the product of these terms will end up being a very small value.

### exploding gradients

