# ML Bootcamp

![image](https://user-images.githubusercontent.com/112813661/228872019-af7e9f83-8301-4912-a380-9fc8f7301c46.png)

# Machine Learning Algorithm Library

Description: Building Machine Learning Algorithm Library.
Implemented Linear (and Polynomial) Regression, Logistic Regression, KNN and an n-layer Neural Networks from scratch.

# Features:
•	Linear Regression: 
In this algorithm, we basically use a linear function to estimate our output, say we draw a straight line though the training data sets and this straight is drawn in such a way that if we give any new feature or input to it, then we are supposed to have a good estimated output from that straight line. 

•	Logistic Regression: 
In this algorithm, we are using the mathematical sigmoid function in order to classify the sets of data into categories by learning from the training sets, and able to classify a new input unknown.

•	k-Nearest Neighbours: 
Algorithm that, given an input, chooses the most common class out of the k nearest datapoints to that point.

•	n-layer Neural Networks: 
Artificial network with an input layer, an output layer and atleast one hidden layer in between.

# Used Softwares:
Python 3.11

![image](https://user-images.githubusercontent.com/112813661/228876719-e19342ba-691f-443d-b82c-469a286c6380.png)

Google Colaboratory

![image](https://user-images.githubusercontent.com/112813661/228879302-58d23f2a-46a7-44d3-a7a8-9e53c91ffb22.png)



# Used libraries:

![image](https://user-images.githubusercontent.com/112813661/228873843-b4dc33d5-2fc0-41a8-b422-7a8ac4f17c2d.png)

![image](https://user-images.githubusercontent.com/112813661/229074595-b66c2f54-6630-4f0b-90ff-f1421ddcf5b0.png)

# About
---> In this library the above features are implemented from scratch in python without use of any mchine learning library.

---> The library file created is ml_ai1.py

---> There are .ipynb files are for each algorithm for its training and testing.

---> In each of the .ipynb files, the ml_ai1.py file is used as library.

---> We can use those .ipynb files to train our data and to save the outputs in a .csv file.

---> The outputs on test data are stored on the respective .csv files.

# Documentation

This library contains classes for each algorithm and some functions defined explicitly

## Classes:
### Linear_Regression
            methods:
            cal_grad(self, X, y, w, b) ---> computes the cost
            cal_cost(self, X, y, w1, b1) ---> computes the gradient
            find_wb(self, X, y, winit, binit, alpha, num_iters, fcost, fgrad) ---> finds optimal weights and biases for given number of iterations
            
### Polynomial_Regression
            methods:
            deg(self) ---> takes input degree
            p_cal_grad(self, X, y, w, b, a) ---> computes the gradient
            p_cal_cost(self, X, y, w1, b1, a) ---> computes the cost
            p_find_wb(self, X, y, winit, binit, alpha, num_iters, fcost, fgrad, a) ---> finds optimal weights and biases for given number of iterations
            
### Logistic_Regression
            methods:
            cal_cost_lgstc(self, X, y, w, b) ---> computes the cost
            cal_grad_lgstc ---> computes the gradient
            find_wb_lgstc(self, X, y, winit, binit, alpha, iters, fcost_lgstc, fgrad_lgstc) ---> finds optimal weights and biases for given number of iterations
            
### Knn
            methods:
            k(self) ---> takes the user input k
            cal_dist(self, X, test) ---> Calculates the distance


### For n layer Neural Network:
# Its Classes
# ---> Gen_layer
# ---> Activation_ReLU

# Activation_Softmax
            methods ---> forward(self, inputs) - for forward propagation

# ---> Loss_CategoricalCrossentropy
# ---> Loss(Loss_CategoricalCrossentropy)
# ---> Actvn_Softmax_Loss_CategoricalCrossentropy
# ---> SGD - stochastic gradient descent

# Note: In each of the above classes except Activation_Softmax, there are two methods:
         forward - For forward propagation,
         backward - For backward propagation

# Other functions:
        neurons() - It takes user input number of neurons.
        iterations() - It takes user input number of iterations
        alpha() - It takes user input learning rate.
     
            

# Contributors

https://github.com/Dyuti01/ml-bootcamp
