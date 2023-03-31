import copy
import math
import numpy as np

#----------------------------------------------------------------------------------------------------------
#---------------------------------------------Linear Regression--------------------------------------------
#----------------------------------------------------------------------------------------------------------

# Find w and b ---> Gradient desecnt
# Calculate gradient
class Linear_Regression:
    def cal_grad(self, X, y, w, b):

        m, n = X.shape
        fwb = (np.matmul(X.reshape(m,n), w.reshape(n,1)) + b)  # using reshape to make desired order and to make more readeable the orders of matrix
        diff = fwb - y.reshape(m,1)

        djdw = np.matmul(diff.reshape(1, m), X)
        djdb = np.sum(diff)

        return (djdw / m), (djdb / m)


    # Calculate cost
    def cal_cost(self, X, y, w1, b1):

        m, n = X.shape
        fwb = (np.matmul(X.reshape(m,n), w1.reshape(n,1)) + b1)
        diff = fwb - y.reshape(m,1)
        acst = np.sum(diff**2)/(2*m)  # Average cost to get a smaller value

        return acst


    # Gradient Descent
    def find_wb(self, X, y, winit, binit, alpha, num_iters, fcost, fgrad):
        # Recording history of cost and the parameters w and b for each iteration
        recJ = []  
        w0 = copy.deepcopy(winit)
        b0 = binit
        for i in range(num_iters):
            djdw, djdb = fgrad(X, y, w0, b0)  # fgrad to call cal_gard
                                            # fcost to call cal_cost
            # Simultaneous update of parameters
            w0 -= alpha * djdw
            b0 -= alpha * djdb
            if i < 100000:
                recJ.append(fcost(X, y, w0, b0))

            if i % (math.ceil(num_iters/10)) == 0:
                print(f"Iteration no. {i}: cost {recJ[-1]:0.4e}")
            
            # Last iterations to check variation
            if (num_iters - 10) <= i <= (num_iters - 1):
                print(f"Iteration no. {i}: cost {recJ[-1]:0.3e}")  

        return w0, b0, recJ

# Iterations
def iterations():
    while True:
        try:
            return int(input("Enter no. of iterations: "))
        except ValueError:
                print("Integer expected...")

# Learning rate
def alpha():
    while True:
        try:
            return float(input("Enter learning rate: "))
        except ValueError:
            print("Give input alpha correctly...")

# Calculate model output
def cal_out(x_f, w_f, b_f):
    f_wb = (np.dot(w_f, x_f)) + b_f
    return f_wb


#---------------------------------------------------------------------------------------------------------
#-------------------------------------Polynomial Regression-----------------------------------------------
#---------------------------------------------------------------------------------------------------------
class Ploynomial_Regression:
    # Find w and b ---> Gradient desecnt
    # Calculate gradient
    def deg(self):
        while True:
            l = input("Enter degree: ")
            if l.isdigit() and int(l) > 0:
                return int(l)
            else:
                print(f"Positive integer expected...")
                continue

    def p_cal_grad(self, X, y, w, b, a):

        m, n = X.shape
        fwb = (np.matmul(X.reshape(m,a), w.reshape(a,1)) + b)  # using reshape to make desired order and to make more readeable the orders of matrix
        diff = fwb - y.reshape(m,1)

        djdw = np.matmul(diff.reshape(1, m), X)
        djdb = np.sum(diff)

        return (djdw / m), (djdb / m)


    # Calculate cost
    def p_cal_cost(self, X, y, w1, b1, a):

        m, n = X.shape
        fwb = (np.matmul(X.reshape(m,a), w1.reshape(a,1)) + b1)
        diff = fwb - y.reshape(m,1)
        acst = np.sum(diff**2)/(2*m)  # Average cost to get a smaller value

        return acst


    # Gradient Descent
    def p_find_wb(self, X, y, winit, binit, alpha, num_iters, fcost, fgrad, a):
        # Recording history of cost and the parameters w and b for each iteration
        recJ = []  
        w0 = copy.deepcopy(winit)
        b0 = binit
        for i in range(num_iters):
            djdw, djdb = fgrad(X, y, w0, b0, a)  # fgrad to call cal_gard
                                            # fcost to call cal_cost
            # Simultaneous update of parameters
            w0 -= alpha * djdw
            b0 -= alpha * djdb
            if i < 100000:
                recJ.append(fcost(X, y, w0, b0, a))

            if i % (math.ceil(num_iters/10)) == 0:
                print(f"Iteration no. {i}: cost {recJ[-1]:0.4e}")
            
            # Last iterations to check variation
            if (num_iters - 10) <= i <= (num_iters - 1):
                print(f"Iteration no. {i}: cost {recJ[-1]:0.3e}")  

        return w0, b0, recJ



#----------------------------------------------------------------------------------------------------------------
#---------------------------------------Logistic Regression------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------
class Logistic_Regression:
    # Logistic Cost
    def cal_cost_lgstc(self, X, y, w, b):
        m, n = X.shape

        z = (np.matmul(X.reshape(m,n), w.reshape(n,1)) + b)
        fwb = 1/(1 + np.exp(-z))
        cost = -np.matmul(y.reshape(1, m), np.log(fwb)) - np.matmul((1 - y.reshape(1, m)), (np.log(1 - fwb)))
        t_cost = np.sum(cost)  # total cost

        return t_cost/m

    # Gradient
    def cal_grad_lgstc(self, X, y, w, b):
        m, n = X.shape
        z = (np.matmul(X.reshape(m,n), w.reshape(n,1)) + b)   # z ---> m x 1
        # element wise sigmoid
        fwb = 1/(1 + np.exp(-z))

        diff = fwb - y.reshape(m, 1)  # m x 1
        djdw = np.matmul(diff.reshape(1, m), X)  # djdw ---> 1 x n
        djdb = np.sum(diff)
        
        return djdw/m , djdb/m

    # Gradient Descent
    def find_wb_lgstc(self, X, y, winit, binit, alpha, iters, fcost_lgstc, fgrad_lgstc):
        recJ = []  
        w = copy.deepcopy(winit)
        b = binit
        for i in range(iters):
            djdw, djdb = fgrad_lgstc(X, y, w, b)  # fgrad to call cal_gard
                                            # fcost to call cal_cost
            # Simultaneous update of parameters
            w -= alpha * djdw
            b -= alpha * djdb
            if i < 100000:
                recJ.append(fcost_lgstc(X, y, w, b))

            if i % (math.ceil(iters/10)) == 0:
                print(f"Itertaion no. {i}: cost {recJ[-1]:0.3e}")
                    
        return w, b, recJ


#----------------------------------------------------------------------------------------------------
#-------------------------------k Nearest Neighbours-------------------------------------------------
#----------------------------------------------------------------------------------------------------
class Knn:
    def k(self):
        while True:
            try:
                k = int(input("Enter k: "))
                if k > 0:
                    return k
            except ValueError:
                print("Positive integer expected...")
            else:
                print("Positive integer expected...")
                continue

    def cal_dist(self, X, test):
        m, n = X.shape
        diff = X.reshape(m, n) - test.reshape(1, n)
        dist = np.sqrt(np.sum(diff**2, axis=1))  # row-wise sum

        return dist

#--------------------------------------------------------------------------------------------------------
#--------------------------------------n layer Neural Network--------------------------------------------
#--------------------------------------------------------------------------------------------------------

# One-hot encoding
"""
The idea of one-hot encoding is that you will have a vector
that is n classes long, so, however many classses you have
that is how long that vector is, and that vector is filled with
zeros except at the index of the target class where you will have 1
"""

np.random.seed(0)  # Slight changes in the values when use nnfs.init()

class Gen_layer:
    def __init__(self, n_inputs, n_neurons):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights = 0.10 * np.random.randn (n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs  # remembered for use in backward
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU: 
    def forward(self, inputs): 
        self.inputs = inputs  # remembered for use in backward
        self.output = np.maximum (0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify the original variable,
        # we make a copy of the values first so that the original gets untouched
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


# Softmax activation
class Activation_Softmax:
    # Forward pass
    def forward(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


# Cross-entropy loss
class Loss_CategoricalCrossentropy:
# Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        # samples = len(y_pred)
        samples = y_pred.shape[0]
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), np.int_(y_true)]
        # Mask values - only for one-hot encoded labels

        # For multi-label
        # elif len(y_true.shape) == 2:
        #     correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Back propagation
    def backward(self, dvalues, y_true):

        # No. of samples
        # samples = len(dvalues)
        y_true1 = copy.deepcopy(y_true)
        n_ctgrs = np.unique(y_true).shape
        samples = dvalues.shape[0]

        # for multiclass
        # One hot encoding
        # y_true = np.eye(n_ctgrs)[y_true1]
        if len(y_true.shape) == 1:
            y_true1 = np.eye(n_ctgrs)[y_true1]


        # Gradient
        # Here, inputs are the output of softmax
        # So, dvalues 
        self.dinputs = -y_true1 / dvalues

        # Taking average i.e., normalisation
        # optimizers sum all of the gradients related to each weight and bias 
        # before multiplying them by the learning rate (or some other factor).
        # the more gradient sets we’ll receive at this step, and the bigger this sum will become. 
        self.dinputs = self.dinputs / samples

class Loss(Loss_CategoricalCrossentropy):
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss




class Actvn_Softmax_Loss_CategoricalCrossentropy():
    
    # Initialising the Softmax and Loss_categoricalCrossentropy
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss()

    # Forward prop
    def forward(self, inputs, y_true):
        # Last layer activation function
        self.activation.forward(inputs)
        # Instance variable for output
        self.output = self.activation.output
        # Loss value
        return self.loss.calculate(self.output, y_true)
    
    # Back prop
    def backward (self, dvalues, y_true):  # Here, dvalues are y_hat - y_true where y_hat is a probability and y_true value is 1 corresponding to its index

        m_Xtrain = dvalues.shape[0]

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # for safety
        self.dinputs = copy.deepcopy(dvalues)

        # combined gradient
        self.dinputs[range(m_Xtrain), np.int_(y_true)] -= 1

        # Taking average for normalisation
        self.dinputs = self.dinputs / m_Xtrain

# Optimization
class SGD():
    # Initialising the optimizer with learning rate
    def __init__(self, learn_rate):
        self.learn_rate = learn_rate

    # Params uodate
    def update_p(self, layer):
        layer.weights -= self.learn_rate * layer.dweights
        layer.biases -= self.learn_rate * layer.dbiases

def neurons():
    while True:
        l = input("Number of neuron: ")
        if l.isdigit() and int(l) > 0:
            return int(l)
        else:
            print(f"Positive integer expected...")
            continue








