import copy
import math
import numpy as np

#----------------------------------------------------------------------------------------------------------
#---------------------------------------------Linear Regression--------------------------------------------
#----------------------------------------------------------------------------------------------------------

# Find w and b ---> Gradient desecnt
# Calculate gradient
def cal_grad(X, y, w, b):

    m, n = X.shape
    fwb = (np.matmul(X.reshape(m,n), w.reshape(n,1)) + b)  # using reshape to make desired order and to make more readeable the orders of matrix
    diff = fwb - y.reshape(m,1)

    djdw = np.matmul(diff.reshape(1, m), X)
    djdb = np.sum(diff)

    return (djdw / m), (djdb / m)


# Calculate cost
def cal_cost(X, y, w1, b1):

    m, n = X.shape
    fwb = (np.matmul(X.reshape(m,n), w1.reshape(n,1)) + b1)
    diff = fwb - y.reshape(m,1)
    acst = np.sum(diff**2)/(2*m)  # Average cost to get a smaller value

    return acst


# Gradient Descent
def find_wb(X, y, winit, binit, alpha, num_iters, fcost, fgrad):
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
            print(f"Itertaion no. {i}: cost {recJ[-1]:0.4e}")
        
        # Last iterations to check variation
        if (num_iters - 10) <= i <= (num_iters - 1):
            print(f"Itertaion no. {i}: cost {recJ[-1]:0.3e}")  

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

# Find w and b ---> Gradient desecnt
# Calculate gradient
def p_cal_grad(X, y, w, b, a):

    m, n = X.shape
    fwb = (np.matmul(X.reshape(m,a), w.reshape(a,1)) + b)  # using reshape to make desired order and to make more readeable the orders of matrix
    diff = fwb - y.reshape(m,1)

    djdw = np.matmul(diff.reshape(1, m), X)
    djdb = np.sum(diff)

    return (djdw / m), (djdb / m)


# Calculate cost
def p_cal_cost(X, y, w1, b1, a):

    m, n = X.shape
    fwb = (np.matmul(X.reshape(m,a), w1.reshape(a,1)) + b1)
    diff = fwb - y.reshape(m,1)
    acst = np.sum(diff**2)/(2*m)  # Average cost to get a smaller value

    return acst


# Gradient Descent
def p_find_wb(X, y, winit, binit, alpha, num_iters, fcost, fgrad, a):
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
            print(f"Itertaion no. {i}: cost {recJ[-1]:0.4e}")
        
        # Last iterations to check variation
        if (num_iters - 10) <= i <= (num_iters - 1):
            print(f"Itertaion no. {i}: cost {recJ[-1]:0.3e}")  

    return w0, b0, recJ



#----------------------------------------------------------------------------------------------------------------
#---------------------------------------Logistic Regression------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------

# Logistic Cost
def cal_cost_lgstc(X, y, w, b):
    m, n = X.shape

    z = (np.matmul(X.reshape(m,n), w.reshape(n,1)) + b)
    fwb = 1/(1 + np.exp(-z))
    cost = -np.matmul(y.reshape(1, m), np.log(fwb)) - np.matmul((1 - y.reshape(1, m)), (np.log(1 - fwb)))
    t_cost = np.sum(cost)  # total cost

    return t_cost/m

# Gradient
def cal_grad_lgstc(X, y, w, b):
    m, n = X.shape
    z = (np.matmul(X.reshape(m,n), w.reshape(n,1)) + b)   # z ---> m x 1
    # element wise sigmoid
    fwb = 1/(1 + np.exp(-z))

    diff = fwb - y.reshape(m, 1)  # m x 1
    djdw = np.matmul(diff.reshape(1, m), X)  # djdw ---> 1 x n
    djdb = np.sum(diff)
    
    return djdw/m , djdb/m

# Gradient Descent
def find_wb_lgstc(X, y, winit, binit, alpha, iters, fcost_lgstc, fgrad_lgstc):
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

def k():
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

def cal_dist(X, test):
    m, n = X.shape
    diff = X.reshape(m, n) - test.reshape(1, n)
    dist = np.sqrt(np.sum(diff**2, axis=1))  # row-wise sum

    return dist









