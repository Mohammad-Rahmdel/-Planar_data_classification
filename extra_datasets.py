import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}


# (choose your dataset)
dataset = "gaussian_quantiles"
X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])

# make blobs binary
if dataset == "blobs":
    Y = Y%2

# Visualize the data
plt.scatter(X[0, :], X[1, :], c=Y.ravel(), s=40, cmap=plt.cm.Spectral);
plt.show()



def layer_sizes(X, Y):
    n_x = X.shape[0] 
    n_h = 4          
    n_y = Y.shape[0] 
    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):  
    b1 = np.zeros((n_h,1))
    b2 = np.zeros((n_y,1))
    W1 = np.random.randn(n_h, n_x) * 0.01
    W2 = np.random.randn(n_y, n_h) * 0.01
    return b1, W1, b2, W2



def ReLu(x):
    return np.maximum(0, x)


def forward_propagation(X, b1, W1, b2, W2):
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    # A1 = ReLu(Z1)   # using ReLu as the activation function.
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)

    return Z1, A1, Z2, A2
    


def compute_cost(A2, Y, W1, W2):
    m = Y.shape[1] 
    cost = (-1/m) * ( np.dot(np.log(A2),Y.T) + np.dot(np.log(1 - A2),(1 - Y).T) )
    cost = np.squeeze(cost)     
    return cost


def reluDerivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x


def backward_propagation(b1, W1, b2, W2, Z1, A1, Z2, A2, X, Y):
    m = X.shape[1]
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2)) # tanh(Z1) = A1  (tanh()).prime = 1 - tanh()**2
    # dZ1 = np.dot(W2.T, dZ2) * reluDerivative(Z1)  #f or relu
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate=1.2):
    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2
    return W1, b1, W2, b2


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    np.random.seed(3)
    n_x, _, n_y = layer_sizes(X, Y)
    b1, W1, b2, W2 = initialize_parameters(n_x, n_h, n_y)
    for i in range(num_iterations):
        Z1, A1, Z2, A2 = forward_propagation(X, b1, W1, b2, W2)
        dW1, db1, dW2, db2 = backward_propagation(b1, W1, b2, W2, Z1, A1, Z2, A2, X, Y)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate=1.2)
        if print_cost:
            if i%1000==0:
                 print ("Cost after iteration %i: %f" % (i, compute_cost(A2, Y, W1, W2)))
    
    return W1, b1, W2, b2


def predict(W1, b1, W2, b2, X):
    _, _, _, A2 = forward_propagation(X, b1, W1, b2, W2)
    Y_hat = np.round(A2)
    return Y_hat



W1, b1, W2, b2 = nn_model(X, Y, n_h = 4, num_iterations=10000, print_cost=True)
plot_decision_boundary(lambda x: predict( W1, b1, W2, b2, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()

predictions = predict(W1, b1, W2, b2, X)
print ('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')







# plt.figure(figsize=(16, 32))
# hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
# for i, n_h in enumerate(hidden_layer_sizes):
#     plt.subplot(5, 2, i + 1)
#     plt.title('Hidden Layer of size %d' % n_h)
#     W1, b1, W2, b2 = nn_model(X, Y, n_h, num_iterations=5000)
#     plot_decision_boundary(lambda x: predict(W1, b1, W2, b2, x.T), X, Y)
#     predictions = predict(W1, b1, W2, b2, X)
#     accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
#     print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
# plt.show()
# """
# Interpretation:
#     The larger models (with more hidden units) are able to fit the training set better,
#     until eventually the largest models overfit the data.
#     The best hidden layer size seems to be around n_h = 5. Indeed, a value around here seems to
#     fits the data well without also incurring noticable overfitting.
#     You will also learn later about regularization, which lets you use very large models (such as n_h = 50)
#     without much overfitting.
# """





# using ReLu function accuracy results :
# Accuracy for 1 hidden units: 63.74999999999999 %
# Accuracy for 2 hidden units: 63.74999999999999 %
# Accuracy for 3 hidden units: 61.25000000000001 %
# Accuracy for 4 hidden units: 70.75 %
# Accuracy for 5 hidden units: 68.25 %
# Accuracy for 20 hidden units: 76.0 %
# Accuracy for 50 hidden units: 82.75 %