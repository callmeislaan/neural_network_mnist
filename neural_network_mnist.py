import numpy as np
import os
import gzip
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

## load data
def load_data(path, kind = 'train'):
    img_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)
    lbl_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)

    with gzip.open(img_path) as img:
        images = np.frombuffer(img.read(), dtype = 'uint8', offset = 16).reshape(-1, 28*28)
    with gzip.open(lbl_path) as lbl:
        labels = np.frombuffer(lbl.read(), dtype = 'uint8', offset = 8)
    return images, labels

X_train, y_train = load_data('Data/mnist/')
X_test, y_test = load_data('Data/mnist', kind = 't10k')


## one-hot coding
from scipy import sparse
def convert_labels(y, C = 10):
    Y = sparse.coo_matrix((np.ones_like(y), (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y

Y_train = convert_labels(y_train)
Y_test = convert_labels(y_test)
X_train = X_train.T
X_test = X_test.T



# N = 1000 # number of points per class
# d0 = 2 # dimensionality
# C = 3 # number of classes
# X = np.zeros((d0, N*C)) # data matrix (each row = single example)
# y = np.zeros(N*C, dtype='uint8') # class labels

# for j in range(C):
#   ix = range(N*j,N*(j+1))
#   r = np.linspace(0.0,1,N) # radius
#   t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
#   X[:,ix] = np.c_[r*np.sin(t), r*np.cos(t)].T
#   y[ix] = j

# Y = convert_labels(y, C = 3)


def softmax(Z):
    e_Z = np.exp(Z - np.max(Z, axis = 0, keepdims=True))
    return e_Z / np.sum(e_Z, axis = 0)

def cost(Yhat, Y, W1, W2, W3, lamda):
    return -np.sum(Y*np.log(Yhat))/Y.shape[1] + lamda*(np.linalg.norm(W1)**2 + np.linalg.norm(W2)**2 + np.linalg.norm(W3)**2)

def mlp_mnist_1hiddenlayer(X, Y, W1, W2, W3, b1, b2, b3, eta = 1, lamda = 1e-5):
    N = X.shape[1]
    max_count = 20
    loss = []
    mini_batch_size = 100
    for _ in tqdm(range(max_count)):
        mix_id = np.random.permutation(N)
        i = 0
        while i < N:
            true_id = mix_id[i:i+mini_batch_size]
            Xi = X[:, true_id]
            Yi = Y[:, true_id]
            ## feed for ward
            Z1 = W1.T.dot(Xi) + b1
            A1 = np.maximum(Z1, 0)
            Z2 = W2.T.dot(A1) + b2
            A2 = np.maximum(Z2, 0)
            Z3 = W3.T.dot(A2) + b3
            A3 = softmax(Z3)

            if i % 1000 == 0:
                loss.append(cost(A3, Yi, W1, W2, W3, lamda))
                if loss[-1] < 1e-5:
                    return W1, W2, W3, b1, b2, b3, loss

            # backpropagation
            E3 = (A3 - Yi)/A3.shape[1]
            dW3 = A2.dot(E3.T)+ lamda*W3
            db3 = np.sum(E3, axis = 1, keepdims = True)
            E2 = W3.dot(E3)
            E2[Z2 <= 0] = 0
            dW2 = A1.dot(E2.T) + lamda*W2
            db2 = np.sum(E2, axis = 1, keepdims = True)
            E1 = W2.dot(E2)
            E1[Z1 <= 0] = 0
            dW1 = Xi.dot(E1.T) + lamda*W1
            db1 = np.sum(E1, axis = 1, keepdims = True)
        
            # update weights and bias
            W1 += -eta*dW1
            b1 += -eta*db1
            W2 += -eta*dW2
            b2 += -eta*db2
            W3 += -eta*dW3
            b3 += -eta*db3

            i += mini_batch_size
    return W1, W2, W3, b1, b2, b3, loss

def predict(X, W1, W2, W3, b1, b2, b3):
    Z1 = W1.T.dot(X) + b1
    A1 = np.maximum(Z1, 0)
    Z2 = W2.T.dot(A1) + b2
    A2 = np.maximum(Z2, 0)
    Z3 = W3.T.dot(A2) + b3
    A3 = softmax(Z3)
    return np.argmax(A3, axis = 0)

d0 = X_train.shape[0]
# d0 = X.shape[0]
d1 = 100
d2 = 100
# d3 = 3
d3 = 10
W1 = 0.01*np.random.randn(d0, d1)
W2 = 0.01*np.random.randn(d1, d2)
W3 = 0.01*np.random.randn(d2, d3)
b1 = np.zeros((d1, 1))
b2 = np.zeros((d2, 1))
b3 = np.zeros((d3, 1))

# mlp = MLPClassifier(max_iter=20, verbose=True).fit(X_train.T, y_train)
# y_pred_test = mlp.predict(X_test.T)
# y_pred_train = mlp.predict(X_train.T)

# X_train = X
# Y_train = Y
# y_train = y
W1, W2, W3, b1, b2, b3, loss = mlp_mnist_1hiddenlayer(X_train, Y_train, W1, W2, W3, b1, b2, b3)
y_pred_test = predict(X_test, W1, W2, W3, b1, b2, b3)
y_pred_train = predict(X_train, W1, W2, W3, b1, b2, b3)
print('test:  ', accuracy_score(y_test, y_pred_test)*100)
print('train: ', accuracy_score(y_train, y_pred_train)*100)
plt.plot(loss)
plt.show()
