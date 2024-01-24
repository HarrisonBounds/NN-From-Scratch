import pandas as pd
import numpy as np


def split(train_data, test_data):
    
    #Convert the dataframe to a numpy array
    train_data = np.array(train_data)
    test_data = np.array(test_data)

    #shuffle the data
    np.random.shuffle(train_data)
    m_train, n_train = train_data.shape
    np.random.shuffle(test_data)
    m_test, n_test = test_data.shape

    X_train = train_data[:, 1:n_train]
    y_train = train_data[:, 0]
    X_test = test_data[:, 1:n_test]
    y_test = test_data[:, 0]

    X_train = X_train / 255
    X_test = X_test / 255

    return X_train, y_train, X_test, y_test


################################################################
#                                                              #
#                     Build NN                                 #
#                                                              #
################################################################

def init(il, hl, ol):

    #initialize random weights for each layer
    W1 = np.random.randn(il, hl)
    W2 = np.random.randn(hl, ol)

    #Initialize biases to zero for each layer
    b1 = np.zeros(hl)
    b2 = np.zeros(ol)

    return W1, b1, W2, b2


################################################################
#                                                              #
#                     Forward Pass                             #
#                                                              #
################################################################

def relu(x):
    return np.maximum(0, x)
    
#Convert output vector into probabilities
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

#Convert y_train into a vector where only the correct label appears
def one_hot(y):
    one_hot = np.zeros((y.size, y.max() + 1)) #size, col: finds 9 + 1 and makes 10 columns
    one_hot[np.arange(y.size), y] = 1 #Selects the row and places the 1 in the correct index

    return one_hot

#Calculate loss by comparing the predictions to the ground truth
def cross_entropy(y, p):
    # Add a small epsilon to prevent division by zero and take the negative log
    epsilon = 1e-10
    return -np.sum(y * np.log(p + epsilon))

def forward(X, W1, b1, W2, b2):
    A0 = X

    Z1 = np.dot(A0, W1) + b1

    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2

    A2 = softmax(Z2)
    print(f'Shape of W1: {W1.shape}')
    print(f'Shape of b1: {b1.shape}')
    print(f'Shape of W2: {W2.shape}')
    print(f'Shape of b2: {b2.shape}')

    print(f'Shape of Z1: {Z1.shape}')
    print(f'Shape of A1: {A1.shape}')
    print(f'Shape of Z2: {Z2.shape}')
    print(f'Shape of A2: {A2.shape}')


    return Z1, A1, Z2, A2

################################################################
#                                                              #
#                     Backwards Pass                           #
#                                                              #
################################################################


#dL     =     dL     *     dA2     *      dZ2
#dW2          dA2          dZ2            dW2

def backward(A0, A1, A2, Z1, W2, y):
    #dL_A2 * dA2_dZ2 can be written as dL_dZ2
    #dL_dZ2 = A2 - y for softmax and categorical cross entropy 
      
    dL_dZ2 = A2 - y
    print(f'Shape of dL_dZ2: {dL_dZ2.shape}')

    dZ2_dW2 = A1

    dL_dW2 = np.dot(dZ2_dW2.T, dL_dZ2)
    print(f'Shape of dL_dW2: {dL_dW2.shape}')
    dL_db2 = np.sum(dL_dZ2)

    dL_dA1 = np.dot(dL_dZ2, W2.T)

    #relu derivative
    dA1_dZ1 = np.where(Z1 > 0, 1, 0)

    dZ1_dW1 = A0

    dL_dZ1 = dL_dA1 * dA1_dZ1

    dL_dW1 = np.dot(dL_dZ1.T, dZ1_dW1)
    dL_db1 = np.sum(dL_dZ1)

    return dL_dW2, dL_db2, dL_dW1, dL_db1


def update(dL_dW2, dL_db2, dL_dW1, dL_db1, W1, b1, W2, b2, lr):
    #Update the weights and biases of the second layer
    W2 = W2 - lr * dL_dW2
    b2 = b2 - lr * dL_db2

    #Update first layer 
    W1 = W1 - lr * dL_dW1.T
    b1 = b1 - lr * dL_db1

    return W1, b1, W2, b2


def main():
    #Labels
    labels = {
        0: 't-shirt/top',
        1: 'trouser',
        2: 'pullover',
        3: 'dress',
        4: 'coat',
        5: 'sandal',
        6: 'shirt',
        7: 'sneaker',
        8: 'bag',
        9: 'ankle boot'
    }

    #Read the data into a dataframe
    filepath = "C:\\Fashion_MNIST\\"
    train_data = pd.read_csv(filepath + "fashion-mnist_train.csv")
    test_data = pd.read_csv(filepath + "fashion-mnist_test.csv")

    learning_rate = 0.01

    X_train, y_train, X_test, y_test = split(train_data, test_data)

    #one hot encode y_train to compare it to the networks predictions
    y_train_encoded = one_hot(y_train)
    y_test_encoded = one_hot(y_test)

    #Define number of neurons in each layer
    input_layer = X_train.shape[1]
    hidden_layer = 24
    output_layer = len(labels)

    W1, b1, W2, b2 = init(input_layer, hidden_layer, output_layer)

    for i in range(100):

        Z1, A1, Z2, A2 = forward(X_train, W1, b1, W2, b2)

        dL_dW2, dL_db2, dL_dW1, dL_db1 = backward(X_train, A1, A2, Z1, W2, y_train_encoded)

        loss = cross_entropy(y_train_encoded, A2)
        print(f'Loss for Iteration {i+1}: {loss}')

        W1, b1, W2, b2 = update(dL_dW2, dL_db2, dL_dW1, dL_db1, W1, b1, W2, b2, learning_rate)


main()




















