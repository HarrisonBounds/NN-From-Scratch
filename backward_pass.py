import pandas as pd
import numpy as np

################################################################
#                                                              #
#                     SPLITTING DATA                           #
#                                                              #
################################################################
#Read the data into a dataframe
filepath = "E:\\Fashion_MNIST\\"
train_data = pd.read_csv(filepath + "fashion-mnist_train.csv")
test_data = pd.read_csv(filepath + "fashion-mnist_test.csv")

#Convert the dataframe to a numpy array
train_data = np.array(train_data)
test_data = np.array(test_data)

print(train_data.dtype)
print(test_data.dtype)


#Split the data into testing and training sets

#shuffle the data
np.random.shuffle(train_data)
m_train, n_train = train_data.shape
np.random.shuffle(test_data)
m_test, n_test = test_data.shape

print(f'Shape of training data: {m_train}x{n_train}')
print(f'Shape of testing data: {m_test}x{n_test}')

X_train = train_data[:, 1:n_train]
y_train = train_data[:, 0]
X_test = test_data[:, 1:n_test]
y_test = test_data[:, 0]


print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)
print("X_train type: ", X_train.dtype)
print("y_train type: ", y_train.dtype)
print("X_test type: ", X_test.dtype)
print("y_test type: ", y_test.dtype)

print("DONE")

################################################################
#                                                              #
#                     Build NN                                 #
#                                                              #
################################################################

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


#Define number of neurons in each layer
input_layer = X_train.shape[1]
hidden_layer = 64
output_layer = len(labels)

#initialize random weights for each layer
W1 = np.random.randn(input_layer, hidden_layer)
W2 = np.random.randn(hidden_layer, output_layer)

#Initialize biases to zero for each layer
b1 = np.zeros(hidden_layer)
b2 = np.zeros(output_layer)

print(f'W1 shape: {W1.shape}')
print(f'W2 shape: {W2.shape}')


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

loss = 0

#Use relu activation function for forward pass
A0 = X_train[0]

Z1 = np.dot(A0, W1) + b1

A1 = relu(Z1)

Z2 = np.dot(A1, W2) + b2

A2 = softmax(Z2)

#one hot encode y_train to compare it to the networks predictions
y_train_encoded = one_hot(y_train)

#Compute the loss of the network using cross entropy loss
loss = cross_entropy(y_train_encoded, A2)

print("Loss:", loss)

print("A0 shape: ", A0.shape)
print("Z1 shape: ", Z1.shape)
print("A1 shape: ", A1.shape)
print("Z2 shape: ", Z2.shape)
print("A2 shape: ", A2.shape)


################################################################
#                                                              #
#                     Backwards Pass                           #
#                                                              #
################################################################



