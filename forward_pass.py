import pandas as pd
import numpy as np

################################################################
#                                                              #
#                     SPLITTING DATA                           #
#                                                              #
################################################################
#Read the data into a dataframe
filepath = "C:\\Fashion_MNIST\\"
data = pd.read_csv(filepath + "fashion_mnist.csv", low_memory=False)

#Convert the dataframe to a numpy array
data = data.to_numpy()
m, n = data.shape
print("Shape of data: ", m, "x", n)

#Split the data into testing and training sets

#shuffle the data
np.random.shuffle(data)

#80% for train data and 20% for test data 
train_percent = int(m * 0.80)

#row, col
y_train = data[:train_percent, 0]
X_train = data[:train_percent, 1:n]
y_test = data[train_percent:m, 0]
X_test = data[train_percent:m, 1:n]

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
hidden_layer_1 = 128
hidden_layer_2 = 64
output_layer = len(labels)

#initialize random weights for each layer
W1 = np.random.randn(input_layer, hidden_layer_1)
W2 = np.random.randn(hidden_layer_1, hidden_layer_2)
W3 = np.random.randn(hidden_layer_2, output_layer)

#Initialize biases to zero for each layer
b1 = np.zeros(hidden_layer_1)
b2 = np.zeros(hidden_layer_2)
b3 = np.zeros(output_layer)

################################################################
#                                                              #
#                     Forward Pass                             #
#                                                              #
################################################################

def relu(x):
    if x < 0:
        return 0
    else:
        return x
    
#Convert output vector into probabilities
def softmax(x):
    e_x = np.exp(x)
    sum_e_x = np.sum(e_x)

    return e_x / sum_e_x

#Convert y_train into a vector where only the correct label appears
def one_hot(y):
    one_hot = np.zeros((y.size, y.max() + 1)) #size, col: finds 9 + 1 and makes 10 columns
    one_hot[np.arange(y.size), y] = 1 #Selects the row and places the 1 in the correct index

    return one_hot

#Calculate loss by comparing the predictions to the ground truth
def cross_entropy(y, p):
    return -np.sum(y * np.log(p))

loss = 0



#Loop over all training images
for i in range(X_train.shape[0]):
    #Use relu activation function for forward pass
    A0 = X_train[i, :]
    # Make sure A0 is a NumPy array
    print(type(A0))

    # Check the shapes of A0, W1, and b1
    print("A0 shape:", A0.shape)
    print("W1 shape:", W1.shape)
    print("b1 shape:", b1.shape)

    Z1 = np.dot(A0, W1) + b1

    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2

    A2 = relu(Z2)
    Z3 = np.dot(A2, W3) + b3

    #Use softmax for output layer
    A3 = softmax(Z3)

    #one hot encode y_train to compare it to the networks predictions
    y_train_encoded = one_hot(y_train)

    #Compute the loss of the network using cross entropy loss
    loss += cross_entropy(y_train_encoded, A3)

print(loss)




