#First commit - Loading in the Fashion MNIST dataset
import pandas as pd
import numpy as np

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

