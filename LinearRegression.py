import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_data_split(dataFrame):
    trainX = dataFrame.iloc[:, :-1]  # Features (all columns except the last one)
    trainY = dataFrame.iloc[:, -1]  # Target (last column)
    return trainX.values, trainY.values


df = pd.read_csv("iceCream.csv")

trainX, trainY = load_data_split(df)


def normalize_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


trainX = normalize_data(trainX)

LR = 0.06
w = 0
b = 0


def model(x):
    return w * x + b


m = len(trainX)


def squared_error_function_w(m, trainX, trainY):
    error = 0
    for i in range(m):
        error += (model(trainX[i]) - trainY[i]) * trainX[i]
    return error / m


def squared_error_function_b(m, trainX, trainY):
    error = 0
    for i in range(m):
        error += model(trainX[i]) - trainY[i]
    return error / m


def gradientDescent(m, trainX, trainY):
    global w, b
    tmp_w = w - LR * squared_error_function_w(m, trainX, trainY)
    tmp_b = b - LR * squared_error_function_b(m, trainX, trainY)

    w = tmp_w
    b = tmp_b


# train linear regression
for i in range(2000):  # Iterate more times for better convergence
    gradientDescent(m, trainX, trainY)

print(w, b)

# Visualize the trained model
plt.scatter(trainX, trainY, color="blue", label="Data")
plt.plot(trainX, model(trainX), color="red", label="Linear Regression")
plt.xlabel("Temperature")
plt.ylabel("Ice Cream Sales")
plt.legend()
plt.show()
