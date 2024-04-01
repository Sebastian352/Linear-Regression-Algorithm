import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data_split(dataframe):
    # Calculate the number of rows for training and testing
    num_rows = len(dataframe)
    num_train_rows = int(num_rows * 0.95)
    num_test_rows = num_rows - num_train_rows

    # Split the dataframe into training and testing sets
    train_df = dataframe.iloc[:num_train_rows]
    test_df = dataframe.iloc[num_train_rows:]

    # Extract features and target for training and testing sets
    trainX = train_df.drop(columns=["charges"])
    trainY = train_df["charges"]
    testX = test_df.drop(columns=["charges"])
    testY = test_df["charges"]

    return trainX.values, trainY.values, testX.values, testY.values


df = pd.read_csv("medical_insurance.csv")

df_encoded = pd.get_dummies(df, columns=["sex", "region", "smoker"])
move = df_encoded.pop("charges")
df_encoded.insert(df_encoded.shape[1], "charges", move)

trainX, trainY, testX, testY = load_data_split(df_encoded)


def normalize_data(data):
    normalized_data = np.zeros(data.shape)

    for i in range(len(data)):
        min_val = np.min(data[i])
        max_val = np.max(data[i])
        normalized_data[i] = (data[i] - min_val) / (max_val - min_val)
    return normalized_data


def normalize_y(data):
    normalized_data = np.zeros(data.shape)
    for i in range(len(data)):
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data[i] = (data[i] - min_val) / (max_val - min_val)
    return normalized_data


trainX = normalize_data(trainX)
trainY = normalize_y(trainY)
testX = normalize_data(testX)
testY = normalize_y(testY)

LR = 0.01
w = np.zeros(df_encoded.shape[1] - 1)
b = 0


def model(x):
    result = np.dot(w, x) + b
    return result


m = len(trainX)


def squared_error_function_w(m, trainX, trainY, j):
    error = 0
    for i in range(m):
        error += (model(trainX[i]) - trainY[i]) * trainX[i][j]
    return error / m


def squared_error_function_b(m, trainX, trainY):
    error = 0
    for i in range(m):
        error += model(trainX[i]) - trainY[i]
    return error / m


def gradientDescent(m, trainX, trainY):
    global w, b
    tmp_w = np.zeros(trainX.shape[1])

    for i in range(df_encoded.shape[1] - 1):
        tmp_w = np.subtract(w, LR * squared_error_function_w(m, trainX, trainY, i))
    tmp_b = b - LR * squared_error_function_b(m, trainX, trainY)
    w = tmp_w
    b = tmp_b


for i in range(2000):  # Iterate more times for better convergence
    gradientDescent(m, trainX, trainY)

print(w, b)
predictions = np.dot(testX, w) + b
# Calculate Mean Absolute Error (MAE)

print(predictions[0], testY[0])

mae = mean_absolute_error(testY, predictions)
print("Mean Absolute Error:", mae)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(testY, predictions)
print("Mean Squared Error:", mse)

# Calculate R-squared (R2) score
r2 = r2_score(testY, predictions)
print("R-squared Score:", r2)

# Plot the results
plt.scatter(testY, predictions, color="blue")
plt.plot(
    [testY.min(), testY.max()],
    [testY.min(), testY.max()],
    linestyle="--",
    color="red",
    linewidth=2,
)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs. Predicted Charges")
plt.show()
