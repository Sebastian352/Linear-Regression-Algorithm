## Linear Regression for Ice Cream Sales Prediction

This Python script implements a simple linear regression model to predict ice cream sales based on temperature. The dataset used for training is provided in a CSV file named "iceCream.csv".

### Requirements

Make sure you have the following libraries installed:

- pandas
- matplotlib
- numpy

You can install them via pip:

```
pip install pandas matplotlib numpy
```

### Usage

1. Clone the repository or download the script `ice_cream_sales_prediction.py` to your local machine.
2. Place the CSV file "iceCream.csv" in the same directory as the script.
3. Run the script using Python:

```
python ice_cream_sales_prediction.py
```

### Code Overview

- `load_data_split`: Function to load and split the dataset into features (temperature) and target (ice cream sales).
- `normalize_data`: Function to normalize the input features to a range between 0 and 1.
- `model`: Linear regression model equation (`y = w * x + b`).
- `squared_error_function_w`: Function to compute the squared error with respect to the weight `w`.
- `squared_error_function_b`: Function to compute the squared error with respect to the bias `b`.
- `gradientDescent`: Function to perform gradient descent optimization to update `w` and `b`.
- Training the linear regression model with gradient descent for 2000 iterations.
- Visualizing the trained model with a scatter plot of temperature vs. ice cream sales and the linear regression line.

### Parameters

- Learning Rate (LR): 0.06
- Number of Iterations: 2000

### Output

The script outputs the optimal values of `w` and `b` after training the model, as well as a scatter plot visualizing the dataset and the linear regression line.

### Notes

- Adjust the LR and the number of iterations as needed for better convergence and accuracy.
- Ensure that the CSV file "iceCream.csv" contains the appropriate data columns (temperature and ice cream sales) for training the model.

Feel free to modify the script or dataset according to your requirements. Happy predicting!
