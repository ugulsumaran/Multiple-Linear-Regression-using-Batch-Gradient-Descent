import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt
import seaborn as sns

# 1 - Read Data
data = pd.read_csv("Student_Performance.csv")
# print(data)

# 2 - EDA & Data Preprocessing
from IPython.display import display

def check_df(dataframe, target=None):
    print("############################")
    print("DATASET OVERVIEW")
    print("############################\n")

    # Basic Info
    print("##### Shape #####")
    print(dataframe.shape)
    
    print("\n##### Data Types #####")
    print(dataframe.dtypes)
    
    print("\n##### First 5 Rows #####")
    display(dataframe.head(5))
    
    print("\n##### Last 5 Rows #####")
    display(dataframe.tail(5))

    # Missing Values
    print("\n##### Missing Values #####")
    print(dataframe.isnull().sum())

check_df(data, target="Performance Index")

## Encode Categorical Values
le = LabelEncoder()

### Convert categorical column to numeric
data['Extracurricular Activities'] = le.fit_transform(data['Extracurricular Activities'])

# EDA in Progress
def check_df(dataframe, target=None):
    # Correlation
    print("\n##### Correlation Matrix #####")
    display(dataframe.corr())

    # Descriptive Stats
    print("\n##### Descriptive Statistics #####")
    display(dataframe.describe([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T)

    # Correlation Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("correlation_heatmap.png")
    plt.show()
    
    # Histograms
    print("\n############################")
    print("Variable Distributions")
    print("############################")
    data.hist(figsize=(12, 8), bins=20, color='skyblue', edgecolor='black')
    plt.suptitle("Distribution of Variables (Histogram)", size=14)
    plt.savefig("Distribution.png")
    plt.show()
    
    # Scatter plots (if target exists)
    if target and target in data.columns:
        print("\n############################")
        print(f"Relationship with '{target}'")
        print("############################")
        
        features = [col for col in data.columns if col != target]
        plt.figure(figsize=(15, 8))
        for i, col in enumerate(features, 1):
            plt.subplot(2, 3, i)
            sns.scatterplot(x=data[col], y=data[target], alpha=0.6)
            plt.title(f"{col} vs {target}")
        plt.tight_layout()
        plt.show()

check_df(data, target="Performance Index")

# 3 - Features (X) and Target (y)
X = data.drop(columns=["Performance Index"])

## - Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

## Target variable
y = data[["Performance Index"]]


# 4 - Split into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.33, random_state=1
)

## Convert targets to numpy arrays
y_train = y_train.values
y_test = y_test.values

# 5 - Batch Gradient Descent
def batch_gradient_descent(X, y, learning_rate=0.01, iterations=550, epsilon=1e-6):
    m, n = X.shape  # m: number of samples, n: number of features
    
    b = 0.0  # initial bias
    w = np.zeros((n, 1))  # initial weights
    cost_history = []

    for i in range(iterations):
        # Prediction: h = Xw + b
        h = X.dot(w) + b
        error = h - y

        # Cost function
        cost = (1 / (2 * m)) * np.sum(error ** 2)
        cost_history.append(cost)

        # Gradients
        db = (1 / m) * np.sum(error)
        dw = (1 / m) * X.T.dot(error)

        # Update parameters
        w_new = w - learning_rate * dw
        b_new = b - learning_rate * db

        # Convergence check
        if np.linalg.norm(w_new - w) < epsilon and abs(b_new - b) < epsilon:
            print(f"Converged at iteration {i + 1}.")
            w, b = w_new, b_new
            break

        # Update weights and bias
        w, b = w_new, b_new

    return b, w, cost_history

# Call the function
b_final, w_final, cost_history = batch_gradient_descent(
    X_train, y_train.reshape(-1, 1), learning_rate=0.01, iterations=10000
)

print("Final bias (b):", b_final)
print("Final weights (w):", w_final.ravel())
print("Final cost:", cost_history[-1])

# 6 - Plot Cost Function
plt.plot(range(len(cost_history)), cost_history, "b-")
plt.xlabel("Iteration")
plt.ylabel("Cost J(θ)")
plt.title("Cost Function Decrease Over Iterations")
plt.grid(True)
plt.savefig("cost_function_plot.png")
plt.show()

# 7 - Metrics

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Make predictions on the test set using the learned parameters
y_pred = X_test.dot(w_final) + b_final

# Calculate regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print metrics
print("\nModel Evaluation Metrics on Test Set")
print("--------------------------------------")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Predictions vs. Actual Values scatter plot
plt.figure(figsize=(6,6))
sns.scatterplot(x=y_test.squeeze(), y=y_pred.ravel(), alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # diagonal line
plt.xlabel("Actual Values")
plt.ylabel("Predictions")
plt.title("Predictions vs. Actual Values")
plt.legend(["Actual Values", "Predictions"])
plt.savefig("Predictions_vs_Actual.png")
plt.show()
