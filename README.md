# Student Performance Prediction with Multiple Linear Regression

This repository contains Python code implementing a **Multiple Linear Regression** model to predict student performance (Performance Index, 0–100) using the **`Student_Performance.csv`** dataset.

The model leverages **Batch Gradient Descent** for optimization and includes **data preprocessing**, **feature scaling**, and **visualization of the cost function's convergence**.

---

## Overview

### Dataset

- **File:** `Student_Performance.csv`
- **Description:** Contains student data with 6 columns (5 features + 1 target: Performance Index)
- **Note:** Ensure the dataset is placed in the project directory before running the code.
- Dataset Link: https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression/data

### Features

The model uses the following features to predict the **Performance Index**:

- Hours Studied
- Previous Scores
- Extracurricular Activities
- Sleep Hours
- Sample Question Papers Practiced

## Project Workflow

The code performs:

1. **Data Loading and Exploratory Analysis (EDA)**
    - Load the dataset using `pandas`
    - Custom function check_df() provides a full overview:
      - Dataset shape, dtypes, and preview (head/tail)
      - Check for missing values
      - Compute and visualize **correlation matrix** using a heatmap to identify feature relationships
      - Display **statistical summaries** (mean, median, quantiles)
      - Plot **histograms** for all numerical variables to examine their distributions
      - Generate **scatter plots** between each feature and the target (`Performance Index`) to visually explore linear relationships
      
<img width="600" height="400" alt="Image" src="https://github.com/user-attachments/assets/c333da2c-ab28-46a5-b6e5-86a7a8e84784" />
    
<img width="800" height="600" alt="Image" src="https://github.com/user-attachments/assets/d43eb62d-1b24-41bb-bf0a-41c73caad541" />

<img width="800" height="600" alt="Image" src="https://github.com/user-attachments/assets/02c22d5e-58e2-4b44-9249-9567425f7cc0" />
    
        
2. **Data Preprocessing**
    - Categorical encoding with `LabelEncoder`
    - Feature scaling with `StandardScaler`
    - Train-test split (67%-33%)
      
3. **Model Implementation — Batch Gradient Descent**
    - Model training using **Batch Gradient Descent**
    - Includes:
        Cost function calculation
        Weight and bias updates
        Convergence check with tolerance (epsilon)
        Iterative cost tracking
      
4. **Visualization**
    - Plotting cost function decline over iterations
      
5. **Model Evaluation**
    - After training, predictions are made on the test set and evaluated using:
        - Mean Absolute Error (MAE)
        - Mean Squared Error (MSE)
        - Root Mean Squared Error (RMSE)
        - R² Score

## Customization

You can easily adjust key parameters:

- `learning_rate` (default: `0.01`)
- `iterations` (default: `10000`)
- `epsilon` (for convergence check)
- `test_size` and `random_state` in `train_test_split`

## Requirements

- **Python 3.x**
- Required libraries:
    
    ```bash
    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn
    
    ```
    

## Results

The model successfully minimizes the cost function using batch gradient descent, demonstrating the learning process in a simple, interpretable way.

- Cost Function Convergence
The following plot shows how the cost function decreases as the model learns through batch gradient descent:
<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/110bf2e5-f39f-4e88-99e8-0351ce309dda" />

- Model Evaluation Summary
    - MAE (Mean Absolute Error): 1.6197
    - MSE (Mean Squared Error): 4.2036
    - RMSE (Root Mean Squared Error): 2.0503
    - R² Score: 0.9886

Click to see detailed metric analysis
<details>
- MAE (Mean Absolute Error): 1.6197
Meaning: On average, the model’s predictions deviate by 1.62 points from the actual values.
Interpretation: On a 0–100 scale, this error is very small — predictions are highly close to real scores.
Conclusion: The model performs with minimal errors, making its predictions reliable and practical.

- MSE (Mean Squared Error): 4.2036    
Meaning: Represents the average of squared errors, giving more weight to larger deviations.
Interpretation: A value of 4.20 indicates that large prediction errors are rare.
Conclusion: Although MSE is a technical metric, reporting it alongside RMSE highlights the depth of analysis in the project.

- RMSE (Root Mean Squared Error): 2.0503
Meaning: The square root of MSE, expressing the average error magnitude in the same unit as the target variable.
Interpretation: The model’s predictions are typically within ±2 points of the actual Performance Index values.
Conclusion: Since RMSE is close to MAE, the model is stable and not overly influenced by extreme values.

    - R² Score: 0.9886
Meaning: The model explains 98.86% of the variance in the target variable.
Interpretation: This indicates an excellent fit — the model captures the relationships between input features and student performance very effectively.
Conclusion: The R² score is the most impressive result, showing that the model has strong predictive capability.
(Note: For smaller datasets, such a high R² might also suggest potential overfitting; comparing train and test results is recommended.)
</details>

## How to Run

1. Clone this repository:
    
    ```bash
    git clone <https://github.com/ugulsumaran/Multiple-Linear-Regression-using-Batch-Gradient-Descent.git>
    
    ```
    
2. Open the project folder in Jupyter Notebook or VS Code.
3. Run all cells to see the preprocessing steps, model training, and plots.


## LICENSE 
Copyright (c) 2025 Ümmügülsüm Aran

MIT License - See [LICENSE](LICENSE) file for details
