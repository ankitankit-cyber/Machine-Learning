Diabetes Prediction Using Machine Learning

This project demonstrates a machine learning approach to predict diabetes based on medical data. It utilizes Python and common libraries like NumPy, Pandas, and Scikit-learn to preprocess data, train a model, and evaluate its accuracy.

Project Overview

The objective of this project is to create a system that predicts whether a person has diabetes based on features such as glucose level, blood pressure, BMI, and more. The dataset used is the PIMA Indian Diabetes dataset.

Key Features

Data Preprocessing: Scaling and splitting the dataset into training and testing sets.

Model Training: Using Support Vector Machines (SVM).

Performance Evaluation: Measuring model accuracy on training and testing data.

Interactive Prediction System: Allowing user input for predictions.

Steps in the Notebook

1. Importing Libraries

The following libraries are imported:

numpy: For numerical operations.

pandas: For data manipulation and analysis.

train_test_split and accuracy_score: For splitting data and measuring model performance.

StandardScaler and svm: For data preprocessing and model training.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

2. Loading and Analyzing Data

The dataset is loaded into a Pandas DataFrame. Basic statistics and structural information about the data are displayed to understand its features.

diabetes_dataset = pd.read_csv('diabetes.csv')
print(diabetes_dataset.describe())
print(diabetes_dataset.info())

3. Data Preprocessing

Separating Features and Labels: Independent variables (X) and the target variable (Outcome) are separated.

Standardization: Features are standardized to ensure uniform scaling for better model performance.

x = diabetes_dataset.drop(columns='Outcome', axis=1)
y = diabetes_dataset['Outcome']
scaler = StandardScaler()
standard_data = scaler.fit_transform(x)
x = standard_data
y = diabetes_dataset['Outcome']

4. Splitting Data

The dataset is split into training and testing subsets (80-20 split) using train_test_split.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

5. Training the Model

A Support Vector Machine (SVM) with a linear kernel is used to train the model on the training dataset.

classifier = SVC(kernel='linear')
classifier.fit(x_train, y_train)

6. Evaluating the Model

The accuracy of the model is calculated on both the training and testing datasets.

x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print("Training Accuracy:", training_data_accuracy)

x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print("Test Accuracy:", test_data_accuracy)

7. Making Predictions

An interactive prediction system is implemented, allowing users to input data for prediction.

input_data = (4,110,92,0,0,37.6,0.191,30)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)

prediction = classifier.predict(std_data)
if prediction[0] == 0:
    print("The person is not diabetic")
else:
    print("The person is diabetic")

How to Run the Project

Clone the Repository:

git clone <repository-url>

Install Required Libraries:

pip install numpy pandas scikit-learn

Open the Jupyter Notebook:

jupyter notebook Diabetes_prediction.ipynb

Execute the Notebook:
Run the notebook step-by-step to train the model and test predictions.

Results

Training Accuracy: 78.66%

Testing Accuracy: 77.27%
