# Task_1
Multi-Class Classification Pipeline (Iris Dataset)
📌 Project Overview
This project implements an end-to-end supervised machine learning classification pipeline using the classic Iris dataset. The notebook demonstrates data ingestion, exploratory analysis, preprocessing, model training, performance evaluation, and decision boundary visualization using standard Python data science libraries.

📊 Pipeline Architecture & Implementation Steps
Data Loading & Inspection:
Loaded the Iris dataset via pandas and inspected data types, summary statistics, and verified that zero missing values exist across all features
(sepal_length, sepal_width, petal_length, petal_width).

Exploratory Data Analysis (EDA):
Generated Seaborn pairplots (diag_kind='kde' colored by species), revealing that the setosa species is clearly distinct and separable, while versicolor and virginica exhibit feature overlap in sepal dimensions alongside a linear relationship in petal measurements.

Data Splitting & Prevention of Data Leakage: 
Split the dataset into features (X) and target (y), and performed an 80/20 train-test split using train_test_split (random_state=42) prior to scaling.

Feature Normalization: 
Applied StandardScaler to normalize training features (X_train_scaled) and transform testing features (X_test_scaled).

Model Training & Evaluation:
Logistic Regression: Initialized and trained a LogisticRegression model (random_state=42) on scaled features, achieving an exact accuracy of 1.0 (100% precision, recall, and F1-score across all classes) on the test set.

Decision Tree Classifier:
Trained a DecisionTreeClassifier (random_state=42), also achieving a test accuracy of 1.0.

Visualization:
Generated confusion matrix heatmaps (ConfusionMatrixDisplay) for both models.

Developed and plotted a custom decision boundary visualization (plot_decision_boundaries) using a meshgrid to illustrate how models partition the feature space.

🛠️ Tech Stack & Dependencies
Programming Language: Python   
Data Manipulation & Analysis: Pandas, NumPy   
Machine Learning: Scikit-Learn   
Data Visualization: Matplotlib, Seaborn