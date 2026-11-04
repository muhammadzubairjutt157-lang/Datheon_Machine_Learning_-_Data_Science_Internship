# California Housing Price Prediction & Regression Pipeline

An end-to-end Machine Learning pipeline built to predict median house values in California block groups using regression models (Ridge and Lasso).

## 📊 Project Workflow
1. Data Ingestion & Inspection:
	Cleaned datasets, handled missing values and reviewed feature distributions.
2. Exploratory Data Analysis (EDA):
	Analyzed feature skews and mapped geographic coordinates (`longitude`, `latitude`) alongside population density and house prices using custom background overlays.
3. Preprocessing & Encoding:
	Applied One-Hot Encoding to categorical features (`ocean_proximity`) and standardized numerical features using `StandardScaler`.
4. Model Training & Tuning:
	Optimized **Ridge** and **Lasso** regression models via 5-fold cross-validation (`GridSearchCV`).
5. Evaluation & Visualization:
	Evaluated models using RMSE, MAE, and $R^2$ scores ($\approx 0.626$), and plotted **Predicted vs. Actual** values alongside **Residual Error Distributions**.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn