# 1. Import Libraries
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
from numpy.ma.core import floor
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# 2. Load Data
df=pd.read_csv("C:\\Users\\User-1\\Documents\\Master Medieninformatik Beuth\\Data Science\\SS 24\\Einsendeaufgabe 8\\Fish.csv")

print(df.head())

# 3. Data Preparation
# to confirmm  the dataset was loaded correctly and to understand its structure before proceeding with further analysis
print(df.shape)

# get a concise summary of a DataFrame. Includes information about the DataFrame's index, column names, non-null counts, and data types of each column.
print(df.info())

# convert type Object of "Species" into to numeric (this is generally necessary for machine learning models
# that do not natively handle categorical data, such as linear regression and random forest.
# One common method is one-hot encoding, which creates binary columns for each category.
# (preparing categorical data for machine learning)
#
# Save the 'Species' column before one-hot encoding
species_column = df['Species'].copy()

# One-hot encoding for the 'Species' column:
df = pd.get_dummies(df, columns=['Species'], drop_first=True)

print(df.info())

# Export DataFrame to CSV (just to double check)
#df.to_csv('C:/Users/User-1/Documents/Master Medieninformatik Beuth/Data Science/SS 24/Einsendeaufgabe 8/data.csv', index=False)

# Check for Duplicates:
print(df.duplicated().sum())

# Review the descriptive statistics of the dataset to understand the distribution of values.
print(df.describe())

# 4. Splitting Data To building Model to predict the weight of the fish based on various features.
X = df.drop('Weight', axis=1) #test data
y = df['Weight'] #expected output data we try to predict
# Explanation:
# Common Machine Learning Model Preparation
# X will be used as the input to train the machine learning model.
# X is typically a DataFrame or a numpy array that contains all the input variables
# (also called predictors or independent variables). In this case, all data except the weight column,
# which is why we drop the column.
# y will be used as the expected output that the model will attempt to predict.
# Splitting Data:
# After defining X and y, the next step involves splitting the data into training
# and testing sets.


# ### Task 1
#
# 1. Split the dataset randomly into training (70%) and testing (30%) sets.

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=42)
# It is a best practice to set random_state to ensure that the analysis is reproducible.
# By setting random_state, one can confidently share the code with others,
# knowing that they will obtain the same results when they run it.

print(x_train)

print("Test Data = ",x_test.shape)
print("Train Data = ",x_train.shape)
print("Target Train = ",y_train.shape)
print("Target Test = ",y_test.shape)

print(y_train)

# 2. Apply the following models:
#
#    - Linear Regression
LR_model = LinearRegression()
LR_model.fit(x_train,y_train)
y_pred_LR = LR_model.predict(x_test)

#    - Random Forest
RF_model = RandomForestRegressor(random_state=42)
RF_model.fit(x_train, y_train)
y_pred_RF = RF_model.predict(x_test)

#3. Calculate RMSE (Root Mean Squared Error) and R2 (Coefficient of Determination) on the test set.
# info: Use of LinearRegression for small to medium-sized datasets where we want a deterministic solution.
# Use of SGDRegressor for larger datasets or when we prefer an iterative approach to optimization.

# RMSE and R2 of Linear Regression
rmse_LR = np.sqrt(mean_squared_error(y_test, y_pred_LR))
r2_LR = r2_score(y_test, y_pred_LR)

# RMSE and R2 of Random Forest
rmse_RF = np.sqrt(mean_squared_error(y_test, y_pred_RF))
r2_RF = r2_score(y_test, y_pred_RF)

print("Linear Regression - RMSE =",rmse_LR)
print("Linear Regression - R2 = ",floor(r2_LR*100),"%")
#formats the R2 score by multiplying it by 100 and applying the floor function,
# which truncates the decimal part. Primarily for display purposes.

print("Random Forest - RMSE =" ,rmse_RF)
print("Random Forest - R2 = " ,floor(r2_RF*100),"%")
#formats the R2 score by multiplying it by 100 and applying the floor function,
# which truncates the decimal part. Primarily for display purposes.

#4. Visualize the predictions by plotting y_pred vs y_real and compare the performance of the models.

plt.figure(figsize=(14, 6))
# Visualization of predictions Linear Regression
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=y_pred_LR)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual (Test data)')
plt.ylabel('Predicted')
plt.title('Linear Regression')

# Visualization of predictions Random Forest
plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test, y=y_pred_RF)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual (Test data)')
plt.ylabel('Predicted')
plt.title('Random Forest')

plt.tight_layout()

plt.show()  # Displays the plots

# Performance comparision of the two models:
if rmse_LR < rmse_RF:
    print("Linear Regression has a lower RMSE, which indicates better performance in terms of prediction error.")
else:
    print("Random Forest has a lower RMSE, which indicates better performance in terms of prediction error.")

if r2_LR > r2_RF:
    print("Linear Regression has a higher R2, indicating a better fit to the data.")
else:
    print("Random Forest has a higher R2, indicating a better fit to the data.")
# Higher R² indicates that the model explains a larger proportion of the variance
# in the target variable (fish weight) compared to the other model

# Side Note - Why is Random Forest a better method in this excercise?
# The Random Forest model's superior performance in terms of both RMSE and R² can be attributed to its ability
# to capture non-linear relationships, model interactions between features, and handle outliers effectively.
# These advantages make Random Forest a more appropriate choice for this particular regression problem,
# where the relationship between the features and the target variable (fish weight) is likely complex and non-linear.

# 5. Provide your opinion on which metric, RMSE or R2, is more appropriate in this case.
# RMSE provides a direct measure of prediction error, making it useful for understanding
# the model's performance in terms of absolute prediction accuracy.
# R2 explains the proportion of variance in the dependent variable that's predictable from
# the independent variables. It's useful for understanding the goodness-of-fit (= Anpassungsgüte,
# gibt an, „wie gut“ ein geschätztes Modell eine Menge von Beobachtungen erklären kann.
# Maße der Anpassungsgüte erlauben eine Aussage über die Diskrepanz zwischen den theoretischen Werten
# der untersuchten Zufallsvariablen,die aufgrund des Modells erwartet bzw. prognostiziert werden,
# und den tatsächlich gemessenen Werten.).

# In this case, both metrics are valuable. However, RMSE is often more interpretable
# in practical scenarios as it gives a direct measure of error in the units of the target variable.

# Both metrics should be used together to get a comprehensive understanding of the model's performance.
# While RMSE gives you an idea of how accurate the predictions are in absolute terms,
# R2 tells you how well the model explains the variability in the data.

# If the goal is to minimize prediction error (e.g., in financial predictions),
# RMSE might be more critical. However, if the goal is to understand the factors influencing the target
# variable and how well they explain it, R2 might be more valuable.

# In our Fish example:
#    - By knowing the RMSE, we can quantify how far off our fish weight predictions are from the actual weights.
#      This is crucial in practical applications where the accuracy of the weight prediction directly impacts decisions,
#      such as pricing or quality control.
#    - By knowing the R2, we can understand how well our model explains the variability in fish weight based on
#      features like species, length, height, and width. This insight is valuable for identifying which features
#      are most influential and for improving the model.

# Conclusion:
# Both RMSE and R2 provide complementary perspectives. RMSE offers an absolute measure of prediction accuracy,
# while R2 provides a relative measure of explanatory power.
# Using both metrics together gives a more complete picture of model performance.


# ### Task 2
#
# 1. Change the training-test split to ensure that each species has 70% of its samples in the training set
# and 30% in the test set.

def species_train_test_split(X, y, species_column, test_size=0.3, random_state=None):
    unique_species = species_column.unique()
    x_train_species, x_test_species = [], []
    y_train_species, y_test_species = [], []

    for species in unique_species:
        # Filter data by species
        X_species = X[species_column == species]
        y_species = y[species_column == species]

        # Split data for the current species
        x_train_spec, x_test_spec, y_train_spec, y_test_spec = \
            train_test_split(X_species, y_species, test_size=test_size, random_state=random_state)

        # Append to the lists
        x_train_species.append(x_train_spec)
        x_test_species.append(x_test_spec)
        y_train_species.append(y_train_spec)
        y_test_species.append(y_test_spec)

    # Concatenate all species-specific splits
    x_train = pd.concat(x_train_species)
    x_test = pd.concat(x_test_species)
    y_train = pd.concat(y_train_species)
    y_test = pd.concat(y_test_species)

    return x_train, x_test, y_train, y_test

x_train_species, x_test_species, y_train_species, y_test_species = species_train_test_split(X, y, species_column, test_size=0.3, random_state=42)

print("Train Data = ", x_train_species.shape)
print("Test Data = ", x_test_species.shape)
print("Target Train = ", y_train_species.shape)
print("Target Test = ", y_test_species.shape)

# 2. Apply the following models:
#
#    - Linear Regression
LR_model.fit(x_train_species, y_train_species)
y_pred_LR_species = LR_model.predict(x_test_species)

#    - Random Forest
RF_model.fit(x_train_species, y_train_species)
y_pred_RF_species = RF_model.predict(x_test_species)

#3. Calculate RMSE (Root Mean Squared Error) and R2 (Coefficient of Determination) on the test set.

# RMSE and R2 of Linear Regression
rmse_LR_species = np.sqrt(mean_squared_error(y_test_species, y_pred_LR_species))
r2_LR_species = r2_score(y_test_species, y_pred_LR_species)

# RMSE and R2 of Random Forest
rmse_RF_species = np.sqrt(mean_squared_error(y_test_species, y_pred_RF_species))
r2_RF_species = r2_score(y_test_species, y_pred_RF_species)

print("Linear Regression - RMSE species =", rmse_LR_species)
print("Linear Regression - R2 species =", floor(r2_LR_species * 100), "%")

print("Random Forest - RMSE species =", rmse_RF_species)
print("Random Forest - R2 species =", floor(r2_RF_species * 100), "%")

#4. Visualize the predictions by plotting y_pred vs y_real and compare the performance of the models.

plt.figure(figsize=(14, 6))

# Visualization of predictions Linear Regression
plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test_species, y=y_pred_LR_species)
plt.plot([y_test_species.min(), y_test_species.max()], [y_test_species.min(), y_test_species.max()], 'k--', lw=2)
plt.xlabel('Actual (Test data)')
plt.ylabel('Predicted')
plt.title('Linear Regression')

# Visualization of predictions Random Forest
plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test_species, y=y_pred_RF_species)
plt.plot([y_test_species.min(), y_test_species.max()], [y_test_species.min(), y_test_species.max()], 'k--', lw=2)
plt.xlabel('Actual (Test data)')
plt.ylabel('Predicted')
plt.title('Random Forest')

plt.tight_layout()

plt.show()  # Displays the plots

# Performance comparision of the two models:
if rmse_LR_species < rmse_RF_species:
    print("Linear Regression (split training-test for each species 70-30) has a lower RMSE, which indicates better performance in terms of prediction error.")
else:
    print("Random Forest (split training-test for each species 70-30) has a lower RMSE, which indicates better performance in terms of prediction error.")

if r2_LR_species > r2_RF_species:
    print("Linear Regression (split training-test for each species 70-30) has a higher R2, indicating a better fit to the data.")
else:
    print("Random Forest (split training-test for each species 70-30) has a higher R2, indicating a better fit to the data.")

# ### Comparison
#
# - Compare the results obtained from Task 1 and Task 2.

#show all plots from task 1 and 2 in one screen:
# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Task 1: Linear Regression
axes[0, 0].scatter(x=y_test, y=y_pred_LR)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axes[0, 0].set_xlabel('Actual (Test data)')
axes[0, 0].set_ylabel('Predicted')
axes[0, 0].set_title('Task 1: Linear Regression')

# Task 1: Random Forest
axes[0, 1].scatter(x=y_test, y=y_pred_RF)
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
axes[0, 1].set_xlabel('Actual (Test data)')
axes[0, 1].set_ylabel('Predicted')
axes[0, 1].set_title('Task 1: Random Forest')

# Task 2: Linear Regression
axes[1, 0].scatter(x=y_test_species, y=y_pred_LR_species)
axes[1, 0].plot([y_test_species.min(), y_test_species.max()], [y_test_species.min(), y_test_species.max()], 'k--', lw=2)
axes[1, 0].set_xlabel('Actual (Test data)')
axes[1, 0].set_ylabel('Predicted')
axes[1, 0].set_title('Task 2: Linear Regression')

# Task 2: Random Forest
axes[1, 1].scatter(x=y_test_species, y=y_pred_RF_species)
axes[1, 1].plot([y_test_species.min(), y_test_species.max()], [y_test_species.min(), y_test_species.max()], 'k--', lw=2)
axes[1, 1].set_xlabel('Actual (Test data)')
axes[1, 1].set_ylabel('Predicted')
axes[1, 1].set_title('Task 2: Random Forest')

# Adjust layout with more space between subplots
plt.tight_layout(pad=4.0)

# Show plots
plt.show()

# Comparison and explanation
print("### Comparison of Results ###\n")

# Linear Regression Comparison
print("Linear Regression:")
print(f"  Task 1 - RMSE: {rmse_LR:.2f}, R2: {r2_LR}%")
print(f"  Task 2 - RMSE: {rmse_LR_species:.2f}, R2: {r2_LR_species}%")

rmse_diff_LR = rmse_LR_species - rmse_LR
r2_diff_LR = r2_LR_species - r2_LR
print(f"  Difference - RMSE: {rmse_diff_LR:.2f}, R2: {r2_diff_LR}%")

if rmse_diff_LR > 0:
    print(f"  The RMSE for Linear Regression increased by {rmse_diff_LR:.2f} in Task 2. This indicates that the model's prediction error worsened when splitting the data by species.")
else:
    print(f"  The RMSE for Linear Regression decreased by {abs(rmse_diff_LR):.2f} in Task 2. This indicates that the model's prediction error improved when splitting the data by species.")

if r2_diff_LR < 0:
    print(f"  The R2 for Linear Regression decreased by {abs(r2_diff_LR)}% in Task 2. This indicates that the model explains less variance in the target variable when splitting the data by species.")
else:
    print(f"  The R2 for Linear Regression increased by {r2_diff_LR}% in Task 2. This indicates that the model explains more variance in the target variable when splitting the data by species.")

print("\n")

# Random Forest Comparison
print("Random Forest:")
print(f"  Task 1 - RMSE: {rmse_RF:.2f}, R2: {r2_RF}%")
print(f"  Task 2 - RMSE: {rmse_RF_species:.2f}, R2: {r2_RF_species}%")

rmse_diff_RF = rmse_RF_species - rmse_RF
r2_diff_RF = r2_RF_species - r2_RF
print(f"  Difference - RMSE: {rmse_diff_RF:.2f}, R²: {r2_diff_RF}%")

if rmse_diff_RF > 0:
    print(f"  The RMSE for Random Forest increased by {rmse_diff_RF:.2f} in Task 2. This indicates that the model's prediction error worsened when splitting the data by species.")
else:
    print(f"  The RMSE for Random Forest decreased by {abs(rmse_diff_RF):.2f} in Task 2. This indicates that the model's prediction error improved when splitting the data by species.")

if r2_diff_RF < 0:
    print(f"  The R2 for Random Forest decreased by {abs(r2_diff_RF)}% in Task 2. This indicates that the model explains less variance in the target variable when splitting the data by species.")
else:
    print(f"  The R2 for Random Forest increased by {r2_diff_RF}% in Task 2. This indicates that the model explains more variance in the target variable when splitting the data by species.")

print("\n### Conclusion ###\n")
print("Random Forest consistently outperformed Linear Regression in both tasks. The performance metrics (RMSE and R²) indicate that Random Forest is better suited for this dataset. When the training-test split was performed such that each species had 70% of its samples in the training set, both models experienced a decrease in performance, but the impact was more pronounced for Linear Regression.")
print("The random Split provides a more generalized and robust model due to diverse training data while the stratified sampling maintains species distribution in both sets but reduces diversity within each training subset and therefore limits model generalization.

# ### Extra Point
# point out which parameters can be adjusted in this exercise to improve model performance. (dont need to run analysis again)
#
# To further improve the model performance, consider adjusting the following parameters:
#
# -  Create new features or transform existing ones to capture more relevant information. Use polynomial features or interaction terms.
# -  Model Hyperparameters:
#   Random Forest:
#   Increase the number of trees (n_estimators).
#   Adjust the maximum depth of trees (max_depth).
#   Tune other parameters like min_samples_split, min_samples_leaf, and max_features.
#   Linear Regression: Regularize the model using Ridge or Lasso regression to prevent overfitting.
# -  Normalize or standardize features to ensure they are on a similar scale.
# -  Address any potential outliers in the dataset.
# -  Use cross-validation to obtain more reliable estimates of model performance.
# -  Consider ensemble methods like Gradient Boosting or stacking multiple models.