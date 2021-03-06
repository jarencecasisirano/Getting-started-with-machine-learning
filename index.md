# General machine learning workflow

## [Basic data exploration](https://www.kaggle.com/code/dansbecker/basic-data-exploration)
Use [Pandas](https://pandas.pydata.org/docs/) library. Access and explore the data.

```python
# Import pandas library
import pandas as pd

# Load the data
data = pd.read_csv(file_path)
data.describe()
```

## Select data subsets for modeling
We select the *prediction target (**y**)* which is the column we want to predict and *predictive features (**X**)* which are the columns we use to make predictions. To select columns, we can use either of the following methods:
1. Dot-notation.
2. Selecting with column list.

```python
# Dot notation
y = data.column_to_predict

# Selecting with column list
features = ['feature_1', 'feature_2', 'feature_3']
X = data[features]
```

## Model Validation
To perform model validation, we split our data into two: *training data* and *test data*. We use the [scikit-learn](https://scikit-learn.org/stable/index.html) library function `train_test_split` to do this. The training data set is used to fit the model while the test data set is used to validate the model.

```python
from sklearn.model_selection import train_test_split

# Use the train_test_split function
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
```

## Building the model
In creating the model, we use the [scikit-learn](https://scikit-learn.org/stable/index.html) library: `from sklearn.tree import DecisionTreeRegressor`.
1. Define the model *(e.g., Decision Tree Regressor)*.
2. Fit the training data to the model.
3. Predict.
4. Evaluate the model *(e.g., Mean Absolute Error or MAE)*.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Defining the model
model = DecisionTreeRegressor(random_state=1)

# Fitting the model
model.fit(X_train, y_train)

# Predicting
prediction = model.predict(X_test)

# Model evaluation
mean_absolute_error(y_test, prediction)
```

## Saving the output
To save the output, we create [Pandas](https://pandas.pydata.org/docs/) dataframe then convert it to a CSV file.

```python
output = pd.DataFrame({'Id': data.Id,
                       'Predicted_Column': prediction})
output.to_csv('submission.csv', index=False)
```

# Machine learning model improvements

#### Note: [Underfitting and overfitting](https://www.kaggle.com/code/dansbecker/underfitting-and-overfitting)
- **Underfitting**: the model fails to capture important distinctions and patterns in the data, so it performs poorly even in training data.
- **Overfitting**: the model matches the training data almost perfectly, but does poorly in validation and other new data.

To overcome underfitting and overfitting, we control the tree depth using the `max_leaf_nodes` argument of the model. We can define a utility function to compare MAE from different `max_leaf_nodes` values.

```python
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

# Define utility function for MAE comparison
def get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    mae = mean_absolute_error(y_test, prediction)
    return(mae)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
```

## Improving the model with [Random Forests](https://www.kaggle.com/code/dansbecker/random-forests)
The machine learning model above uses a **decision tree**. A much more sophisticated machine learning algorithm is **random forest**. The random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree. It generally has much better predictive accuracy than a single decision tree and it works well with default parameters.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Build, fit, and predict with the model
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(X_train, y_train)
prediction = forest_model.predict(X_test)
print(mean_absolute_error(y_test, prediction))
```

## [Handling Missing Values](https://www.kaggle.com/code/alexisbcook/missing-values)
We usually have to deal with missing data values. There are three approaches in handling missing values:
- Drop columns with missing values: If only few entries are missing, the model might lose access to a lot of potentially useful information.
- Imputation: Filling in missing values with some number.
- Extended Imputation: Imputing missing values and adding a column to keep track of imputed entries.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# Function for comparing different approaches
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    return mean_absolute_error(y_test, prediction)

# DROPPING
# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test = X_test.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))

# IMPUTATION
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_test.columns = X_test.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))

# EXTENDED IMPUTATION
# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_test_plus = X_test.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_test_plus[col + '_was_missing'] = X_test_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_test_plus = pd.DataFrame(my_imputer.transform(X_test_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_test_plus.columns = X_test_plus.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))
```

## [Categorical Variables](https://www.kaggle.com/code/alexisbcook/categorical-variables)
In most machine learning models in Python, preprocessing categorical variables is a must. There are three approaches in handling categorical variables:
- Drop categorical variables: only works well if the columns did not contain useful information.
- Ordinal encoding: assigns each unique value to a different integer.
- One-Hot Encoding: creates new columns indicating the presence (or absence) of each possible value in the original.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

# Function for comparing different approaches
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    return mean_absolute_error(y_test, prediction)
    
# DROPPING
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_test = X_test.select_dtypes(exclude=['object'])

print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_test, y_train, y_test))

# ORDINAL ENCODING
# Make copy to avoid changing original data 
label_X_train = X_train.copy()
label_X_test = X_test.copy()

# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_test[object_cols] = ordinal_encoder.transform(X_test[object_cols])

print("MAE from Approach 2 (Ordinal Encoding):") 
print(score_dataset(label_X_train, label_X_test, y_train, y_test))

# ONE-HOT ENCODING
# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_test.index = X_test.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_test = X_test.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)

print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_test, y_train, y_test))
```

## [Pipelines](https://www.kaggle.com/code/alexisbcook/pipelines)
We use pipelines to keep our data preprocessing and modeling code organized. Constructing a pipeline has three steps:
1. Define preprocessing steps.
2. Define the model.
3. Create and evaluate the pipeline.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# PREPROCESSING
# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# MODEL
model = RandomForestRegressor(n_estimators=100, random_state=0)

# CREATE AND EVALUATE PIPELINE
# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
prediction = my_pipeline.predict(X_test)

# Evaluate the model
score = mean_absolute_error(y_test, prediction)
print('MAE:', score)
```

## [Cross-Validation](https://www.kaggle.com/code/alexisbcook/cross-validation)
Cross-validation means running the modeling process on different subsets of the data (called folds) to get multiple measures of model quality. From [scikit-learn](https://scikit-learn.org/stable/index.html), we can set the number of folds using the `cv` parameter and obtain the cross-validation scores with the `cross_val_score()` function. Using a pipeline makes cross-validation code remarkably straightforward.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=50,
                                                              random_state=0))
                             ])
                             
# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)
print("Average MAE score (across experiments):")
print(scores.mean())
```

## [XGBoost](https://www.kaggle.com/code/alexisbcook/xgboost)
XGBoost stands for extreme gradient boosting, which is an implementation of gradient boosting with several additional features focused on performance and speed. Gradient boosting is a method that goes through cycles to iteratively add models into an ensemble.
### Parameter Tuning
XGBoost has a few parameters that can dramatically affect accuracy and training speed. Some of these are:
- `n_estimators`: specifies how many times to go through the modeling cycle described above. It is equal to the number of models that we include in the ensemble.
- `early_stopping_rounds`: offers a way to automatically find the ideal value for n_estimators. Early stopping causes the model to stop iterating when the validation score stops improving, even if we aren't at the hard stop for n_estimators.
- `learning_rate`: In general, a small learning rate and large number of estimators will yield more accurate XGBoost models, though it will also take the model longer to train since it does more iterations through the cycle.
- `n_jobs`: On larger datasets where runtime is a consideration, you can use parallelism to build your models faster.

```python
from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_test, y_test)], 
             verbose=False)
```

## [Data Leakage](https://www.kaggle.com/code/alexisbcook/data-leakage)
**Data leakage** (or leakage) happens when your training data contains information about the target, but similar data will not be available when the model is used for prediction. This leads to high performance on the training set (and possibly even the validation data), but the model will perform poorly in production. There are two main types of leakage:
1. Target leakage - occurs when your predictors include data that will not be available at the time you make predictions.
2. Train-test contamination - occurs when you aren't careful to distinguish training data from validation data.

# Machine learning explainability

## [Permutation Importance](https://www.kaggle.com/code/dansbecker/permutation-importance/tutorial)
**Feature importance** is a concept that answers the question: what features have the biggest impact on predictions? One such approach is **permutation importance**. Note that permutation importance is calculated after a model has been fitted. So we won't change the model or change what predictions we'd get for a given value of features. The process of this approach is as follows:
1. Get a trained model.
2. Shuffle the values in a single column, make predictions using the resulting dataset. Use these predictions and the true target values to calculate how much the loss function suffered from shuffling. That performance deterioration measures the importance of the variable you just shuffled.
3. Return the data to the original order (undoing the shuffle from step 2). Now repeat step 2 with the next column in the dataset, until you have calculated the importance of each column.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import eli5
from eli5.sklearn import PermutationImportance

# Load the data
data = pd.read_csv('../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')

# Assign prediction target to a variable
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary

# Assign predictive features to a variable
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]

# Split the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Define the model
my_model = RandomForestClassifier(n_estimators=100,
                                  random_state=0).fit(X_train, y_train)

perm = PermutationImportance(my_model, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())
```

## [Partial Dependence Plots](https://www.kaggle.com/code/dansbecker/partial-plots)
Partial dependence plots show *how* a feature affects predictions. Like permutation importance, partial dependence plots are calculated after a model has been fit. The model is fit on real data that has not been artificially manipulated in any way.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Load data, assign X and y, split data into train and test data
data = pd.read_csv('../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(X_train, y_train)

# Decision Tree
from sklearn import tree
import graphviz

tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=feature_names)
graphviz.Source(tree_graph)

# Partial Dependence Plot using PDPBox Library
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Sample Plot 1
pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=X_test, model_features=feature_names, feature='Goal Scored')

pdp.pdp_plot(pdp_goals, 'Goal Scored')
plt.show()

# Sample Plot 2
feature_to_plot = 'Distance Covered (Kms)'
pdp_dist = pdp.pdp_isolate(model=tree_model, dataset=X_test, model_features=feature_names, feature=feature_to_plot)

pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()

# Random Forest PDP
# Build Random Forest model
rf_model = RandomForestClassifier(random_state=0).fit(X_train, y_train)

pdp_dist = pdp.pdp_isolate(model=rf_model, dataset=X_test, model_features=feature_names, feature=feature_to_plot)

pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()

# 2D Partial Dependence Plot
# Similar to previous PDP plot except we use pdp_interact instead of pdp_isolate and pdp_interact_plot instead of pdp_isolate_plot
features_to_plot = ['Goal Scored', 'Distance Covered (Kms)']
inter1  =  pdp.pdp_interact(model=tree_model, dataset=X_test, model_features=feature_names, features=features_to_plot)

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')
plt.show()
```

## [SHAP Values](https://www.kaggle.com/code/dansbecker/shap-values)
SHAP Values (an acronym from SHapley Additive exPlanations) break down a prediction to show the impact of each feature. SHAP values interpret the impact of having a certain value for a given feature in comparison to the prediction we'd make if that feature took some baseline value.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data, assign X and y variables, split data into train and test data sets
data = pd.read_csv('../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64, np.int64]]
X = data[feature_names]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(random_state=0).fit(X_train, y_train)

row_to_show = 5
data_for_prediction = X_test.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

my_model.predict_proba(data_for_prediction_array)

import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)

shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)

# use Kernel SHAP to explain test set predictions
k_explainer = shap.KernelExplainer(my_model.predict_proba, X_train)
k_shap_values = k_explainer.shap_values(data_for_prediction)
shap.force_plot(k_explainer.expected_value[1], k_shap_values[1], data_for_prediction)
```

## [Advanced Uses of SHAP Values](https://www.kaggle.com/code/dansbecker/advanced-uses-of-shap-values)
### Summary Plots
SHAP summary plots give us a birds-eye view of feature importance and what is driving it.
- Vertical location shows what feature it is depicting
- Color shows whether that feature was high or low for that row of the dataset
- Horizontal location shows whether the effect of that value caused a higher or lower prediction

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data, assign X and y variables, split data into train and test data sets
data = pd.read_csv('../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64, np.int64]]
X = data[feature_names]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(random_state=0).fit(X_train, y_train)

import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# calculate shap values. This is what we will plot.
# Calculate shap_values for all of X_test rather than a single row, to have more data for plot.
shap_values = explainer.shap_values(X_test)

# Make plot. Index of [1] is explained in text below.
shap.summary_plot(shap_values[1], X_test)
```

### SHAP Dependence Contribution Plots
SHAP dependence contribution plots provide a similar insight to PDP's, but they add a lot more detail.

```python
import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# calculate shap values. This is what we will plot.
shap_values = explainer.shap_values(X)

# make plot.
shap.dependence_plot('Ball Possession %', shap_values[1], X, interaction_index="Goal Scored")
```
