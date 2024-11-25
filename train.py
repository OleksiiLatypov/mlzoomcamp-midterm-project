import pandas as pd
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, fbeta_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna
import pickle


df_train = pd.read_csv('/workspaces/mlzoomcamp-midterm-project/data/train.csv')
df_original = pd.read_csv('/workspaces/mlzoomcamp-midterm-project/data/final_depression_dataset_1.csv')

#print(df_train.head())

df_train = df_train.drop('id', axis=1)
#print(df_train.head(3))

df_train.columns = df_train.columns.str.lower().str.replace(' ', '_').str.replace('_?', '')
df_original.columns = df_original.columns.str.lower().str.replace(' ', '_').str.replace('_?', '')

print(df_train.head(3))

#Convert the depression column from categorical strings ('Yes', 'No') to binary integers (1, 0).
df_original['depression'] = (df_original['depression'] == 'Yes').astype('int')


#Concatenate the df_train and df_original DataFrames along the rows (axis=0) to combine the datasets.
df = pd.concat([df_train, df_original], axis=0, ignore_index=True)

#print(df.shape)

#Checking dataset imbalance ratio
negative_depression = df[df['depression'] == 0].shape[0]
positive_depression = df[df['depression'] == 1].shape[0]
imbalanced_ratio = 100 * (positive_depression / negative_depression)
print(np.round(imbalanced_ratio))

df.duplicated().sum()

#dropping high missing values column 'study_satisfaction', 'academic_pressure', 'cgpa'
df = df.drop(['study_satisfaction', 'academic_pressure', 'cgpa'], axis=1)


categorical_features = df.select_dtypes(include=['object', 'category'])
for col in categorical_features.columns:
  print(f'{col} has {df[col].nunique()} unique values')

#dropping high cardinality columns
df = df.drop(['name', 'city'], axis=1)



def clean_columns(df: pd.DataFrame, column: str, valid_categories: List[str]) -> pd.DataFrame:
    """
    Cleans a given column in a DataFrame by replacing any values that are not in the list of valid categories
    with 'Noise'. This ensures that the column only contains valid, expected categories.

    Args:
    - df (pd.DataFrame): The pandas DataFrame containing the column to be cleaned.
    - column (str): The name of the column to clean.
    - valid_categories (List[str]): A list of valid categories that the column values should belong to.

    Returns:
    - pd.DataFrame: The updated DataFrame with the cleaned column.
    """

    # Apply a lambda function to each entry in the column to check if it's in valid categories
    df[column] = df[column].apply(lambda x: x if x in valid_categories else 'Noise')

    return df

valid_sleep_duration = [
    "Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"
]
valid_dietary_habits = ["Healthy", "Moderate", "Unhealthy"]


df = clean_columns(df, 'sleep_duration', valid_sleep_duration)
df = clean_columns(df, 'dietary_habits', valid_dietary_habits)


# check new values for sleep duration
print(df['sleep_duration'].value_counts())



def remove_noise(df: pd.DataFrame, columns: List[str], threshold: int = 100) -> pd.DataFrame:
    """
    Cleans specified columns in a DataFrame by replacing categories with a low frequency
    (less than the threshold) with a generic label 'Other'. This helps to deal with rare categories
    that may not add significant predictive value to the analysis.

    Args:
    - df (pd.DataFrame): The pandas DataFrame containing the columns to be cleaned.
    - columns (List[str]): A list of column names to clean.
    - threshold (int): The frequency threshold below which categories will be replaced with 'Other'.
                        Default is 100.

    Returns:
    - pd.DataFrame: The updated DataFrame with the noisy categories replaced by 'Other'.
    """

    # Iterate through each specified column
    for column in columns:
        # Get the frequency count of each category in the column
        value_counts = df[column].value_counts()

        # Identify categories with frequency less than the threshold
        low_freq_categories = value_counts[value_counts < threshold].index

        # Replace low frequency categories with 'Other'
        df[column] = df[column].apply(lambda x: x if x not in low_freq_categories else 'Other')

    return df

df = remove_noise(df, ['profession', 'degree'])

# check new values for profession column
print('check new values for profession column')
print(df['profession'].value_counts())


#Converting the features 'work_pressure', 'job_satisfaction', and 'financial_stress' to the category data type
df['work_pressure'] = df['work_pressure'].astype('str')
df['job_satisfaction'] = df['job_satisfaction'].astype('str')
df['financial_stress'] = df['financial_stress'].astype('str')


print(df.info())


#Filling missing values of 'work_pressure', 'job_satisfaction', 'financial_stress', 'degree', 'profession' (NaN) in the column with 'Unknown' value.
fill_with_unknown = ['work_pressure', 'job_satisfaction', 'financial_stress', 'degree', 'profession']

for col in fill_with_unknown:
    # Fill missing values with 'Unknown'
    df[col] = df[col].fillna('Unknown')


print(df.isna().sum())


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1, stratify=df['depression'])


from sklearn.metrics import mutual_info_score


categorical = df_full_train.select_dtypes(include=['object']).columns.to_list()

def mutual_info_depression_score(series):
    return mutual_info_score(series, df_full_train.depression)



mi = df_full_train[categorical].apply(mutual_info_depression_score)
mi.sort_values(ascending=False)

y = df_full_train.depression
X = df_full_train.drop('depression', axis=1)




def train_model(X, y, model):
    """
    Trains a model using cross-validation and calculates performance metrics.

    Parameters:
    - X: The features of the training data.
    - y: The target variable for the training data.
    - model: The machine learning model (e.g., RandomForestClassifier).

    Returns:
    - res: A dictionary with average accuracy, precision, recall, F1 score, and ROC AUC score.
    - dv: The trained DictVectorizer.
    """

    # Lists to store metrics
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)  # 5-fold cross-validation


    #Stratify cross-validation loop
    for fold, (train_index, val_index) in enumerate(skf.split(X, y), start=1):
        # Split data into training and validation sets
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Transform the data using DictVectorizer (if categorical features exist)
        dv = DictVectorizer(sparse=False)
        X_train_dict = X_train.to_dict(orient='records')
        X_val_dict = X_val.to_dict(orient='records')
        X_train_transformed = dv.fit_transform(X_train_dict)
        X_val_transformed = dv.transform(X_val_dict)

        # Train the model
        model.fit(X_train_transformed, y_train)

        # Get predicted class labels and probabilities
        y_pred = model.predict(X_val_transformed)

        # Calculate metrics for this fold
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='binary')  # Use 'binary' or 'macro' depending on classification type
        recall = recall_score(y_val, y_pred, average='binary')
        f1 = f1_score(y_val, y_pred, average='binary')

        # Append metrics to the lists
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)


        # Print metrics for each fold
        print(f"Fold {fold} Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
              f"F1 Score: {f1:.4f}")

    # After all folds, aggregate results
    res = {
        'Average_Accuracy': sum(accuracy_scores) / len(accuracy_scores),
        'Average_Precision': sum(precision_scores) / len(precision_scores),
        'Average_Recall': sum(recall_scores) / len(recall_scores),
        'Average_F1': sum(f1_scores) / len(f1_scores),
    }

    return res, dv

### Linear Models ###
model_lr = LogisticRegression(max_iter=500, class_weight='balanced', random_state=1)
results_lr, dv_lr = train_model(X, y, model_lr)
print('Logistic Regression')
print(results_lr, dv_lr)

model_lr = RidgeClassifier(alpha=1.0)
results_ridge, dv_ridge = train_model(X, y, model_lr)
print('Ridge Regression')
print(results_ridge, dv_ridge)

# ### Ensemble models ###

# #RandomForestClassifier
model_rf = RandomForestClassifier(class_weight='balanced', random_state=1)
results_rf, dv_rf = train_model(X, y, model_rf)
print(results_rf, dv_rf)

# #XGBClassifier
model_xgb =  XGBClassifier(n_estimators=500, scale_pos_weight=1, random_state=1, eval_metric='logloss')
results_xgb, dv_xgb = train_model(X, y, model_xgb)
print('XGBClassifier')
print(results_xgb, dv_xgb)

#CatBoostClassifier
model_cat =  CatBoostClassifier(iterations=500, random_state=1, eval_metric='Logloss', silent=True)
results_cat, dv_cat = train_model(X, y, model_cat)
print('CatBoostClassifier')
print(results_cat, dv_cat)

#LightGBM
model_lgb = LGBMClassifier(n_estimators=500, random_state=1, verbosity=-1)
results_lgb, dv_lgb = train_model(X, y, model_lgb)
print('LightGBM Classifier')
print(results_lgb, dv_lgb)


#Hyperparametrs tuning using OPTUNA

def objective(trial, X, y):
    """
    Objective function for Optuna hyperparameter optimization.

    Parameters:
    - trial: Optuna trial object that defines the hyperparameter space.
    - X: Features for the training data.
    - y: Target variable for the training data.

    Returns:
    - Average accuracy for the given set of hyperparameters across folds.
    """
    # Define hyperparameters for CatBoost using the trial object
    param = {
        'iterations': trial.suggest_int('iterations', 100, 1000),  # Number of trees
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),  # Learning rate
        'depth': trial.suggest_int('depth', 4, 8),  # Depth of the trees
        #'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0),  # Regularization term
        #'border_count': trial.suggest_int('border_count', 32, 255),  # Number of splits
        'eval_metric': 'Logloss',  # Evaluation metric
        'random_state': 1,
        'silent': True  # Suppress verbose output
    }

    # Initialize the CatBoostClassifier with suggested parameters
    model = CatBoostClassifier(**param)

    # List to store accuracy scores
    accuracy_scores = []

    # Initialize StratifiedKFold for cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)  # 5-fold cross-validation

    # Cross-validation loop
    for train_index, val_index in skf.split(X, y):
        # Split data into training and validation sets
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Transform the data using DictVectorizer (if categorical features exist)
        dv = DictVectorizer(sparse=False)
        X_train_dict = X_train.to_dict(orient='records')
        X_val_dict = X_val.to_dict(orient='records')
        X_train_transformed = dv.fit_transform(X_train_dict)
        X_val_transformed = dv.transform(X_val_dict)

        # Train the model
        model.fit(X_train_transformed, y_train)

        # Get predicted class labels
        y_pred = model.predict(X_val_transformed)

        # Calculate accuracy for this fold
        accuracy = accuracy_score(y_val, y_pred)
        accuracy_scores.append(accuracy)

    # Return the average accuracy for the current set of hyperparameters
    return sum(accuracy_scores) / len(accuracy_scores)

def tune_hyperparameters(X, y):
    """
    Tunes hyperparameters for the CatBoostClassifier using Optuna.

    Parameters:
    - X: Features for the training data.
    - y: Target variable for the training data.

    Returns:
    - best_trial: The best trial (with the optimal hyperparameters).
    - best_params: The best hyperparameters found by Optuna.
    """
    # Set up the Optuna study to maximize the objective function (maximize accuracy)
    study = optuna.create_study(direction='maximize')  # We want to maximize accuracy
    study.optimize(lambda trial: objective(trial, X, y), n_trials=25)  # 20 trials for hyperparameter search

    # Retrieve the best trial (best hyperparameters)
    best_trial = study.best_trial
    best_params = best_trial.params
    #print(f"Best hyperparameters: {best_params}")
    #print(f"Best accuracy: {best_trial.value}")

    return best_trial, best_params

# Example usage:
# X and y are your feature and target variables
#best_trial, best_params = tune_hyperparameters(X, y)

# Optionally, you can retrain your model using the best parameters

# best_model = CatBoostClassifier(**best_params)
# dv = DictVectorizer(sparse=False)
# X_dict = X.to_dict(orient='records')
# X_transformed = dv.fit_transform(X_dict)
# best_model.fit(X_transformed, y)

# Now you have the best model trained with optimized hyperparameters

#Best hyperparameters: {'iterations': 819, 'learning_rate': 0.04405886196667592, 'depth': 4}


# Lists to store metrics
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
feature_importances_list = []

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)  # 5-fold cross-validation

# Initialize CatBoostClassifier with the specified hyperparameters
model_cat_tuned = CatBoostClassifier(
    iterations=819,
    learning_rate=0.04405886196667592,
    depth=4,
    eval_metric='Logloss',  # You can change this depending on your objective
    random_state=1,
    silent=True  # Suppress output during training
)

# Cross-validation loop
for fold, (train_index, val_index) in enumerate(skf.split(X, y), start=1):
    # Split data into training and validation sets
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Transform the data using DictVectorizer (if categorical features exist)
    dv = DictVectorizer(sparse=False)
    X_train_dict = X_train.to_dict(orient='records')
    X_val_dict = X_val.to_dict(orient='records')
    X_train_transformed = dv.fit_transform(X_train_dict)
    X_val_transformed = dv.transform(X_val_dict)

    # Train the CatBoost model
    model_cat_tuned.fit(X_train_transformed, y_train)

    # Get predicted class labels
    y_pred = model_cat_tuned.predict(X_val_transformed)

    # Calculate metrics for this fold
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='binary')  # Use 'binary' or 'macro' depending on classification type
    recall = recall_score(y_val, y_pred, average='binary')
    f1 = f1_score(y_val, y_pred, average='binary')

    # Append metrics to the lists
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    # Capture feature importances for this fold
    feature_importances = model_cat_tuned.feature_importances_
    feature_importances_list.append(feature_importances)

    # Print metrics for each fold
    print(f"Fold {fold} Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, "
          f"F1 Score: {f1:.4f}")

# After all folds, aggregate results
res = {
    'Average_Accuracy': sum(accuracy_scores) / len(accuracy_scores),
    'Average_Precision': sum(precision_scores) / len(precision_scores),
    'Average_Recall': sum(recall_scores) / len(recall_scores),
    'Average_F1': sum(f1_scores) / len(f1_scores),
}

# Print the overall performance metrics
print("\nAverage Metrics across all folds:")
print(f"Average Accuracy: {res['Average_Accuracy']:.4f}")
print(f"Average Precision: {res['Average_Precision']:.4f}")
print(f"Average Recall: {res['Average_Recall']:.4f}")
print(f"Average F1 Score: {res['Average_F1']:.4f}")

# Optionally, you can also aggregate the feature importances
average_feature_importance = sum(feature_importances_list) / len(feature_importances_list)
print("\nAverage Feature Importances:")
print(average_feature_importance)


# Aggregate feature importances (average across folds)
average_feature_importances = np.mean(feature_importances_list, axis=0)

# Get feature names from DictVectorizer
feature_names = dv.get_feature_names_out()

# Create a DataFrame to combine feature names with their importance values
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': average_feature_importances
})

# Sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print the feature importance
print("Feature Importance:\n", importance_df)

output_file = f'model.bin'
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model_cat_tuned), f_out)

print(f'the model is saved to {output_file}')




