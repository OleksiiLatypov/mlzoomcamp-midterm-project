
# mlzoomcamp-midterm-project
# Exploring Mental Health Data

### PROJECT DESCRIPTION
### About Dataset
Dataset was taken from Kaggle Competition

https://www.kaggle.com/competitions/playground-series-s4e11

This dataset was collected as part of a comprehensive survey aimed at understanding the factors contributing to depression risk among adults. It was collected during an anonymous survey conducted between January and June 2023. The survey was conducted across various cities, targeting individuals from diverse backgrounds and professions. Participants, ranging from 18 to 60 years old, voluntarily provided inputs on factors such as age, gender, city, degree, job satisfaction, study satisfaction, study/work hours, and family history among others. Participants were asked to provide inputs without requiring any professional mental health assessments or diagnostic test scores.

The target variable, 'Depression', represents whether the individual is at risk of depression, marked as 'Yes' or 'No', based on their responses to lifestyle and demographic factors. The dataset has been curated to provide insights into how everyday factors might correlate with mental health risks, making it a useful resource for machine learning models aimed at mental health prediction.

This dataset can be used for predictive modeling in mental health research, particularly in identifying key contributors to mental health challenges in a non-clinical setting.

My Goal: My goal is to use data from a mental health survey to explore factors that may cause individuals to experience depression.


### Depression Analysis Dataset

This dataset contains information about individuals and various factors that may contribute to depression, including personal details, academic/work-related stressors, and mental health status. Below is an explanation of each feature in the dataset:

| **Feature**                    | **Description**                                                                                                              |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| **Name**                        | Identifier for the individual in the dataset (typically not used for analysis).                                              |
| **Gender**                      | Gender of the individual (male = 1, female = 0, or other categories if applicable).                                           |
| **Age**                         | The age of the individual (numeric value).                                                                                     |
| **City**                        | The city or region where the individual resides (e.g., urban or rural classification).                                        |
| **Working Professional or Student** | Occupation of the individual (working professional = 1, student = 0). Helps assess the impact of work vs. academic stress.    |
| **Profession**                  | The individual's job or field of work (e.g., healthcare, engineering, education, etc.). Used to analyze the impact of professions on mental health. |
| **Academic Pressure**           | The level of academic pressure felt by the individual (numeric or ordinal scale). Used to assess the impact of academic stress on mental health. |
| **Work Pressure**               | The level of work-related pressure felt by the individual (numeric or ordinal scale). Used to assess the relationship between work stress and depression. |
| **CGPA**                        | Cumulative Grade Point Average (CGPA) of the individual (for students). Helps analyze how academic performance correlates with depression. |
| **Study Satisfaction**          | The level of satisfaction the individual feels about their studies (numeric or ordinal scale). Used to explore the relationship between academic satisfaction and mental health. |
| **Depression Status**           | Whether the individual has depression (1 = depressed, 0 = not depressed). This is the target variable for classification or prediction tasks. |

### Classification Models Used

In this project, we will test the hypothesis of whether a health insurance claim will be made based on the provided features. To explore this, we will use the following classification models:

- **Logistic Regression**
- **Ridge Classifier**
- **Decision Trees**
- **Random Forest**
- **XGBoost**
- **CatBoost**
- **LightGBM**

Each of these models will be evaluated to determine which provides the best performance in predicting health insurance claims. We will compare their results using appropriate evaluation metrics, such as accuracy, precision, recall, and F1-score.

### File Description

The folder **Midterm Project** includes the following files:

### File Description

The folder **Midterm Project** includes the following files:

### File Description

The folder **Midterm Project** includes the following files:

| **File Name**             | **Description**                                                                                         |
|---------------------------|---------------------------------------------------------------------------------------------------------|
| **data**                  | Directory containing two CSV files with the dataset for analysis.                                        |
| **notebook.ipynb**        | Jupyter notebook for data preprocessing, cleaning, and model selection.                                   |
| **train.py**              | Python script for training the final machine learning model.                                              |
| **model.bin**             | The saved model file, serialized using pickle for later use.                                              |
| **predict.py**            | Script to load the trained model and serve it through a web service (using Flask).                       |
| **predict_test.py**       | Script to test the functionality of the trained model.                                                    |
| **Pipfile & Pipfile.lock**| Files for managing project dependencies within a Python virtual environment using Pipenv.                |
| **Dockerfile**            | Configuration file for setting up the environment and running the project in Docker.                     |


### How to Set Up the Project

Follow these steps to set up the project from GitHub on your local machine:

#### 1. Clone the Repository

First, clone the repository from GitHub to your local machine. Open a terminal or command prompt and run:


`git clone https://github.com/OleksiiLatypov/mlzoomcamp-midterm-project`

### 2. Navigate to the Project Directory
Change to the project directory:

`cd mlzoomcamp-midterm-project`

### 3. Set Up the Virtual Environment And Install Pipenv
For Windows:

`python -m venv venv
.\venv\Scripts\activate
`

For macOS/Linux:

`
python3 -m venv venv
source venv/bin/activate
`

Then install Pipenenv:

`pip install pipenv`
### 4. Install Project Dependencies
Once the virtual environment is activated, install the required dependencies using Pipenv (which is used to manage the project dependencies):

`pipenv shell `

you can install all the dependencies from Pipfile by running
`pipenv install`

or manually

`pipenv install pandas numpy scikit-learn flask optuna catboost seaborn xgboost lightgbm gunicorn`

### 5.Install Docker
- Download & Intall Docker Desktop https://www.docker.com/

`docker build -t depression-prediction .`

- Run it, execute the command below:

`docker run -it -p 9696:9696 depression-prediction:latest`

### 6. Run app and make prediction

- Open new terminal and run:

`python predict_test.py`

You should see:

{'depression': False, 'depression_probability': 0.017845543314410916}

Person does not have depression.













