
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

### Dataset Usage
This dataset is intended to help researchers and analysts explore the relationship between various stress factors (academic, work, etc.) and depression. It can be used for:
- **Exploratory Data Analysis (EDA)**
- **Correlation studies**
- **Predictive modeling for depression status**




