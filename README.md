# Customer Churn Prediction Model using Decision Trees

Our case study focuses on a churn dataset, where "churned customers" are those ending relationships with their current company. XYZ, a service provider, offers a one-year subscription plan and wants to predict customer renewal.
  
## Table of Contents
 
- [Description](#description)
- [Architecture](#architecture)
- [Features](#features)
- [Modular_Code_Overview](#modular_code_overview) 
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing) 
- [Contact](#contact)

## Description

A Decision Tree is a supervised learning method suitable  for both classification and regression tasks, often used for solving classification problems. It's structured as a tree-like classifier, with nodes representing dataset features, branches as decision rules, and leaf nodes as outcomes.

This graphical representation offers potential solutions based on given conditions. Beginning from a root node, it grows branches and forms a tree structure. The tree poses questions and branches further based on Yes/No answers.

Our case study focuses on a churn dataset, where "churned customers" are those ending relationships with their current company. XYZ, a service provider, offers a one-year subscription plan and wants to predict customer renewal.

Previously, we explored logistic regression on this dataset. It's recommended to review the [Build a Logistic Regression Model](https://github.com/diegovillatoromx/logistic_regresion_model) project. Now, we aim to apply the decision tree classifier to the same dataset.
## Architecture

![Architecture Diagram](https://github.com/diegovillatoromx/Customer_Churn_Prediction_Model/blob/main/architecture_diagram.png)


## Data Description
The CSV consists of around 2000 rows and 16 columns in the [dataset](https://github.com/diegovillatoromx/Customer_Churn_Prediction_Model/blob/main/input/data_regression.csv)

### Features:
- Year
- Customer_id - unique id
- Phone_no - customer phone no
- Gender -Male/Female
- Age
- No of days subscribed - the number of days since the subscription
- Multi-screen - does the customer have a single/ multiple screen subscription
- Mail subscription - customer receive mails or not
- Weekly mins watched - number of minutes watched weekly
- Minimum daily mins - minimum minutes watched
- Maximum daily mins - maximum minutes watched
- Weekly nights max mins - number of minutes watched at night time
- Videos watched - total number of videos watched
- Maximum_days_inactive - days since inactive
- Customer support calls - number of customer support calls
- Churn
  - 0 No
  - 1 Yes 

## Modular_Code_Overview

```
  input
    |_data_regression.csv

  ML_pipeline
    |_evaluate_metrics.py
    |_feature.py
    |_ml_model.py
    |_plot_model.py
    |_utils.py

  Tutorial
    |_decision_tree.ipynb

  output
    |_Decision_Tree_plot.png
    |_Feature_Importance.png
    |_model.pkl
```
1. Input - It contains all the data that we have for analysis. There is one csv
file in our case:
   - Data_regression.csv
2. ML_Pipeline
   - The ML_pipeline is a folder that contains all the functions put into different
      python files, which are appropriately named. These python functions are
      then called inside the engine.py file.

3. Output
   – The output folder contains the best-fitted models that we trained
for this data. These models can  be easily loaded and used for future use and
the user need not have to train all the models from the beginning.
Note: This model is built over a chunk of data. One can obtain the model for the
entire data by running engine.py by taking the entire data to train the models.

4. Tutorial - This is a reference folder. It contains the ipython notebook tutorial.

## Installation

Below are the steps required to set up the environment and run this Data Science project on your local machine. Make sure you have the following installed:
- Python 3.x: You can download it from [python.org](https://www.python.org/downloads/).
- Pip: The Python package manager. In most cases, it comes pre-installed with Python. If not, you can install it by following [this guide](https://pip.pypa.io/en/stable/installing/).

### Prerequisites

Install required packages using the requirements.txt file:
``` bash
pip install -r requirements.txt
```
### Installation Steps

1. **Clone the Repository:**

   Clone this repository to your local machine using Git:

   ```bash
   git clone https://github.com/diegovillatoromx/Customer_Churn_Prediction_Model
   cd yourproject
   ```
## Usage

How to utilize and operate the Data Science project after completing the installation steps.
### Data Preparation
Before analysis, prepare data by loading and processing it:
1. ##### Import the required libraries
    ```terminal
    import pickle
    from ML_Pipeline.utils import read_data,inspection,null_values
    from ML_Pipeline.ml_model import prepare_model_smote,run_model
    from ML_Pipeline.evaluate_metrics import confusion_matrix,roc_curve
    from ML_Pipeline.feature_imp import plot_feature_importances
    from ML_Pipeline.plot_model import plot_model
    import matplotlib.pyplot as plt
    ```
2. ##### Data loading
    If data is in CSV format, load it using Pandas:
    ```terminal
    datapath = 'input/data_regression.csv'
    df = read_data(datapath)
    df.head(5)
    ```
    ![df_head](https://github.com/diegovillatoromx/Customer_Churn_Prediction_Model/blob/main/images/dfhead.png)
 
3. #### Inspection and cleaning the data
    ```terminal
    x = inspection(df)
    ```
    ![inspection](https://github.com/diegovillatoromx/Customer_Churn_Prediction_Model/blob/main/images/inspection.png)
4. #### Cleaning and Preprocessing:
   Clean data by handling missing values, normalization, etc.
    ```terminal
    df = null_values(df)
    ```
### Training Model
Perform analysis and modeling on prepared data:

1. #### Model Selection
   Selecting only the numerical columns and excluding the columns we specified in the function
   ```terminal
    X_train, X_test, y_train, y_test = prepare_model_smote(df,class_col='churn',
                                                 cols_to_exclude=['customer_id','phone_no', 'year'])
    ```
### Evaluation

1. #### Evaluation Metrics
   ```terminal
   model_dectree,y_pred = run_model(X_train,X_test,y_train,y_test)
   ```
   ![running_model](https://github.com/diegovillatoromx/Customer_Churn_Prediction_Model/blob/main/images/run_model.png)


2. #### Performance metrics
   ```terminal
   conf_matrix = confusion_matrix(y_test,y_pred)
   ```
   ![running_model](https://github.com/diegovillatoromx/Customer_Churn_Prediction_Model/blob/main/images/cof_matrix.png)

   ```terminal
   roc_val = roc_curve(model_dectree,X_test,y_test)
   ```
   ![ROC](https://github.com/diegovillatoromx/Customer_Churn_Prediction_Model/blob/main/images/Log_ROC.png)

   ```terminal
   decision_tree_plot = plot_model(model_dectree,['not churn','churn'])
   plt.savefig("output/"+"Decision_Tree_plot.png")
   ```
   ![tree](https://github.com/diegovillatoromx/Customer_Churn_Prediction_Model/blob/main/images/Decision_Tree_plot.png)

3. #### Feature Importance
   ```terminal
   fea_imp = plot_feature_importances(model_dectree)
   plt.savefig("output/"+"Feature_Importance.png")
   ```
   ![running_model](https://github.com/diegovillatoromx/Customer_Churn_Prediction_Model/blob/main/images/Feature_Importance.png)


# Contributing

Encourage contributions to the Data Science project.

## How to Contribute
### Fork the Repository:
Fork this repository to your GitHub account.

### Clone Your Fork:
Clone your fork to your local machine:
```
git clone https://github.com/yourusername/yourproject.git
cd yourproject
```
### Create a Branch:
Create a new branch to work on:
```
git checkout -b your-branch-name
```
### Make Changes:
Make necessary changes to the code.

### Testing and Feedback:
Test changes and seek feedback from collaborators.

### Submit a Pull Request:
Send a pull request from your branch to the main repository.

## Contribution Guidelines
1. Focus changes on specific improvements.
2. Follow project's coding style.
3. Provide detailed descriptions in pull requests.
## Reporting Issues
Use "Issues" to report bugs or suggest improvements.

# Contact
For questions or contact, [email](diegovillatormx@gmail.com) or [Twitter](https://twitter.com/diegovillatomx) .



