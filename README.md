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

A Decision Tree is a supervised learning method suitable for both classification and regression tasks, often used for solving classification problems. It's structured as a tree-like classifier, with nodes representing dataset features, branches as decision rules, and leaf nodes as outcomes.

This graphical representation offers potential solutions based on given conditions. Beginning from a root node, it grows branches and forms a tree structure. The tree poses questions and branches further based on Yes/No answers.

Our case study focuses on a churn dataset, where "churned customers" are those ending relationships with their current company. XYZ, a service provider, offers a one-year subscription plan and wants to predict customer renewal.

Previously, we explored logistic regression on this dataset. It's recommended to review the [Build a Logistic Regression Model](https://github.com/diegovillatoromx/logistic_regresion_model) project. Now, we aim to apply the decision tree classifier to the same dataset.
## Architecture

![Architecture Diagram](https://github.com/diegovillatoromx/Customer_Churn_Prediction_Model/blob/main/architecture_diagram.png)

Diagram description: Briefly explain the components and major flows in your architecture. You can mention process stages, technologies used, and how they connect.

## Data Description
The CSV consists of around 2000 rows and 16 columns in the [dataset]
(https://github.com/diegovillatoromx/Customer_Churn_Prediction_Model/blob/main/input/data_regression.csv)

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

  src
    |_Engine.py
    |_ML_pipeline
              |_encoding.py
              |_evaluate_metrics.py
              |_feature_engg.py
              |_imbalanced_data.py
              |_ml_model.py
              |_stats_model.py
              |_rescale_variables.py
              |_scaler.py
              |_train_model.py
              |_utils.py

  lib
    |_logistic_regresion.ipynb

  output
    |_adjusted_model.pkl
    |_balanced_model1.pkl
    |_balanced_model2.pkl
    |_log_ROC.pkl
    |_model_rfe_feat.pkl
    |_model_stats.pkl
    |_model_var_feat.pkl
    |_model1.pkl
    |_smote_model.pkl
```
1. Data Folder - It contains all the data that we have for analysis. There is one csv
file in our case:
   - Data_regression.csv
2. Src folder -This is the most important folder of the project. This folder contains
all the modularized code for all the above steps in a modularized manner. This
folder consists of:
   - Engine.py
   - ML_Pipeline
     - The ML_pipeline is a folder that contains all the functions put into different
      python files, which are appropriately named. These python functions are
      then called inside the engine.py file.

3. Output folder – The output folder contains the best-fitted models that we trained
for this data. These models can  be easily loaded and used for future use and
the user need not have to train all the models from the beginning.
Note: This model is built over a chunk of data. One can obtain the model for the
entire data by running engine.py by taking the entire data to train the models.

4. Lib folder - This is a reference folder. It contains the [ipython notebook tutorial](https://github.com/diegovillatoromx/logistic_regresion_model/blob/main/lib/logistic_regression.ipynb).


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

