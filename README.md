# Customer Churn Prediction Model using Decision Trees

Our case study focuses on a churn dataset, where "churned customers" are those ending relationships with their current company. XYZ, a service provider, offers a one-year subscription plan and wants to predict customer renewal.

## Table of Contents

- [Description](#description)
- [Architecture](#architecture)
- [Features](#features)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
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
The CSV consists of around 2000 rows and 16 columns in the [dataset](https://github.com/diegovillatoromx/logistic_regresion_model/blob/main/Data/data_regression.csv)

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
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
   ```
