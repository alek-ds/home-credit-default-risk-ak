# Home Credit Default Risk

Author: Aleksander Kopera
Data used: Anna Montoya, inversion, KirillOdintsov, and Martin Kotek. Home Credit Default Risk. https://kaggle.com/competitions/home-credit-default-risk, 2018. Kaggle.

## Project Overview
This project focuses on analysis of credit data in order to improve decion making by creting preditions of credit risk by classifying customers as good or bad.
The final goal is to create an app that will intuitively improve credit acceptance process based on machine learning model.

## Project status
The project is in early-stage:
- project structure setup
- Git initialization
- environment setup
- initial data exploration

## Planned workflow
1. Data Exploration in Jupyter Notebook
2. Data cleaning and preprocessing
3. Feature engineering
4. Model training and evalutaion
5. Creating complete pipeline
6. Streamlit app
7. Docker containerization

## Project structure
```text
home_credit_default_risk/
├── app/
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
├── tests/
├── .gitignore
├── environment.yml
└── README.md
```

## Environment setup
```bash
conda env create -f environment.yml
conda activate home_credits_env
```

## Notes
Raw data is not tracked in Git.
The original data is provided with prior train-test split.
This project will conduct analysis and modeling of train part (with custom train-split) and leave test data for demonstration of results. 

## Notebooks
Notebooks contain logic behind conducted exploratory analysis and feature engineering. For each data set in this project there are number of notebooks which name starts with relevant prefix (e.g. application_train.csv -> app_...).
- application_train.csv
    - app_col_review_missing_analysis - quick review of columns and analysis of variables with substantial number of NAN values in 'application_train.csv'
    - app_univariate_bivariate_app_process - analysis of application process data
    - app_univariate_bivariate_documents_provided - analysis of documentation data
    - app_univariate_bivariate_loan - analysis of loan data
    - app_univariate_bivariate_financial_material - analysis of financial and material data
    - app_univariate_bivariate_family_demographic - analysis of familiy and demographic data
    - app_multivariate - multivarate analysis for application data
    - app_preprocessing - preprocessing of application data
- bureau.csv
    - bureau_col_review_missing_analysis

## src .py files
These files contain all the necessary code to perform analysis and automatize whole pipeline.
- eda_module.py - funtions for exploratory data analysis
- preprocess_module - functions for data preprocessing