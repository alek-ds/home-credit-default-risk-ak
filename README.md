# Home Credit Default Risk

Author: Aleksander Kopera
Data used: Anna Montoya, inversion, KirillOdintsov, and Martin Kotek. Home Credit Default Risk. https://kaggle.com/competitions/home-credit-default-risk, 2018. Kaggle.

## Project Overwiev
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
├── data/
├── notebooks/
├── src/
├── models/
├── app/
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
Raw data is not tracked in Git