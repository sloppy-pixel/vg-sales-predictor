# Video Game Sales Predictor

Predicts global video game sales using a Random Forest model.

## Dataset
Video game sales data from Kaggle — 16k+ games across platforms, genres, and regions.

## Results
- MAE: 0.04 million units
- R2 Score: 0.82

## Features used
- Platform, Genre, Year
- Regional sales (NA, EU, JP, Other)

## Concepts covered
- Data cleaning and preprocessing
- Categorical encoding
- Random Forest regression
- Feature importance analysis

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install pandas scikit-learn matplotlib seaborn
```

## Run
```bash
python explore.py   # inspect the data
python train.py     # train and evaluate
```