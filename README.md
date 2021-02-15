# auto-insurance
ML model to predict which of the customers were likely to be in a Car Crash or not

# Data
The data is available on the [Auto Insurance Kaggle Challenge](https://www.kaggle.com/c/auto-insurance-fall-2017/data) and we wish to predict the 'TARGET_FLAG' column

# Solution
1. The notebook `auto_insurance_DS.ipynb` has the end-to-end code for :
      1. Loading data and checking for duplicates
      2. Exploratory Data Analysis with Visualizations
      3. Correlation tests against TARGET_FLAG
      4. Imputation Techniques to handle missing data (MICE Imputation)
      5. Feature Engineering and Feature Encoding
      6. Model Comparison and Selection (ROC AUC)
           1. Logistic Regression
           2. RandomForest Classifier
           3. XGBoost Classifier
      7. Hyperparameter Tuning and Model Validation
      8. Prediction on test data and saving to csv 
      
      All the analysis, observation and comments are documented within the Python notebook itself.

2. There are 2 Python scripts under `src/` :
      1. AutoInsurance.py
      2. Train.py
      
      The model file is stored in `model/`
      
      The 2 scripts perform all the steps in A) and explicitly output the performance of algorithms. 
      Additionally, they save the model after training, make predictions on test data and save them to `predicted_TARGET_FLAG.csv`
  
      To run the scripts :
      - Download data and place it in root directory
      - cd into `src/` and run : 

            `python -W ignore AutoInsurance.py --train`

