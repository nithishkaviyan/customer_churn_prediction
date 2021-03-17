"""
    This script creates a baseline model for customer churn model

    Dependencies:
        ../model_data_prep/model_data_prep.py
        ../model_data_prep/model_data_eda_and_feature_engg.ipynb
        ../model_data_prep/data_cleaning_and_data_split.ipynb
        data_downsampling.py
"""

# Load packages
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, log_loss
import mlflow


# Load train dataset
data_folder = Path.cwd().parents[0].joinpath('data', 'processed_data')
x_train = np.load(data_folder.joinpath('x_train_onehot_downsampled.npy'))
y_train = np.load(data_folder.joinpath('y_train_downsampled.npy'))

# Load validation dataset
x_val = np.load(data_folder.joinpath('x_val_onehot_encode.npy'))
y_val = np.load(data_folder.joinpath('y_val.npy'))

# Scale data
data_scaler = StandardScaler()
data_scaler.fit(x_train)
x_train_scaled = data_scaler.transform(x_train)
x_val_scaled = data_scaler.transform(x_val)

# Load numeric encoded dataset
x_train_2 = np.load(data_folder.joinpath('x_train_num_downsampled.npy'))
x_val_2 = np.load(data_folder.joinpath('x_val_num_encode.npy'))

# Initialize models
models_dict = {'Logistic_Regression': LogisticRegression(random_state=1),
               'XGBoost': xgb.XGBClassifier(random_state=2),
               'RandomForestClassifier': RandomForestClassifier(random_state=3),
               'LightGBM': lgb.LGBMClassifier(random_state=4)
               }

# Choose train dataset
model_data = {'Logistic_Regression': 'x_train_scaled',
              'XGBoost': 'x_train',
              'RandomForestClassifier': 'x_train',
              'LightGBM': 'x_train_2'
              }

# Choose validation dataset
val_data = {'Logistic_Regression': 'x_val_scaled',
            'XGBoost': 'x_val',
            'RandomForestClassifier': 'x_val',
            'LightGBM': 'x_val_2'}

# Develop baseline models
for model in models_dict.items():
    print(f"Running {model[0]}")
    with mlflow.start_run(run_name="Baseline: " + model[0]):
        # Compute cross validation score
        cv_score = cross_val_score(estimator=model[1], X=eval(model_data[model[0]]), y=y_train, scoring='f1', cv=5)

        # Get cv mean and std score
        mean_cv_score = np.mean(cv_score)
        std_cv_score = np.std(cv_score)

        # Fit model
        model[1].fit(eval(model_data[model[0]]), y_train)

        # Get scores for validation data
        val_pred_prob = model[1].predict_proba(eval(val_data[model[0]]))[:, -1]
        val_f1_score = f1_score(y_val, model[1].predict(eval(val_data[model[0]])))
        val_roc_score = roc_auc_score(y_val, val_pred_prob)
        val_log_loss = log_loss(y_val, val_pred_prob)

        # Log metrics
        mlflow.log_metrics({'mean_cv_f1_score': mean_cv_score,
                            'std_cv_f1_score': std_cv_score,
                            'validation_f1_score': val_f1_score,
                            'validation_auc_roc': val_roc_score,
                            'validation_log_loss': val_log_loss})
