# This script contains code to use mlflow for churn modelling

# Load packages
from pathlib import Path
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import log_loss
import mlflow


# Read train and val data
data_folder = Path.cwd().parents[0].joinpath('data', 'processed_data')
x_train = np.load(data_folder.joinpath('x_train_downsampled.npy'))
y_train = np.load(data_folder.joinpath('y_train_downsampled.npy'))
x_val = np.load(data_folder.joinpath('x_val.npy'))
y_val = np.load(data_folder.joinpath('y_val.npy'))


with mlflow.start_run():
    model_params = {'n_estimators': 1000,
                    'max_depth': 3,
                    'learning_rate': 0.001,
                    'subsample': 1.0}
    xgb_model = xgb.XGBClassifier(**model_params)
    xgb_model.fit(x_train, y_train, verbose=True, eval_metric='logloss')

    # Evaluate model
    y_pred = xgb_model.predict(x_val)

    f1_score_val = f1_score(y_val, y_pred)
    roc_auc_score_val = roc_auc_score(y_val, y_pred)
    precision_score_val = precision_score(y_val, y_pred)
    recall_score_val = recall_score(y_val, y_pred)
    log_loss_val = log_loss(y_val, y_pred)

    # Log params
    mlflow.log_params(model_params)

    # Log metrics
    mlflow.log_metrics({"log_loss": log_loss_val,
                       "precision": precision_score_val,
                       "recall": precision_score_val,
                       "f1_score": f1_score_val,
                       "roc_auc": roc_auc_score_val})

    mlflow.xgboost.log_model(xgb_model, "XGB model")
