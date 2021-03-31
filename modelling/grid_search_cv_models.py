"""
    This script performs Randomized Grid Search CV for customer churn model

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
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, roc_auc_score, log_loss
import mlflow
import time
import joblib


def load_data_function():
    """
        Function to load model data
    """
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

    return x_train, x_train_2, x_train_scaled, x_val, x_val_2, x_val_scaled, y_train, y_val


def main():
    # Load model data
    x_train, x_train_2, x_train_scaled, x_val, x_val_2, x_val_scaled, y_train, y_val = load_data_function()

    # Initialize models
    models_dict = {
                   'Logistic_Regression': LogisticRegression(random_state=1),
                   'XGBoost': xgb.XGBClassifier(random_state=2, use_label_encoder=False, n_jobs=-1),
                   'LightGBM': lgb.LGBMClassifier(random_state=3, n_jobs=-1)
                   }

    # Create a dict of parameters to be tuned
    model_params_dict = {'Logistic_Regression': [{'penalty': ['l2'], 'C': [1.0, 0.5]},
                                                 {'penalty': ['l1'], 'C': [1.0, 0.5], 'solver': ['liblinear']}],
                         'RandomForestClassifier': {'n_estimators': [100, 250, 500, 1000],
                                                    'criterion': ['gini', 'entropy'],
                                                    'min_samples_split': [2, 5, 10, 25, 50],
                                                    'min_samples_leaf': [1, 5, 10, 25]},
                         'XGBoost': {'n_estimators': [100, 500, 1000], 'max_depth': [3, 5, 10],
                                     'learning_rate': [0.5, 0.1, 0.01, 0.001], 'subsample': [0.6, 0.8, 1.0]},
                         'LightGBM': {'boosting_type': ['gbdt', 'goss'], 'learning_rate': [0.001, 0.01, 0.1],
                                      'max_depth': [3, 5, 10, 50], 'n_estimators': [50, 100, 500, 1000],
                                      'subsample': [1, 0.8], 'reg_alpha': [0, 0.5], 'reg_lambda': [0, 0.5],
                                      'min_data_in_leaf': [100, 500, 1000], 'num_leaves': [6, 24, 500]}}

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

    # Create a dict for grid search
    models_grid_search = dict()

    start_time = time.time()

    for model in models_dict.items():
        print(f"Running {model[0]}")
        # Start mlflow run
        with mlflow.start_run(run_name="Grid search: " + model[0]):
            models_grid_search[model[0]] = RandomizedSearchCV(estimator=model[1],
                                                              param_distributions=model_params_dict[model[0]],
                                                              n_iter=50,
                                                              scoring=['neg_log_loss', 'f1', 'roc_auc'],
                                                              cv=5,
                                                              refit='f1',
                                                              verbose=3)

            if model[0] == "LightGBM":
                models_grid_search[model[0]].fit(eval(model_data[model[0]]), y_train,
                                                 categorical_feature=[eval(model_data[model[0]]).shape[1]-1]
                                                 )
            else:
                models_grid_search[model[0]].fit(eval(model_data[model[0]]), y_train)

            # Get best cv score
            mean_cv_score = np.mean(models_grid_search[model[0]].best_score_)
            std_cv_score = np.std(models_grid_search[model[0]].best_score_)

            # Get scores for validation data
            val_pred_prob = models_grid_search[model[0]].best_estimator_.predict_proba(eval(val_data[model[0]]))[:, -1]
            val_f1_score = f1_score(y_val,
                                    models_grid_search[model[0]].best_estimator_.predict(eval(val_data[model[0]])))
            val_roc_score = roc_auc_score(y_val, val_pred_prob)
            val_log_loss = log_loss(y_val, val_pred_prob)

            # Log metrics
            mlflow.log_metrics({'mean_cv_f1_score': mean_cv_score,
                                'std_cv_f1_score': std_cv_score,
                                'validation_f1_score': val_f1_score,
                                'validation_auc_roc': val_roc_score,
                                'validation_log_loss': val_log_loss})

            # Store the best models
            Path.cwd().parents[0].joinpath('saved_models_randomized_cv_search').mkdir(parents=True, exist_ok=True)
            joblib.dump(models_grid_search[model[0]],
                        Path.cwd().parents[0].joinpath('saved_models_randomized_cv_search',
                                                       model[0].lower() + '.joblib'))

    print(f"Run time: {np.round(time.time() - start_time, 4)}")
    print("Finished modelling")


if __name__ == "__main__":
    main()
