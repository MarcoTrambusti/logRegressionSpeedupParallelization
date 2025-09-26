import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from config import DATASETS_FOLDER, CV_RESULTS_PATH
from data_utils import load_data


def cross_validation_lambda(X, y, lambdas, k_folds=5):
    param_grid = {'C': [1 / l for l in lambdas]}
    model = LogisticRegression(penalty='l2', fit_intercept=False, solver='lbfgs', max_iter=100000)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=k_folds, scoring='neg_log_loss', n_jobs=-1)
    grid_search.fit(X, y)

    best_lambda = 1 / grid_search.best_params_['C']
    return best_lambda


def cross_validate_on_datasets():
    results = []
    for file in os.listdir(os.fsencode(DATASETS_FOLDER)):
        filename = file.decode('utf-8')
        print(f'--------- DATASET: {filename} ---------')

        X, y = load_data(filename)
        lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]

        best_lambda = cross_validation_lambda(X, y, lambdas)

        results.append({
            'dataset': filename,
            'best_lambda': best_lambda
        })
        print(f"Best lambda: {best_lambda}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(CV_RESULTS_PATH, index=False)
    print("I risultati sono stati salvati in " + CV_RESULTS_PATH)


if __name__ == '__main__':
    cross_validate_on_datasets()
