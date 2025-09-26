import os
import time
import gc
import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from config import TEST_RESULTS_PATH, CV_RESULTS_PATH, DATASETS_FOLDER
from data_utils import load_data
from logistic_regression_l2 import LogisticRegressionL2


def load_lambda(dataset_name, csv_path=CV_RESULTS_PATH):
    df = pd.read_csv(csv_path)
    row = df[df['dataset'] == dataset_name]
    return float(row['best_lambda'].values[0])


def test_loss():
    results = []
    for file in os.listdir(os.fsencode(DATASETS_FOLDER)):
        filename = file.decode('utf-8')
        print('--------- DATASET: ', filename, ' ---------')
        X, y = load_data(filename)
        n_rows = X.shape[0]
        n_features = X.shape[1]
        print("shape X", X.shape)
        lambda_reg = load_lambda(filename)

        # Modello sklearn con regolarizzazione L2
        start_time = time.time()
        model = LogisticRegression(C=1 / lambda_reg, penalty='l2', fit_intercept=False, solver='lbfgs', max_iter=100000)
        model.fit(X, y)
        t = (time.time() - start_time)
        w = model.coef_.flatten()
        z = X @ w
        sigma = 1 / (1 + np.exp(-y * z))
        r = - y * (1 - sigma)
        gradiente = X.T @ r + lambda_reg * w
        loss_sklearn = log_loss(y, model.predict_proba(X), normalize=False)
        reg_term = 0.5 * lambda_reg * np.sum(model.coef_ ** 2)
        loss_total = loss_sklearn + reg_term
        results.append({'name': 'sklearn', 'n_rows': n_rows, 'n features': n_features, 'time': round(t, 2), 'iterations': 0,
                        'loss': round(loss_total, 4), 'grad_norm': np.linalg.norm(gradiente)})
        print("Loss Sklearn:", loss_total)

        # Test calsse custom
        lr = LogisticRegressionL2(lambda_reg=lambda_reg)

        lr.initialize_params(n_features)
        t, k = lr.gradient_descent_armijo(X, y)
        print("Execution time: ", t, "seconds")
        print("iterazioni:", k)
        loss_total = lr.compute_loss(X, y, lr.w)
        print("loss GD: ", loss_total)
        gradiente = lr.compute_loss_gradient(X, y, lr.w)
        results.append(
            {'name': 'GD with Armijo LS', 'n_rows': n_rows, 'n features': n_features, 'time': round(t, 2), 'iterations': k,
             'loss': round(loss_total, 4), 'grad_norm': np.linalg.norm(gradiente)})

        lr.initialize_params(n_features)
        t, k = lr.conjugate_gradient_wolfe(X, y)
        print("Execution time: ", t, "seconds")
        print("iterazioni:", k)
        loss_total = lr.compute_loss(X, y, lr.w)
        print("loss CG: ", loss_total)
        gradiente = lr.compute_loss_gradient(X, y, lr.w)

        results.append(
            {'name': 'CG with Wolfe LS', 'n_rows': n_rows, 'n features': n_features, 'time':  round(t, 2), 'iterations': k,
             'loss': round(loss_total, 4), 'grad_norm': np.linalg.norm(gradiente)})
        del X, y, model, w, z, sigma, r, gradiente, loss_sklearn, reg_term, loss_total, lr
        gc.collect()

        with open(TEST_RESULTS_PATH, 'w', newline='') as csvfile:
            fieldnames = ['name', 'n_rows', 'n features', 'time', 'iterations', 'loss', 'grad_norm']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)


if __name__ == '__main__':
    test_loss()
