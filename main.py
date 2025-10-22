import os
import time
import gc
import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import plotter
from config import TEST_RESULTS_PATH, DATASETS_FOLDER, RESULT_FOLDER
from data_utils import load_data, load_lambda
from logistic_regression_l2 import LogisticRegressionL2
from logistic_regression_l2_cuda import LogisticRegressionL2Cuda


def test_loss():
    results = []
    threads_per_block_list = [32, 64, 128, 256, 512, 1024]
    n_measurements = 5

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
        results.append({'name': 'sklearn', 'n_rows': n_rows, 'n features': n_features, 'n_threads': 1, 'time': round(t, 2), 'speedup': 1, 'efficiency': 1, 'iterations': 0,
                        'loss': round(loss_total, 4), 'grad_norm': np.linalg.norm(gradiente)})

        # Test calsse custom
        lr = LogisticRegressionL2(lambda_reg=lambda_reg)
        total_time = 0
        for _ in range(n_measurements):
            lr.initialize_params(n_features)
            t, k = lr.gradient_descent_armijo(X, y)
            total_time += t
        avgDurationSeqArmijo = total_time / n_measurements
        loss_total = lr.compute_loss(X, y, lr.w)
        print("loss GD: ", loss_total)
        gradiente = lr.compute_loss_gradient(X, y, lr.w)
        results.append(
            {'name': 'GD with Armijo LS', 'n_rows': n_rows, 'n features': n_features, 'n_threads': 1, 'time': round(avgDurationSeqArmijo, 2), 'speedup': 1, 'efficiency': 1, 'iterations': k,
             'loss': round(loss_total, 4), 'grad_norm': np.linalg.norm(gradiente)})

        total_time = 0
        for _ in range(n_measurements):
            lr.initialize_params(n_features)
            t, k = lr.conjugate_gradient_wolfe(X, y)
            total_time += t
        avgDurationSeqWolfe = total_time / n_measurements
        loss_total = lr.compute_loss(X, y, lr.w)
        print("loss CG: ", loss_total)
        gradiente = lr.compute_loss_gradient(X, y, lr.w)
        results.append(
            {'name': 'CG with Wolfe LS', 'n_rows': n_rows, 'n features': n_features, 'n_threads': 1, 'time':  round(avgDurationSeqWolfe, 2), 'speedup': 1, 'efficiency': 1, 'iterations': k,
             'loss': round(loss_total, 4), 'grad_norm': np.linalg.norm(gradiente)})

        for tpb in threads_per_block_list:
            blocks = (n_rows + tpb - 1) // tpb
            reg_blocks = (w.size + tpb - 1) // tpb
            blocks_r = (n_rows + tpb - 1) // tpb
            blocks_f = (n_features + tpb - 1) // tpb

            lr_cuda = LogisticRegressionL2Cuda(lambda_reg=lambda_reg, TPB=tpb)
            total_time = 0
            for _ in range(n_measurements):
                lr_cuda.initialize_params(n_features)
                lr_cuda.cache_data(X, y)
                lr_cuda.cache_buffers(*X.shape)
                t, k = lr_cuda.gradient_descent_armijo(X, y)
                total_time += t

            avgDurationParArmijo = total_time / n_measurements
            speedup = avgDurationSeqArmijo / avgDurationParArmijo
            efficiency = speedup / ((blocks + reg_blocks + blocks_r + blocks_f) * tpb)
            loss_total = lr_cuda.compute_loss(X, y, lr_cuda.w)
            gradiente = lr_cuda.compute_loss_gradient(X, y, lr_cuda.w)
            results.append({'name': f'GD CUDA TPB {tpb}', 'n_rows': n_rows, 'n features': n_features, 'n_threads': tpb, 'time': round(avgDurationParArmijo, 2), 'speedup': round(speedup, 2), 'efficiency': efficiency, 'iterations': k, 'loss': round(loss_total, 4), 'grad_norm': np.linalg.norm(gradiente)})

            total_time = 0
            for _ in range(n_measurements):
                lr_cuda.initialize_params(n_features)
                lr_cuda.cache_data(X, y)
                lr_cuda.cache_buffers(*X.shape)
                t, k = lr_cuda.conjugate_gradient_wolfe(X, y)
                total_time += t

            avgDurationParWolfe = total_time / n_measurements
            speedup = avgDurationSeqWolfe / avgDurationParWolfe
            efficiency = speedup / ((blocks + reg_blocks + blocks_r + blocks_f) * tpb)
            loss_total = lr_cuda.compute_loss(X, y, lr_cuda.w)
            gradiente = lr_cuda.compute_loss_gradient(X, y, lr_cuda.w)
            results.append({'name': f'CG CUDA TPB {tpb}', 'n_rows': n_rows, 'n features': n_features, 'n_threads': tpb, 'time': round(avgDurationParWolfe, 2), 'speedup': round(speedup, 2), 'efficiency': efficiency, 'iterations': k, 'loss': round(loss_total, 4), 'grad_norm': np.linalg.norm(gradiente)})
            del lr_cuda

        del X, y, model, w, z, sigma, r, gradiente, loss_sklearn, reg_term, loss_total, lr
        gc.collect()

        with open(RESULT_FOLDER + '/' + TEST_RESULTS_PATH, 'w', newline='') as csvfile:
            fieldnames = ['name', 'n_rows', 'n features', 'n_threads', 'time', 'speedup', 'efficiency', 'iterations', 'loss', 'grad_norm']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)


if __name__ == '__main__':
    test_loss()
    plotter.plot_and_save()
