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

        # ---------- SKLEARN ----------
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

        # Save sklearn reference result
        baseline_result = {
            'name': 'sklearn',
            'n_rows': n_rows,
            'n features': n_features,
            'n_threads': 1,
            'time': round(t, 2),
            'grad_time': round(t, 2),
            'loss_time': round(t, 2),
            'speedup': 1,
            'grad_speedup': 1,
            'loss_speedup': 1,
            'efficiency': 1,
            'grad_efficiency': 1,
            'loss_efficiency': 1,
            'iterations': 0,
            'loss': round(loss_total, 4),
            'grad_norm': np.linalg.norm(gradiente)
        }
        results.append(baseline_result)

        # ---------- CUSTOM SEQUENTIAL METHODS ----------
        lr = LogisticRegressionL2(lambda_reg=lambda_reg)

        seq_armijo_total, seq_armijo_grad, seq_armijo_loss = 0, 0, 0
        for _ in range(n_measurements):
            lr.initialize_params(n_features)

            k = lr.gradient_descent_armijo(X, y)

            seq_armijo_total += lr.total_time
            seq_armijo_grad += lr.gradient_time
            seq_armijo_loss += lr.loss_time

        avg_total_armijo = seq_armijo_total / n_measurements
        avg_grad_armijo = seq_armijo_grad / n_measurements
        avg_loss_armijo = seq_armijo_loss / n_measurements

        loss_total = lr.compute_loss(X, y, lr.w)
        gradiente = lr.compute_loss_gradient(X, y, lr.w)
        results.append({
            'name': 'GD with Armijo LS',
            'n_rows': n_rows,
            'n features': n_features,
            'n_threads': 1,
            'time': round(avg_total_armijo, 2),
            'grad_time': round(avg_grad_armijo, 2),
            'loss_time': round(avg_loss_armijo, 2),
            'speedup': 1,
            'grad_speedup': 1,
            'loss_speedup': 1,
            'efficiency': 1,
            'grad_efficiency': 1,
            'loss_efficiency': 1,
            'iterations': k,
            'loss': round(loss_total, 4),
            'grad_norm': np.linalg.norm(gradiente)
        })

        # ---------- CONJUGATE GRADIENT WOLFE ----------
        seq_wolfe_total, seq_wolfe_grad, seq_wolfe_loss = 0, 0, 0
        for _ in range(n_measurements):
            lr.initialize_params(n_features)

            k = lr.conjugate_gradient_wolfe(X, y)

            seq_wolfe_total += lr.total_time
            seq_wolfe_grad += lr.gradient_time
            seq_wolfe_loss += lr.loss_time

        avg_total_wolfe = seq_wolfe_total / n_measurements
        avg_grad_wolfe = seq_wolfe_grad / n_measurements
        avg_loss_wolfe = seq_wolfe_loss / n_measurements

        loss_total = lr.compute_loss(X, y, lr.w)
        gradiente = lr.compute_loss_gradient(X, y, lr.w)
        results.append({
            'name': 'CG with Wolfe LS',
            'n_rows': n_rows,
            'n features': n_features,
            'n_threads': 1,
            'time': round(avg_total_wolfe, 2),
            'grad_time': round(avg_grad_wolfe, 2),
            'loss_time': round(avg_loss_wolfe, 2),
            'speedup': 1,
            'grad_speedup': 1,
            'loss_speedup': 1,
            'efficiency': 1,
            'grad_efficiency': 1,
            'loss_efficiency': 1,
            'iterations': k,
            'loss': round(loss_total, 4),
            'grad_norm': np.linalg.norm(gradiente)
        })

        # ---------- PARALLEL CUDA ----------
        for tpb in threads_per_block_list:
            blocks = (n_rows + tpb - 1) // tpb
            reg_blocks = (n_features + tpb - 1) // tpb
            blocks_r = (n_rows + tpb - 1) // tpb
            blocks_f = (n_features + tpb - 1) // tpb
            total_blocks = blocks + reg_blocks + blocks_r + blocks_f

            # Armijo CUDA
            lr_cuda = LogisticRegressionL2Cuda(lambda_reg=lambda_reg, TPB=tpb)

            par_armijo_total, par_armijo_grad, par_armijo_loss = 0, 0, 0
            for _ in range(n_measurements):
                lr_cuda.initialize_params(n_features)
                lr_cuda.cache_data(X, y)
                lr_cuda.cache_buffers(*X.shape)

                k = lr_cuda.gradient_descent_armijo(X, y)
                par_armijo_total += lr_cuda.total_time
                par_armijo_grad += lr_cuda.gradient_time
                par_armijo_loss += lr_cuda.loss_time

            avg_par_total = par_armijo_total / n_measurements
            avg_par_grad = par_armijo_grad / n_measurements
            avg_par_loss = par_armijo_loss / n_measurements

            results.append({
                'name': f'GD CUDA TPB {tpb}',
                'n_rows': n_rows,
                'n features': n_features,
                'n_threads': tpb,
                'time': round(avg_par_total, 2),
                'grad_time': round(avg_par_grad, 2),
                'loss_time': round(avg_par_loss, 2),
                'speedup': round(avg_total_armijo / avg_par_total, 2),
                'grad_speedup': round(avg_grad_armijo / avg_par_grad, 2),
                'loss_speedup': round(avg_loss_armijo / avg_par_loss, 2),
                'efficiency': round((avg_total_armijo / avg_par_total) / total_blocks, 4),
                'grad_efficiency': round((avg_grad_armijo / avg_par_grad) / total_blocks, 4),
                                'loss_efficiency': round((avg_loss_armijo / avg_par_loss) / total_blocks, 4),
                'iterations': k,
                'loss': round(lr_cuda.compute_loss(X, y, lr_cuda.w), 4),
                'grad_norm': np.linalg.norm(lr_cuda.compute_loss_gradient(X, y, lr_cuda.w))
            })

            # Wolfe CUDA
            par_wolfe_total, par_wolfe_grad, par_wolfe_loss = 0, 0, 0
            for _ in range(n_measurements):
                lr_cuda.initialize_params(n_features)
                lr_cuda.cache_data(X, y)
                lr_cuda.cache_buffers(*X.shape)

                k = lr_cuda.conjugate_gradient_wolfe(X, y)
                par_wolfe_total += lr_cuda.total_time
                par_wolfe_grad += lr_cuda.gradient_time
                par_wolfe_loss += lr_cuda.loss_time

            avg_par_total = par_wolfe_total / n_measurements
            avg_par_grad = par_wolfe_grad / n_measurements
            avg_par_loss = par_wolfe_loss / n_measurements

            results.append({
                'name': f'CG CUDA TPB {tpb}',
                'n_rows': n_rows,
                'n features': n_features,
                'n_threads': tpb,
                'time': round(avg_par_total, 2),
                'grad_time': round(avg_par_grad, 2),
                'loss_time': round(avg_par_loss, 2),
                'speedup': round(avg_total_wolfe / avg_par_total, 2),
                'grad_speedup': round(avg_grad_wolfe / avg_par_grad, 2),
                'loss_speedup': round(avg_loss_wolfe / avg_par_loss, 2),
                'efficiency': round((avg_total_wolfe / avg_par_total) / total_blocks, 4),
                'grad_efficiency': round((avg_grad_wolfe / avg_par_grad) / total_blocks, 4),
                'loss_efficiency': round((avg_loss_wolfe / avg_par_loss) / total_blocks, 4),
                'iterations': k,
                'loss': round(lr_cuda.compute_loss(X, y, lr_cuda.w), 4),
                'grad_norm': np.linalg.norm(lr_cuda.compute_loss_gradient(X, y, lr_cuda.w))
            })

            del lr_cuda

        # Cleanup
        del X, y, model, w, z, sigma, r, gradiente, loss_sklearn, reg_term, loss_total, lr
        gc.collect()

        # ---------- WRITE RESULTS TO FILE ----------
        with open(os.path.join(RESULT_FOLDER, TEST_RESULTS_PATH), 'w', newline='') as csvfile:
            fieldnames = [
                'name', 'n_rows', 'n features', 'n_threads',
                'time', 'grad_time', 'loss_time',
                'speedup', 'grad_speedup', 'loss_speedup',
                'efficiency', 'grad_efficiency', 'loss_efficiency',
                'iterations', 'loss', 'grad_norm'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)


if __name__ == '__main__':
    #test_loss()
    plotter.plot_and_save()
