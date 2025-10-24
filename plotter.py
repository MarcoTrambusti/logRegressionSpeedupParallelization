import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Inserisci qui i tuoi dati CSV
import config
import matplotlib
matplotlib.use('Agg')


# Funzione per creare e salvare il grafico
def plot_and_save():
    csv_path = os.path.join(config.RESULT_FOLDER, config.TEST_RESULTS_PATH)

    # Controlla se il file esiste
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Il file CSV non esiste: {csv_path}")

    # Leggi i dati
    df = pd.read_csv(csv_path)

    # Filtra GD e CG CUDA
    gd_cuda = df[df['name'].str.startswith('GD CUDA')]
    cg_cuda = df[df['name'].str.startswith('CG CUDA')]

    for metric, ylabel, suffix in [
        ('speedup', 'Speedup (log scale)', 'speedup'),
        ('efficiency', 'Efficiency (log scale)', 'efficiency'),
        ('loss_speedup', 'Loss Speedup (log scale)', 'loss_speedup'),
        ('loss_efficiency', 'Loss Efficiency (log scale)', 'loss_efficiency'),
        ('grad_speedup', 'Gradient Speedup (log scale)', 'grad_speedup'),
        ('grad_efficiency', 'Gradient Efficiency (log scale)', 'grad_efficiency')
    ]:
        for method_prefix, filename, title in [
            ('GD CUDA', f'{config.RESULT_FOLDER}/gd_cuda_{suffix}.png', f'{ylabel} vs Threads - GD CUDA'),
            ('CG CUDA', f'{config.RESULT_FOLDER}/cg_cuda_{suffix}.png', f'{ylabel} vs Threads - CG CUDA')
        ]:
            # Filtra i dati CUDA e aggiungi anche i casi sequenziali corrispondenti
            cuda_data = df[df['name'].str.startswith(method_prefix)]
            base_method = 'GD with Armijo LS' if method_prefix == 'GD CUDA' else 'CG with Wolfe LS'
            base_data = df[df['name'] == base_method]
            combined = pd.concat([cuda_data, base_data])

            plt.figure(figsize=(10, 6))
            for (n_rows, n_features), group in combined.groupby(['n_rows', 'n features']):
                label = f"{n_rows} rows × {n_features} features"
                group_sorted = group.sort_values('n_threads')
                plt.plot(group_sorted['n_threads'], group_sorted[metric], marker='o', label=label)

            plt.yscale('log')  # Scala logaritmica sull'asse Y
            plt.xlabel('Threads')
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()
