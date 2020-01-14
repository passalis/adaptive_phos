from os.path import join
import pickle
import numpy as np
from train_utils.training import  get_average_metrics

def print_line(results):
    metrics = get_average_metrics(results)
    acc, precision, recall, f1, kappa = metrics


    print("$ %3.2f \\pm %3.2f$ & $ %3.2f \\pm %3.2f$ "
          % (
             100 * np.mean(f1), 100 * np.std(f1),
             np.mean(kappa), np.std(kappa)))

def print_results(results_path):
    with open(results_path, 'rb') as f:
        metrics = pickle.load(f)


    print("--------")
    print(results_path)
    print_line(metrics)



print_results('results/recurrent_sigmoid_0.pickle')
print_results('results/recurrent_photonic_xavier_0.pickle')
print_results('results/recurrent_photonic_he_0.pickle')
print_results('results/recurrent_photonic_xavier_proposed_0_new.pickle')
print_results('results/recurrent_photonic_he_proposed_0_new.pickle')

