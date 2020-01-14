from train_utils.training import train_evaluate_anchored
from time import time
from exp1_baseline.models import MLP_NN, LSTM_NN
import pickle


def run_exp(model, train_epochs=20, window=10, output_path=None, splits=[0], horizon=0):
    a = time()
    results = train_evaluate_anchored(model, window=window, train_epochs=train_epochs, horizon=horizon,
                                      splits=splits)
    b = time()
    print("Elapsed time = ", b - a)


    with open(output_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

model = lambda: MLP_NN()
run_exp(model, output_path="results/mlp_012_0.pickle", splits=[0, 1, 2, 3], horizon=0)

model = lambda: LSTM_NN()
run_exp(model, output_path="results/lstm_012_0.pickle", splits=[0, 1, 2, 3], horizon=0)

