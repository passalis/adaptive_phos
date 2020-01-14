from train_utils.training import train_evaluate_anchored
from time import time
from train_utils.initializers import weights_xavier_normal, weights_kaiming_normal, create_custom_initializer
from exp2_photonic.models import Photonic_RNN
import pickle


def run_exp(model, train_epochs=20, window=10, output_path=None, splits=[3], horizon=0, use_adaptive_init=False,
            initializer=None):
    a = time()
    results = train_evaluate_anchored(model, window=window, train_epochs=train_epochs, horizon=horizon,
                                      splits=splits, use_adaptive_init=use_adaptive_init,
                                      weight_initializer=initializer)
    b = time()
    print("Elapsed time = ", b - a)

    with open(output_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


# Sigmoid baseline
model = lambda: Photonic_RNN(activation='sigmoid')
run_exp(model, output_path="results/recurrent_sigmoid_0.pickle", splits=[0, 1, 2, 3], horizon=0, train_epochs=20,
        use_adaptive_init=False, initializer=weights_xavier_normal)

# Photonic - xavier
model = lambda: Photonic_RNN(activation='photonic')
run_exp(model, output_path="results/recurrent_photonic_xavier_0.pickle", splits=[0, 1, 2, 3], horizon=0,
        train_epochs=20, use_adaptive_init=False, initializer=weights_xavier_normal)

# Photonic - kaiming
model = lambda: Photonic_RNN(activation='photonic')
run_exp(model, output_path="results/recurrent_photonic_he_0.pickle", splits=[0, 1, 2, 3], horizon=0, train_epochs=20,
        use_adaptive_init=False, initializer=weights_kaiming_normal)

# Photonic - proposed
model = lambda: Photonic_RNN(activation='photonic')
run_exp(model, output_path="results/recurrent_photonic_xavier_proposed_0_new.pickle", splits=[0, 1, 2, 3], horizon=0,
        train_epochs=20, use_adaptive_init=True, initializer=weights_xavier_normal)

# Photonic - proposed
model = lambda: Photonic_RNN(activation='photonic')
run_exp(model, output_path="results/recurrent_photonic_he_proposed_0_new.pickle", splits=[0, 1, 2, 3], horizon=0,
        train_epochs=20, use_adaptive_init=True, initializer=weights_kaiming_normal)
