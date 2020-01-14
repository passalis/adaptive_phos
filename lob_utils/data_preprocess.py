import h5py
import numpy as np


def load_feature_vectors(h5_path='/home/nick/Data/Datasets/hf_lob.h5', n_features=5):
    """
    Loads the dataset and returns the features, the training/testing targets as well as the indices for
    training/testing using both the day split evaluation and the stock split evaluation

    For each time step the most recent n_features feature vectors are extracted (144-dimensional features vectors)

    :param h5_path:
    :param n_features:
    :return: features, targets, day_splits, targets_splits (see below for a detailed description)

    features: an array with (453975, n_features, 144) values for each time step
    targets: an array with (453975, 5) values. Five different predictions targets are extracted
        (only targets 0 (next time step), 3 (next 5 time steps) and 4 (next 10 timesteps)
        are used in the paper for the evaluation, three different target are used: 0 down, 1 stay, 2 up )
    day_splits: a list with 9 items (one for each evaluation setup)
    target_splits: a list with 5 items (one for each evaluation setup)
    Each split is composed of two-values tuple, with the fist value referring to the training indices,
    while the second one referring to the testing indices.
    """
    file = h5py.File(h5_path, 'r', )
    features = file['features']

    mid_feature = int(features.shape[1] / 2 - 1)

    features = np.float32(features[:, mid_feature - n_features + 1:mid_feature + 1, :])
    targets = np.int32(file['targets'])

    # Get the splits for day-based testing
    day_train_split_idx = file['day_train_split_idx'][:].astype('bool')
    day_test_split_idx = file['day_test_split_idx'][:].astype('bool')

    # Get the indices for each stock
    stock_idx = file['stock_test_split_idx'][:].astype('bool')

    return features, targets, day_train_split_idx, day_test_split_idx, stock_idx


if __name__ == '__main__':
    """
    This script creates the intermediate data (without duplicated entries) that can be used with the developed PyTorch loader
    """

    features, targets, day_train_split_idx, day_test_split_idx, stock_idx = load_feature_vectors(n_features=1)
    features = features.reshape(-1, 144)

    file = h5py.File('/home/nick/Data/Workspace/Python/2018/lob_rnn/lob_utils/lob_raw.h5', 'w')
    file.create_dataset('features', data=features, dtype='float32')
    file.create_dataset('targets', data=targets, dtype='int32')
    file.create_dataset('day_train_split_idx', data=day_train_split_idx, dtype='bool')
    file.create_dataset('day_test_split_idx', data=day_test_split_idx, dtype='bool')
    file.create_dataset('stock_idx', data=stock_idx, dtype='bool')
    file.close()
