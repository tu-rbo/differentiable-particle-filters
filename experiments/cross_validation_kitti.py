from utils.data_utils_kitti import load_kitti_sequences
import tensorflow as tf
from methods.dpf_kitti import DPF
from utils.exp_utils_kitti import get_default_hyperparams
import numpy as np

def run_cross_validation(i):

    print('RUNNING CROSS VALIDATION TRAINING FOR TESTING {}'.format(i))

    model_path = '../models/tmp/cross_validation_ind_e2e/model_trained_ex_{}'.format(i)

    training_subsequences = [j for j in range(11) if j not in [i]]

    # Load all subsequences
    data = load_kitti_sequences(training_subsequences)

    # Assign weights to all subsequences based on the length of the subsequence
    weights = np.zeros((data['seq_num'].shape[0],))
    weights[0] = data['seq_num'][0]
    weights[1:] = data['seq_num'][1:] - data['seq_num'][:-1]
    weights = weights/data['seq_num'][-1]
    data['weights'] = weights

    # reset tensorflow graph
    tf.reset_default_graph()

    # instantiate method
    hyperparams = get_default_hyperparams()
    hyperparams['train']['split_ratio'] = 0.9  # -> 18/2 split

    method = DPF(**hyperparams['global'])

    with tf.Session() as session:
        # train method and save result in model_path
        method.fit(session, data, model_path, plot=False, **hyperparams['train'])

if __name__ == '__main__':
    for i in range(11):
        run_cross_validation(i)
