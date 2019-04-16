from utils.data_utils_kitti import load_kitti_sequences
import tensorflow as tf
from methods.dpf_kitti import DPF
from utils.exp_utils_kitti import get_default_hyperparams
import numpy as np
import os

def run_cross_validation(i):

    print('RUNNING CROSS VALIDATION TRAINING FOR TESTING {}'.format(i))

    model_path = '../models/tmp/cross_validation_ind_e2e/model_trained_deepvo_{}'.format(i)

    # training_subsequences = [j for j in range(11) if j not in [i]]
    training_subsequences = [0, 2, 8, 9, 10]

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
    hyperparams['train']['split_ratio'] = 0.8  # -> 18/2 split

    method = DPF(**hyperparams['global'])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:
        # train method and save result in model_path
        method.fit(session, data, model_path, plot=False, **hyperparams['train'])

if __name__ == '__main__':
    # for i in range(11):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run_cross_validation(0)
