import tensorflow as tf
import numpy as np
from methods.deepvo_lstm import DeepVOLSTM
from utils.data_utils import load_data, noisyfy_data, make_batch_iterator, remove_state, wrap_angle
from utils.exp_utils import get_default_hyperparams
from keras import backend as K
import matplotlib.pyplot as plt
import os

def train_deepvo():

    # reset tensorflow graph
    tf.reset_default_graph()

    # instantiate method
    hyperparams = get_default_hyperparams()
    method = DeepVOLSTM(**hyperparams['global'])

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:
        # train method and save result in model_path
        K.set_session(session)

        method.fit(session,**hyperparams['train'])


# def test_dpf(task='nav01', data_path='../data/100s', model_path='../models/tmp'):
#
#     # load test data
#     # test_data = load_data(data_path=data_path, filename=task + '_test')
#     # noisy_test_data = noisyfy_data(test_data)
#     # test_batch_iterator = make_batch_iterator(noisy_test_data, seq_len=50)
#
#     # reset tensorflow graph
#     tf.reset_default_graph()
#
#     # instantiate method
#     hyperparams = get_default_hyperparams()
#     method = DeepVOLSTM(**hyperparams['global'])
#
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#
#     with tf.Session(config=config) as session:
#         # load method and apply to new data
#         method.load(session, model_path)
#         for i in range(10):
#             test_batch_input = remove_state(test_batch, provide_initial_state=False)
#             result = method.predict(session, test_batch_input, **hyperparams['test'])

def get_evaluation_stats(model_path='../models/tmp/best_deepvo_weights',
                         test_trajectories=[5, 6, 7, 10], seq_lengths = [32], plot_results=False):

    # data = load_kitti_sequences(test_trajectories)

    # reset tensorflow graph
    tf.reset_default_graph()

    # instantiate method
    hyperparams = get_default_hyperparams()
    method = DeepVOLSTM(**hyperparams['global'])

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:

        # load method and apply to new data
        method.load(session, model_path, **hyperparams['test'])

        errors = dict()

        test_filenames = ["../data/kitti_tf_records/kitti_{}.tfrecords".format(i) for i in test_trajectories]


        for i, test_traj in enumerate(test_trajectories):

            errors[test_traj] = dict()

            for batch_seq_len in seq_lengths:

                errors[test_traj][batch_seq_len] = {'trans': [], 'rot': []}

                try:
                    while True:
                        test_dataset = method.generate_test_dataset(test_filenames[i], batch_seq_len=batch_seq_len)
                        # test_batch_iterator = make_batch_iterator_for_evaluation(data, start_step, trajectory=i, batch_size=1, seq_len=end_step-start_step)
                        test_batch_iterator = test_dataset.make_one_shot_iterator()
                        image, state = test_batch_iterator.get_next()

                        # batch = next(test_batch_iterator)
                        # batch_input = remove_state(batch, provide_initial_state=True)

                        prediction, true_state = method.predict(session, image, state)
                        dist = tf.sqrt((prediction[0, -1, 0]-prediction[0, 0, 0])**2 + (prediction[0, -1, 1]-prediction[0, 0, 1])**2)
                        error_x = true_state[0, -1, 0] - prediction[0, -1, 0]
                        error_y = true_state[0, -1, 1] - prediction[0, -1, 1]
                        error_trans = np.sqrt(error_x ** 2 + error_y ** 2) / dist
                        error_rot = abs(wrap_angle(tf.atan2(true_state[0, -1, 2], true_state[0, -1, 3]) -
                                                   tf.atan2(prediction[0, -1, 2], prediction[0, -1, 3])))/dist * 180 / np.pi

                        errors[test_traj][batch_seq_len]['trans'].append(error_trans)
                        errors[test_traj][batch_seq_len]['rot'].append(error_rot)
                except tf.errors.OutOfRangeError:
                    pass

    return errors


def find_all_cross_val_models(model_path):
    import os
    models = ([name for name in os.listdir(model_path) if not os.path.isfile(os.path.join(model_path, name))])
    trajs = [int(name.split('_')[3]) for name in models]
    print (models, trajs)
    # return zip(models, trajs)
    return zip(['best_deepvo_weights.h5', 'best_deepvo_weights.h5', 'best_deepvo_weights.h5','best_deepvo_weights.h5'], [5, 6, 7, 10])

def main():
    plt.ion()

    errors = dict()
    average_errors = {'trans': {i: [] for i in [32]},   #100, 200, 400, 800
                      'rot': {i: [] for i in [32]}}
    model_path = '../models/tmp/'
    for model, traj in find_all_cross_val_models(model_path):
        print('!!! Evaluatng model {} on trajectory {}'.format(model, traj))
        new_errors = get_evaluation_stats(model_path=model_path+model, test_trajectories=[traj], plot_results=False)
        errors.update(new_errors)
        print('')
        print('Trajectory {}'.format(traj))
        for seq_len in sorted(errors[traj].keys()):
            for measure in ['trans', 'rot']:
                e = errors[traj][seq_len][measure]
                mean_error = np.mean(e)
                se_error = np.std(e, ddof=1) / np.sqrt(len(e))
                average_errors[measure][seq_len].append(mean_error)
                print('{:>5} error for seq_len {}: {:.4f} +- {:.4f}'.format(measure, seq_len, mean_error, se_error))

        print('Averaged errors:')
        for measure in ['trans', 'rot']:
            e_means = []
            e_ses = []
            for seq_len in sorted(average_errors[measure].keys()):
                e = np.array(average_errors[measure][seq_len])
                e = e[~np.isnan(e)]
                mean_error = np.mean(e)
                e_means.append(mean_error)
                se_error = np.std(e, ddof=1) / np.sqrt(len(e))
                e_ses.append(se_error)
                print('{:>5} error for seq_len {}: {:.4f} +- {:.4f}'.format(measure, seq_len, mean_error, se_error))
            print('{:>5} error averaged over seq_lens: {:.4f} +- {:.4f}'.format(measure, np.mean(e_means), np.std(e_means, ddof=1) / np.sqrt(len(e_means))))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

    train_deepvo()
