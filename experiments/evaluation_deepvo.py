import tensorflow as tf
import numpy as np
from methods.deepvo_lstm import DeepVOLSTM
from utils.data_utils import load_data, noisyfy_data, make_batch_iterator, remove_state, wrap_angle
from utils.exp_utils import get_default_hyperparams
from keras import backend as K
import matplotlib.pyplot as plt
import os
import keras

def get_evaluation_stats(model_path='../models/tmp/best_deepvo_model',
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
        # session = method.load(session, model_path, **hyperparams['test'])

        errors = dict()

        test_filenames = ["../data/kitti_tf_records/kitti_{}.tfrecords".format(i) for i in test_trajectories]


        for i, test_traj in enumerate(test_trajectories):

            errors[test_traj] = dict()

            for batch_seq_len in seq_lengths:

                errors[test_traj][batch_seq_len] = {'trans': [], 'rot': []}

                test_dataset = method.generate_test_dataset([test_filenames[i]], batch_seq_len=batch_seq_len)
                handle = tf.placeholder(tf.string, shape=[])
                iterator = tf.data.Iterator.from_string_handle(
                    handle, test_dataset.output_types, test_dataset.output_shapes)
                method.image, method.state = iterator.get_next()

                # test_batch_iterator = make_batch_iterator_for_evaluation(data, start_step, trajectory=i, batch_size=1, seq_len=end_step-start_step)
                test_batch_iterator = test_dataset.make_one_shot_iterator()

                test_handle = session.run(test_batch_iterator.string_handle())

                method.image_input = keras.Input(shape=(batch_seq_len - 1, 384, 1280, 6), tensor=method.image[:, 1:, :, :, :])
                method.connect_modules()
                saver = tf.train.Saver()
                saver.restore(session, model_path)

                # session = method.load(session, model_path, **hyperparams['test'])


                while True:

                    try:

                        prediction, true_state = method.predict(session, handle, test_handle)
                        dist = np.sqrt((true_state[0, -1, 0]-true_state[0, 0, 0])**2 + (true_state[0, -1, 1]-true_state[0, 0, 1])**2)
                        error_x = true_state[0, -1, 0] - prediction[0, 0]
                        error_y = true_state[0, -1, 1] - prediction[0, 1]
                        error_trans = np.sqrt(error_x ** 2 + error_y ** 2) / dist
                        error_rot = abs(wrap_angle(true_state[0, -1, 2] - prediction[0, 2]))/dist * 180 / np.pi
                        errors[test_traj][batch_seq_len]['trans'].append(error_trans)
                        errors[test_traj][batch_seq_len]['rot'].append(error_rot)

                    except tf.errors.OutOfRangeError:
                        print(sum(errors[test_traj][batch_seq_len]['trans'])/len(errors[test_traj][batch_seq_len]['trans']))
                        break

    return errors


def find_all_cross_val_models(model_path):
    import os
    # models = ([name for name in os.listdir(model_path) if not os.path.isfile(os.path.join(model_path, name))])
    # trajs = [int(name.split('_')[3]) for name in models]
    # print (models, trajs)
    # return zip(models, trajs)
    return zip(['best_deepvo_model_loss_last_step_dpf_theta']*4, [0, 2, 8, 9]) # [3, 4, 5, 6, 7]

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
            for measure in ['trans', 'rot']:  #, 'rot'
                e = errors[traj][seq_len][measure]
                mean_error = np.mean(e)
                se_error = np.std(e, ddof=1) / np.sqrt(len(e))
                average_errors[measure][seq_len].append(mean_error)
                print('{:>5} error for seq_len {}: {:.4f} +- {:.4f}'.format(measure, seq_len, mean_error, se_error))

        print('Averaged errors:')
        for measure in ['trans', 'rot']:  #, 'rot'
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    main()
