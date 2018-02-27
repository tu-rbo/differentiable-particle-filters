import tensorflow as tf

from methods.dpf_kitti import DPF
from methods.odom import OdometryBaseline
from utils.data_utils_kitti import load_data, noisyfy_data, make_batch_iterator, remove_state, split_data, load_kitti_sequences, make_batch_iterator_for_evaluation, wrap_angle, plot_video
from utils.exp_utils_kitti import get_default_hyperparams
import matplotlib.pyplot as plt
import numpy as np

def get_evaluation_stats(model_path='../models/tmp/', test_trajectories=[11], seq_lengths = [100, 200, 400, 800], plot_results=False):

    data = load_kitti_sequences(test_trajectories)

    # reset tensorflow graph
    tf.reset_default_graph()

    # instantiate method
    hyperparams = get_default_hyperparams()
    method = DPF(**hyperparams['global'])

    with tf.Session() as session:

        # load method and apply to new data
        method.load(session, model_path)

        errors = dict()

        for i, test_traj in enumerate(test_trajectories):

            s_test_traj = data['s'][0:data['seq_num'][i*2]]  # take care of duplicated trajectories (left and right camera)
            distance = compute_distance_for_trajectory(s_test_traj)
            errors[test_traj] = dict()

            for seq_len in seq_lengths:

                errors[test_traj][seq_len] = {'trans': [], 'rot': []}

                for start_step in range(0, distance.shape[0], 1):

                    end_step, dist = find_end_step(distance, start_step, seq_len, use_meters=False)  #--> Put use_meters = True for official KITTI benchmark results

                    if end_step == -1:
                        continue

                    # test_batch_iterator = make_batch_iterator(test_data, seq_len=50)
                    test_batch_iterator = make_batch_iterator_for_evaluation(data, start_step, trajectory=i, batch_size=1, seq_len=end_step-start_step)

                    batch = next(test_batch_iterator)
                    batch_input = remove_state(batch, provide_initial_state=True)

                    prediction, particle_list, particle_prob_list = method.predict(session, batch_input, return_particles=True)
                    error_x = batch['s'][0, -1, 0] - prediction[0, -1, 0]
                    error_y = batch['s'][0, -1, 1] - prediction[0, -1, 1]
                    error_trans = np.sqrt(error_x ** 2 + error_y ** 2) / dist
                    error_rot = abs(wrap_angle(batch['s'][0, -1, 2] - prediction[0, -1, 2]))/dist * 180 / np.pi

                    errors[test_traj][seq_len]['trans'].append(error_trans)
                    errors[test_traj][seq_len]['rot'].append(error_rot)

                    if plot_results:

                        dim_names = ['pos', 'theta', 'vel_f', 'vel_th']
                        fig = plt.figure()
                        ax1 = fig.add_subplot(221)
                        ax2 = fig.add_subplot(222)
                        ax3 = fig.add_subplot(223)
                        ax4 = fig.add_subplot(224)

                        for t in range(particle_list.shape[1]):
                            dim = 0
                            ax1.scatter(particle_list[0, t, :, dim], particle_list[0, t, :, dim+1], c=particle_prob_list[0, t, :], cmap='viridis_r', marker='o', s=15, alpha=0.1,
                                                linewidths=0.05,
                                                vmin=0.0,
                                                vmax=0.02)

                            ax1.plot([prediction[0, t, dim]], [prediction[0, t, dim+1]], 'o', markerfacecolor='None', markeredgecolor='b',
                                             markersize=0.5)

                            ax1.plot([batch['s'][0, t, dim]], [batch['s'][0, t, dim+1]], '+', markerfacecolor='None', markeredgecolor='r',
                                             markersize=0.5)

                            ax1.set_aspect('equal')

                            dim = 2
                            ax2.scatter(t * np.ones_like(particle_list[0, t, :, dim]), particle_list[0, t, :, dim], c=particle_prob_list[0, t, :], cmap='viridis_r', marker='o', s=15, alpha=0.1,
                                                linewidths=0.05,
                                                vmin=0.0,
                                                vmax=0.02)
                                                #np.max(
                                                    #s_add_probs_list[s, i, :, 0]))  # , vmin=-1/filter.num_particles,)
                            current_state = prediction[0, t, dim]
                            ax2.plot([t], [current_state], 'o', markerfacecolor='None', markeredgecolor='k',
                                             markersize=2.5)
                            true = batch['s'][0, t, dim]
                            ax2.plot([t], [true], '+', markerfacecolor='None', markeredgecolor='r',
                                             markersize=2.5)

                            dim = 3
                            ax3.scatter(t * np.ones_like(particle_list[0, t, :, dim]), particle_list[0, t, :, dim], c=particle_prob_list[0, t, :], cmap='viridis_r', marker='o', s=15, alpha=0.1,
                                                linewidths=0.05,
                                                vmin=0.0,
                                                vmax=0.02)
                                                #np.max(
                                                    #s_add_probs_list[s, i, :, 0]))  # , vmin=-1/filter.num_particles,)
                            current_state = prediction[0, t, dim]
                            ax3.plot([t], [current_state], 'o', markerfacecolor='None', markeredgecolor='k',
                                             markersize=2.5)
                            true = batch['s'][0, t, dim]
                            ax3.plot([t], [true], '+', markerfacecolor='None', markeredgecolor='r',
                                             markersize=2.5)

                            dim = 4
                            ax4.scatter(t * np.ones_like(particle_list[0, t, :, dim]), particle_list[0, t, :, dim], c=particle_prob_list[0, t, :], cmap='viridis_r', marker='o', s=15, alpha=0.1,
                                                linewidths=0.05,
                                                vmin=0.0,
                                                vmax=0.02)

                            current_state = prediction[0, t, dim]
                            ax4.plot([t], [current_state], 'o', markerfacecolor='None', markeredgecolor='k',
                                             markersize=2.5)
                            true = batch['s'][0, t, dim]
                            ax4.plot([t], [true], '+', markerfacecolor='None', markeredgecolor='r',
                                             markersize=2.5)

                        plt.pause(0.05)

                        ax1.set_title(dim_names[0])
                        ax2.set_title(dim_names[1])
                        ax3.set_title(dim_names[2])
                        ax4.set_title(dim_names[3])

    return errors



def compute_distance_for_trajectory(s):

    # for ii in range(len(output_oxts_file)):
    distance = [0]
    for i in range(1, s.shape[0]):
        diff_x = s[i, 0, 0] - s[i-1, 0, 0]
        diff_y = s[i, 0, 1] - s[i-1, 0, 1]
        dist = distance[-1] + np.sqrt(diff_x ** 2 + diff_y ** 2)
        distance.append(dist)
    distance = np.asarray(distance)
    return distance

def find_end_step(distance, start_step, length, use_meters=True):

    for i in range(start_step, distance.shape[0]):
        if (use_meters and distance[i] > (distance[start_step] + length)) or \
            (not use_meters and (i - start_step) >= length):
            return i, distance[i] - distance[start_step]
    return -1, 0

def find_all_cross_val_models(model_path):
    import os
    models = ([name for name in os.listdir(model_path) if not os.path.isfile(os.path.join(model_path, name))])
    trajs = [int(name.split('_')[3]) for name in models]
    return zip(models, trajs)

def main():
    plt.ion()

    errors = dict()
    average_errors = {'trans': {i: [] for i in [100, 200, 400, 800]},
                      'rot': {i: [] for i in [100, 200, 400, 800]}}
    model_path = '../models/tmp/cross_validation_ind_e2e/'
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
    main()
