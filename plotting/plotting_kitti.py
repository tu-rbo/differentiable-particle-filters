import tensorflow as tf

from methods.dpf_kitti import DPF
from methods.odom import OdometryBaseline
from utils.data_utils_kitti import load_data, noisyfy_data, make_batch_iterator, remove_state, split_data, load_kitti_sequences, make_batch_iterator_for_evaluation, wrap_angle, plot_video
from utils.exp_utils_kitti import get_default_hyperparams
import matplotlib.pyplot as plt
import numpy as np

def get_evaluation_stats(model_path='../models/tmp/', test_trajectories=[9], seq_lengths = [100], plot_results=True):

    data = load_kitti_sequences(test_trajectories)
    # data = load_all_data(test_trajectories, train=False)
    # data['o'] = data['o-m']  # flip ops to apply models that were trained on inverted data
    # plot_video(data)

    # reset tensorflow graph
    tf.reset_default_graph()

    # instantiate method
    hyperparams = get_default_hyperparams()
    method = DPF(**hyperparams['global'])

    with tf.Session() as session:

        # load method and apply to new data
        # method.load(session, model_path)

        errors = dict()

        for i, test_traj in enumerate(test_trajectories):
            # pick statest for traj
            s_test_traj = data['s'][0:data['seq_num'][i*2]]  # take care of duplicated trajectories (left and right camera)
            distance = compute_distance_for_trajectory(s_test_traj)
            errors[test_traj] = dict()

            for seq_len in seq_lengths:

                errors[test_traj][seq_len] = {'trans': [], 'rot': []}

                for start_step in range(0, 1):

                    # print('start_step:', start_step)

                    # end_step, dist = find_end_step(distance, start_step, seq_len, use_meters=False)
                    # print('!!!', start_step, seq_len[seq_len], end_step, dist)
                    end_step = distance.shape[0]
                    dist = distance[-1]

                    if end_step == -1:
                        continue

                    # test_batch_iterator = make_batch_iterator(test_data, seq_len=50)
                    test_batch_iterator = make_batch_iterator_for_evaluation(data, start_step, trajectory=0, batch_size=1, seq_len=end_step-start_step)

                    batch = next(test_batch_iterator)
                    # batch_input = remove_state(batch, provide_initial_state=True)

                    # prediction, particle_list, particle_prob_list = method.predict(session, batch_input, return_particles=True)
                    # np.savez('./plot_results_traj_9', prediction, particle_list, particle_prob_list)
                    npzfile = np.load('plot_results_traj_9.npz')
                    prediction = npzfile['arr_0']
                    particle_list = npzfile['arr_1']
                    particle_prob_list = npzfile['arr_2']
                    error_x = batch['s'][0, -1, 0] - prediction[0, -1, 0]
                    error_y = batch['s'][0, -1, 1] - prediction[0, -1, 1]
                    error_trans = np.sqrt(error_x ** 2 + error_y ** 2) / dist
                    error_rot = abs(wrap_angle(batch['s'][0, -1, 2] - prediction[0, -1, 2]))/dist * 180 / np.pi

                    errors[test_traj][seq_len]['trans'].append(error_trans)
                    errors[test_traj][seq_len]['rot'].append(error_rot)

                    if plot_results:

                        dim_names = ['pos']
                        fig1 = plt.figure(figsize=[3,3])
                        fig2 = plt.figure(figsize=[3,3])
                        grid = plt.GridSpec(3, 6)
                        ax1 = fig2.add_subplot(111)
                        ax2 = fig1.add_subplot(grid[0, :3])
                        ax3 = fig1.add_subplot(grid[0, 3:6])
                        ax4 = fig1.add_subplot(grid[1, :3])
                        ax5 = fig1.add_subplot(grid[1, 3:6])
                        ax6 = fig1.add_subplot(grid[2, :3])
                        ax7 = fig1.add_subplot(grid[2, 3:6])
                        # ax4 = fig.add_subplot(224)
                        # ax6 = fig.add_subplot(326)
                        # for t in range(particle_list.shape[1]):
                        dim = 0
                            # ax1.scatter(particle_list[0, t, :, dim], particle_list[0, t, :, dim+1], c=particle_prob_list[0, t, :], cmap='viridis_r', marker='o', s=1, alpha=0.1,
                            #                     linewidths=0.05,
                            #                     vmin=0.0,
                            #                     vmax=0.02)

                        ax1.plot(prediction[0, :, dim], prediction[0, :, dim+1], 'b')

                        ax1.plot(batch['s'][0, :, dim], batch['s'][0, :, dim+1], 'r')
                        ax1.plot(batch['s'][0, 100:350:100, dim], batch['s'][0, 100:350:100, dim+1], 'ok', markersize=3, markerfacecolor='None')
                        ax1.plot(batch['s'][0, 0, dim], batch['s'][0, 0, dim+1], 'xk', markersize=5, markerfacecolor='None')

                        ax1.set_aspect('equal')
                        ax1.set_ylim([-450, 320])
                            # ax2.scatter(particle_list[0, t, :, dim], particle_list[0, t, :, dim+1], c=particle_prob_list[0, t, :], cmap='viridis_r', marker='o', s=1, alpha=0.1,
                            #                     linewidths=0.05,
                            #                     vmin=0.0,
                            #                     vmax=0.02)
                            #
                            # ax1.plot([prediction[0, t, dim]], [prediction[0, t, dim+1]], 'o', markerfacecolor='None', markeredgecolor='b',
                            #                  markersize=0.5)
                            #
                            # ax1.plot([batch['s'][0, t, dim]], [batch['s'][0, t, dim+1]], '+', markerfacecolor='None', markeredgecolor='r',
                            #                  markersize=0.5)

                        ax2.imshow(np.clip(batch['o'][0, 100, :, :, 0:3]/255.0, 0.0, 1.0), interpolation='nearest')
                        ax3.imshow(np.clip(batch['o'][0, 100, :, :, 3:6]/255.0 + 0.5, 0.0, 1.0), interpolation='nearest')
                        ax4.imshow(np.clip(batch['o'][0, 200, :, :, 0:3]/255.0, 0.0, 1.0), interpolation='nearest')
                        ax5.imshow(np.clip(batch['o'][0, 200, :, :, 3:6]/255.0 + 0.5, 0.0, 1.0), interpolation='nearest')
                        ax6.imshow(np.clip(batch['o'][0, 300, :, :, 0:3]/255.0, 0.0, 1.0), interpolation='nearest')
                        ax7.imshow(np.clip(batch['o'][0, 300, :, :, 3:6]/255.0 + 0.5, 0.0, 1.0), interpolation='nearest')
                        ax2.set_axis_off()
                        ax3.set_axis_off()
                        ax4.set_axis_off()
                        ax5.set_axis_off()
                        ax6.set_axis_off()
                        ax7.set_axis_off()
                        # ax2.set_axis_off()
                        ax1.set_xlabel('x (m)')
                        ax1.set_ylabel('y (m)')
                        ax1.legend(['Predicted pose','Ground truth'])
                        # ax1.set_title(dim_names[0])
                        # ax2.set_title(dim_names[1])
                        # ax3.set_title(dim_names[2])
                        # ax4.set_title(dim_names[3])
                        fig1.savefig('{}.pdf'.format('test'), bbox_inches='tight')
                        fig2.savefig('{}.pdf'.format('test2'), bbox_inches='tight')
                        # plt.savefig('../plots/800_{}'.format(start_step))

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

if __name__ == '__main__':
    plt.ion()

    # errors = dict()
    # average_errors = {'trans': {i: [] for i in [100, 200, 400, 800]},
    #                   'rot': {i: [] for i in [100, 200, 400, 800]}}
    # model_path = '../models/tmp/Cross_validation_plot/'
    # for model, traj in find_all_cross_val_models(model_path):
    #     print('!!! Evaluatng model {} on trajectory {}'.format(model, traj))
    new_errors = get_evaluation_stats()
        # errors.update(new_errors)
        # print('')
        # print('Trajectory {}'.format(traj))
        # for seq_len in sorted(errors[traj].keys()):
        #     for measure in ['trans', 'rot']:
        #         e = errors[traj][seq_len][measure]
        #         mean_error = np.mean(e)
        #         se_error = np.std(e, ddof=1) / np.sqrt(len(e))
        #         average_errors[measure][seq_len].append(mean_error)
        #         print('{:>5} error for seq_len {}: {:.4f}+-{:.4f}'.format(measure, seq_len, mean_error, se_error))
        #
        # print('Averaged errors:')
        # for measure in ['trans', 'rot']:
        #     mean_error_over_all_subsequences = []
        #     for seq_len in sorted(average_errors[measure].keys()):
        #         e = np.array(average_errors[measure][seq_len])
        #         e = e[~np.isnan(e)]
        #         mean_error = np.mean(e)
        #         se_error = np.std(e, ddof=1) / np.sqrt(len(e))
        #         mean_error_over_all_subsequences.append(mean_error)
        #         print('{:>5} error for seq_len {}: {:.4f}+-{:.4f}'.format(measure, seq_len, mean_error, se_error))
        #     print('{:>5} mean error over all sequence_lengths: {:.4f}'.format(measure, np.mean(np.asarray(mean_error_over_all_subsequences))))
