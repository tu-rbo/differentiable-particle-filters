import tensorflow as tf
import pickle
from methods.dpf import DPF
from methods.rnn import RNN
from utils.data_utils import load_data, noisyfy_data, make_batch_iterator, reduce_data
from utils.exp_utils import get_default_hyperparams
from utils.method_utils import compute_sq_distance
from utils.plotting_utils import plot_maze, plot_observations
from methods.odom import OdometryBaseline
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable

head_scale = 1.5
quiv_kwargs = {'scale_units':'xy', 'scale':1./40., 'width': 0.003, 'headlength': 5*head_scale, 'headwidth': 3*head_scale, 'headaxislength': 4.5*head_scale}
marker_kwargs = {'markersize': 4.5, 'markerfacecolor':'None', 'markeredgewidth':0.5}

def plot_measurement_model(session, method, statistics, batch, task, num_examples, variant):

    batch_size = len(batch['o'])

    x = np.linspace(100.0 / 4, 1000.0 - 100.0 / 4, 20)
    y = np.linspace(100.0 / 4, 500.0 - 100.0 / 4, 10)
    theta = np.linspace(-np.pi, np.pi, 12 + 1)[1:]
    g = np.meshgrid(x, y, theta)

    poses = np.vstack([np.ravel(x) for x in g]).transpose([1, 0])
    test_poses = tf.tile(tf.constant(poses, dtype='float32')[None, :, :], [batch_size, 1, 1])
    measurement_model_out = method.measurement_update(method.encodings[0, :], test_poses, statistics['means'],
                                                      statistics['stds'])

    # define the inputs and train/run the model
    input_dict = {**{method.placeholders[key]: batch[key] for key in 'osa'},
                  }

    obs_likelihood = session.run(measurement_model_out, input_dict)
    print(obs_likelihood.shape)

    for i in range(num_examples):
        # plt.figure("%s likelihood" % i)
        fig, (ax, cax) = plt.subplots(1, 2, figsize=(2.4 / 0.83 / 0.95 / 0.97, 1.29 / 0.9),
                                      gridspec_kw={"width_ratios": [0.97, 0.03]}, num="%s %s likelihood" % (variant, i))
        # plt.gca().clear()
        plot_maze(task, margin=5, linewidth=0.5, ax=ax)

        idx = obs_likelihood[i,:] > 1*np.mean(obs_likelihood[i,:])
        # idx = obs_likelihood[i, :] > 0 * np.mean(obs_likelihood[i, :])
        max = np.max(obs_likelihood[i, :])

        # ax.scatter([poses[:, 0]], [poses[:, 1]], s=[0.001], c=[(0.8, 0.8, 0.8)], marker='.')

        quiv = ax.quiver(poses[idx, 0] + 0 * np.cos(poses[idx, 2]), poses[idx, 1] + 0* np.sin(poses[idx, 2]), np.cos(poses[idx, 2]),
                         np.sin(poses[idx, 2]), obs_likelihood[i, idx],
                         cmap='viridis_r',
                         clim=[0.0, max],
                         **quiv_kwargs
                         )

        ax.plot([batch['s'][0, i, 0]], [batch['s'][0, i, 1]], 'or', **marker_kwargs)

        ax.quiver([batch['s'][0, i, 0]], [batch['s'][0, i, 1]], np.cos([batch['s'][0, i, 2]]),
                  np.sin([batch['s'][0, i, 2]]), color='red',
                  **quiv_kwargs
                  )
        ax.axis('off')
        fig.colorbar(quiv, cax=cax, orientation="vertical", label='Obs. likelihood', ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        plt.subplots_adjust(left=0.0, bottom=0.05, right=0.83, top=0.95, wspace=0.05, hspace=0.00)
        plt.savefig('../plots/models/measurement_model{}.pdf'.format(i), transparent=True, dpi=600, frameon=False, facecolor='w', pad_inches=0.01)


def plot_proposer(session, method, statistics, batch, task, num_examples, variant):

    num_particles = 1000
    proposer_out = method.propose_particles(method.encodings[0, :], num_particles, statistics['state_mins'], statistics['state_maxs'])

    # define the inputs and train/run the model
    input_dict = {**{method.placeholders[key]: batch[key] for key in 'osa'},
                  }
    particles = session.run(proposer_out, input_dict)

    for i in range(num_examples):
        fig = plt.figure(figsize=(2.4, 1.29/0.9), num="%s %s proposer" % (variant, i))
        # plt.gca().clear()
        plot_maze(task, margin=5, linewidth=0.5)

        quiv = plt.quiver(particles[i, :, 0], particles[i, :, 1], np.cos(particles[i, :, 2]),
                         np.sin(particles[i, :, 2]), np.ones([num_particles]), cmap='viridis_r', clim=[0, 2], alpha=1.0,
                          **quiv_kwargs
                          )

        plt.quiver([batch['s'][0, i, 0]], [batch['s'][0, i, 1]], np.cos([batch['s'][0,i, 2]]),
                  np.sin([batch['s'][0, i, 2]]), color='red',
                **quiv_kwargs)  # width=0.01, scale=100
        plt.plot([batch['s'][0, i, 0]], [batch['s'][0, i, 1]], 'or', **marker_kwargs)


        plt.gca().axis('off')
        plt.subplots_adjust(left=0.0, bottom=0.05, right=1.0, top=0.95, wspace=0.0, hspace=0.00)

        # plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.001, hspace=0.1)
        plt.savefig('../plots/models/prop{}.pdf'.format(i), transparent=True, dpi=600, frameon=False, facecolor='w', pad_inches=0.01)


def plot_motion_model(session, method, statistics, batch, task, num_examples, num_particles, variant):

    motion_samples = method.motion_update(method.placeholders['a'][:, 1],
                                        tf.tile(method.placeholders['s'][:, :1], [1, num_particles, 1]),
                                        statistics['means'], statistics['stds'], statistics['state_step_sizes'])

    # define the inputs and train/run the model
    input_dict = {**{method.placeholders[key]: batch[key] for key in 'osa'},
                  }
    particles = session.run(motion_samples, input_dict)

    fig = plt.figure(figsize=(2.4, 1.29), num="%s motion model" % (variant))
    # plt.gca().clear()
    plot_maze(task, margin=5, linewidth=0.5)

    for i in range(num_examples):

        plt.quiver(particles[i, :, 0], particles[i, :, 1], np.cos(particles[i, :, 2]),
                          np.sin(particles[i, :, 2]), np.ones([num_particles]), cmap='viridis_r',
                   **quiv_kwargs,
                   alpha=1.0, clim=[0, 2])  # width=0.01, scale=100

        plt.quiver([batch['s'][i, 0, 0]], [batch['s'][i, 0, 1]], np.cos([batch['s'][i, 0, 2]]),
                   np.sin([batch['s'][i, 0, 2]]), color='black',
                   **quiv_kwargs,
                   )  # width=0.01, scale=100

        plt.plot(batch['s'][i, :2, 0], batch['s'][i, :2, 1], '--', color='black', linewidth=0.3)
        plt.plot(batch['s'][i, :1, 0], batch['s'][i, :1, 1], 'o', color='black', linewidth=0.3, **marker_kwargs)
        plt.plot(batch['s'][i, 1:2, 0], batch['s'][i, 1:2, 1], 'o', color='red', linewidth=0.3, **marker_kwargs)

        plt.quiver([batch['s'][i, 1, 0]], [batch['s'][i, 1, 1]], np.cos([batch['s'][i, 1, 2]]),
                   np.sin([batch['s'][i, 1, 2]]), color='red',
                   **quiv_kwargs)  # width=0.01, scale=100

    plt.gca().axis('off')

    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.001, hspace=0.1)
    plt.savefig('../plots/models/motion_model{}.pdf'.format(i), transparent=True, dpi=600, frameon=False, facecolor='w', pad_inches=0.01)


def plot_particle_filter(session, method, statistics, batch, task, num_examples, num_particles, variant):
    color_list = plt.cm.tab10(np.linspace(0, 1, 10))
    colors = {'lstm': color_list[0], 'pf_e2e': color_list[1], 'pf_ind_e2e': color_list[2], 'pf_ind': color_list[3],
              'ff': color_list[4], 'odom': color_list[4]}

    pred, s_particle_list, s_particle_probs_list = method.predict(session, batch, num_particles, return_particles=True)

    num_steps = 20 # s_particle_list.shape[1]

    for s in range(num_examples):

        plt.figure("example {}, vartiant: {}".format(s, variant), figsize=[12, 5.15])
        plt.gca().clear()

        for i in range(num_steps):
            ax = plt.subplot(4, 5, i + 1, frameon=False)
            plt.gca().clear()

            plot_maze(task, margin=5, linewidth=0.5)

            if i < num_steps - 1:
                ax.quiver(s_particle_list[s, i, :, 0], s_particle_list[s, i, :, 1],
                           np.cos(s_particle_list[s, i, :, 2]), np.sin(s_particle_list[s, i, :, 2]),
                           s_particle_probs_list[s, i, :], cmap='viridis_r', clim=[.0, 2.0/num_particles], alpha=1.0,
                          **quiv_kwargs
                          )

                current_state = batch['s'][s, i, :]
                plt.quiver(current_state[0], current_state[1], np.cos(current_state[2]),
                           np.sin(current_state[2]), color="red", **quiv_kwargs)

                plt.plot(current_state[0], current_state[1], 'or', **marker_kwargs)
            else:

                ax.plot(batch['s'][s, :num_steps, 0], batch['s'][s, :num_steps, 1], '-', linewidth=0.6, color='red')
                ax.plot(pred[s, :num_steps, 0], pred[s, :num_steps, 1], '-', linewidth=0.6,
                        color=colors['pf_ind_e2e'])

                ax.plot(batch['s'][s, :1, 0], batch['s'][s, :1, 1], '.', linewidth=0.6, color='red', markersize=3)
                ax.plot(pred[s, :1, 0], pred[s, :1, 1], '.', linewidth=0.6, markersize=3,
                        color=colors['pf_ind_e2e'])


            plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.001, hspace=0.1)
            plt.gca().set_aspect('equal')
            plt.xticks([])
            plt.yticks([])

        plt.savefig('../plots/models/pf{}.pdf'.format(s), transparent=True, dpi=600, frameon=False, facecolor='w', pad_inches=0.01)

        plt.figure('colorbar', (12, 0.6))
        a = np.array([[0, 2.0/num_particles]])
        img = plt.imshow(a, cmap="viridis_r")
        plt.gca().set_visible(False)
        cax = plt.axes([0.25, 0.75, 0.50, 0.2])
        plt.colorbar(orientation="horizontal", cax=cax, label='Particle weight', ticks=[0, 0.001, 0.002])

        plt.savefig('../plots/models/colorbar.pdf'.format(s), transparent=True, dpi=600, frameon=False, facecolor='w', pad_inches=0.01)

def plot_prediction(pred1, pred2, statistics, batch, task, num_examples, variant):
    color_list = plt.cm.tab10(np.linspace(0, 1, 10))
    colors = {'lstm': color_list[0], 'pf_e2e': color_list[1], 'pf_ind_e2e': color_list[2], 'pf_ind': color_list[3],
                'ff': color_list[4], 'odom': color_list[4]}

    num_steps = 50
    init_steps = 20

    for s in range(num_examples):

        fig = plt.figure(figsize=(2.4, 1.29), num="%s prediction %s" % (variant, s))

        # plt.figure("example {}, vartiant: {}".format(s, variant), figsize=[12, 5.15])
        plt.gca().clear()
        plot_maze(task, margin=5, linewidth=0.5)

        plt.plot(batch['s'][s, :num_steps, 0], batch['s'][s, :num_steps, 1], '-', linewidth=0.3, color='gray')
        plt.plot(pred1[s, :init_steps, 0], pred1[s, :init_steps, 1], '--', linewidth=0.3, color=colors['pf_ind_e2e'])
        plt.plot(pred1[s, init_steps-1:num_steps, 0], pred1[s, init_steps-1:num_steps, 1], '-', linewidth=0.3, color=colors['pf_ind_e2e'])
        plt.plot(pred2[s, :init_steps, 0], pred2[s, :init_steps, 1], '--', linewidth=0.3, color=colors['lstm'])
        plt.plot(pred2[s, init_steps-1:num_steps, 0], pred2[s, init_steps-1:num_steps, 1], '-', color=colors['lstm'], linewidth=0.3)

        # for i in range(init_steps, num_steps):
        #
        #     p = pred1[s, i, :]
        #     plt.quiver(p[0], p[1], np.cos(p[2]),
        #                np.sin(p[2]), color=colors['pf_ind_e2e'], **quiv_kwargs)
        #     p = pred2[s, i, :]
        #     plt.quiver(p[0], p[1], np.cos(p[2]),
        #                np.sin(p[2]), color=colors['lstm'], **quiv_kwargs)
        #     # plt.plot(p[0], p[1], 'og', **marker_kwargs)
        #
        #     current_state = batch['s'][s, i, :]
        #     plt.quiver(current_state[0], current_state[1], np.cos(current_state[2]),
        #                np.sin(current_state[2]), color="black", **quiv_kwargs)
        #     # plt.plot(current_state[0], current_state[1], 'or', **marker_kwargs)

        plt.gca().set_aspect('equal')
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.001, hspace=0.1)
        plt.savefig('../plots/models/pred{}.pdf'.format(s), transparent=True, dpi=600, frameon=False, facecolor='w', pad_inches=0.01)


def plot_observation(batch, i, t=0):

    plt.figure('%r obs' % i, (2, 2))
    plt.imshow(np.clip(batch['o'][i, t, :, :, :] / 255.0, 0.0, 1.0), interpolation='nearest')
    plt.axis('off')
    # plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.001, hspace=0.001)
    plt.subplots_adjust(left=0.0, bottom=0.15, right=1.0, top=0.85, wspace=0.0, hspace=0.00)
    plt.savefig('../plots/models/obs{}.png'.format(i+t), transparent=True, dpi=600, frameon=False, facecolor='w', pad_inches=0.01)


def plot_measurement_statistics(session, method, statistics, batch_iterator, batch_size, variant):

    color_list = plt.cm.tab10(np.linspace(0, 1, 10))
    colors = {'lstm': color_list[0], 'e2e': color_list[1], 'ind_e2e': color_list[2], 'ind': color_list[3], 'ff': color_list[4], 'odom': color_list[4]}
    labels = {'e2e': 'DPF(e2e)', 'ind_e2e': 'DPF(ind+e2e)', 'ind': 'DPF(ind)'}

    x = np.linspace(100.0 / 4, 1500.0 - 100.0 / 4, 30)
    y = np.linspace(100.0 / 4, 900.0 - 100.0 / 4, 18)
    theta = np.linspace(-np.pi, np.pi, 12 + 1)[1:]
    g = np.meshgrid(x, y, theta)

    poses = np.vstack([np.ravel(x) for x in g]).transpose([1, 0])
    test_poses = tf.tile(tf.constant(poses, dtype='float32')[None, :, :], [batch_size, 1, 1])
    measurement_model_out = method.measurement_update(method.encodings[:, 0], test_poses, statistics['means'],
                                                      statistics['stds'])
    true_measurement_model_out = method.measurement_update(method.encodings[:, 0], method.placeholders['s'][:, 0, None, :], statistics['means'],
                                                      statistics['stds'])

    hist = 0.0
    true_hist = 0.0

    for i in range(1000000): # 1000000
        # define the inputs and train/run the model
        batch = next(batch_iterator)
        input_dict = {**{method.placeholders[key]: batch[key] for key in 'osa'},
                      }
        if i < 100:
            obs_likelihood, true_obs_likelihood = session.run([measurement_model_out, true_measurement_model_out], input_dict)
            h, bins = np.histogram(obs_likelihood, 50, [0,1])
            hist += h
        else:
            true_obs_likelihood = session.run(true_measurement_model_out, input_dict)
        h, true_bins = np.histogram(true_obs_likelihood, 20, [0,1])
        true_hist += h

    true_hist = true_hist / np.sum(true_hist) * len(true_hist)
    hist = hist / np.sum(hist) * len(hist)
    plt.figure('Observation likelihood statistics', [3.3,2.5])
    plt.plot(bins[1:] - (bins[1]-bins[0])/2, hist, '--', color=colors[variant])
    plt.plot(true_bins[1:] - (true_bins[1]-true_bins[0])/2, true_hist, '-', color=colors[variant], label=labels[variant])
    plt.legend(loc='upper center')
    plt.yticks([0, 1, 2, 3])
    plt.ylim([0,3])
    plt.xlabel('Estimated observation likelihood')
    plt.ylabel('Density')
    # plt.gca().set_yscale("log", nonposx='clip')
    plt.tight_layout()
    plt.savefig('../plots/models/measurement_statistics.pdf', transparent=True, dpi=600, frameon=False, facecolor='w', pad_inches=0.01)



def plot_motion_statistics(session, method, statistics, batch_iterator, task, variant):

    color_list = plt.cm.tab10(np.linspace(0, 1, 10))
    colors = {'lstm': color_list[0], 'e2e': color_list[1], 'ind_e2e': color_list[2], 'ind': color_list[3], 'ff': color_list[4], 'odom': color_list[4]}
    labels = {'e2e': 'DPF(e2e)', 'ind_e2e': 'DPF(ind+e2e)', 'ind': 'DPF(ind)'}

    num_particles = 100

    motion_samples = method.motion_update(method.placeholders['a'][:, 1],
                                        tf.tile(method.placeholders['s'][:, :1], [1, num_particles, 1]),
                                        statistics['means'], statistics['stds'], statistics['state_step_sizes'])

    odom = OdometryBaseline()
    error_hist = 0.0
    odom_error_hist = 0.0
    for i in range(10000): # 100000
        # define the inputs and train/run the model
        batch = next(batch_iterator)
        # define the inputs and train/run the model
        input_dict = {**{method.placeholders[key]: batch[key] for key in 'osa'},
                      }

        # action_size = compute_sq_distance(batch['s'][:, 0, :], batch['s'][:, 1, :], state_step_sizes=statistics['state_step_sizes']) ** 0.5
        action_size = abs(batch['s'][:, 0, 0] - batch['s'][:, 1, 0]) / statistics['state_step_sizes'][0]
        action_size /= action_size

        odom_pred = odom.predict(None, batch)
        # odom_errors = compute_sq_distance(odom_pred[:, 1, :], batch['s'][:, 1, :], state_step_sizes=statistics['state_step_sizes']) ** 0.5
        odom_errors = (odom_pred[:, 1, 0] - batch['s'][:, 1, 0]) / statistics['state_step_sizes'][0]
        # odom_error_hist += np.histogram(odom_errors / action_size, 100, range=[0, 2])[0]
        odom_error_hist += np.histogram(odom_errors / action_size, 101, range=[-1, 1])[0]

        if i < 10000:
            particles = session.run(motion_samples, input_dict)
            # errors = compute_sq_distance(particles, batch['s'][:, 1, None, :], state_step_sizes=statistics['state_step_sizes']) ** 0.5
            errors = (particles[:, :, 0] - odom_pred[:, 1, None, 0]) / statistics['state_step_sizes'][0]
            # h, bins = np.histogram(errors / action_size[:, None], 100, range=[0, 2])
            h, bins = np.histogram(errors / action_size[:, None], 101, range=[-1, 1])
            error_hist += h
        elif variant != 'e2e':
            break

    error_hist = error_hist / np.sum(error_hist) * 50.5
    odom_error_hist = odom_error_hist / np.sum(odom_error_hist) * 50.5

    plt.figure('motion statistics', [2.5,2.5])
    plt.plot(bins[1:] - (bins[1]-bins[0])/2, error_hist, color=colors[variant], label=labels[variant])
    if variant == 'e2e':
        plt.plot(bins[1:] - (bins[1]-bins[0])/2, odom_error_hist, ':', color='k', label='True noise')
    plt.xlabel('Predicted pos. relative to odom.')
    plt.ylabel('Density')
    # plt.legend()
    plt.tight_layout()
    plt.savefig('../plots/models/motion_statistics.pdf', transparent=True, dpi=600, frameon=False, facecolor='w', pad_inches=0.01)



def plot_models():
    task = 'nav01'
    data_path = '../data/100s'
    test_data = load_data(data_path=data_path, filename=task + '_test')
    noisy_test_data = noisyfy_data(reduce_data(test_data, 10))
    num_examples = 10
    # same seqlen and batchsize needed here!
    # test_batch_iterator = make_batch_iterator(noisy_test_data, seq_len=50, batch_size=50)
    test_batch_iterator = make_batch_iterator(noisy_test_data, seq_len=50, batch_size=num_examples)
    batch = next(test_batch_iterator)

    # for i in range(num_examples):
    #     plot_observation(batch, i=0, t=i)

    predictions = dict()

    for variant, file_name in {
                               'ind_e2e': '2017-12-23_03:32:47_compute-0-9_nav01_pf_ind_e2e_1000',
                               # 'ind_e2e': '2017-12-22_18:30:30_compute-0-1_nav02_pf_ind_e2e_1000',
                               # 'lstm': '2017-12-24_13:25:53_compute-0-1_nav01_lstm_1000',
                               # 'lstm': '2017-12-22_18:29:21_compute-1-2_nav02_lstm_1000',
                               # 'ind': '2017-12-23_00:48:08_compute-0-74_nav01_pf_ind_500',
                               # 'e2e': '2017-12-22_18:29:49_compute-0-15_nav01_pf_e2e_500',
                               }.items():

        with open('../log/lc/'+file_name, 'rb') as f:
            log = pickle.load(f)
        hyper_params = log['hyper_params'][0]
        model_path = '../models/' + log['exp_params'][0]['model_path'].split('/models/')[-1] # ['exp_params']['model_path]

        # reset tensorflow graph
        tf.reset_default_graph()

        # instantiate method
        if 'lstm' in variant:
            method = RNN(**hyper_params['global'])
        else:
            method = DPF(**hyper_params['global'])

        with tf.Session() as session:
            # load method and apply to new data
            statistics = method.load(session, model_path)
            # print('predicting now')
            # predictions[variant] = method.predict(session, batch, num_particles=1000, return_particles=False)
            # print('prediction done')
            # plot_measurement_model(session, method, statistics, batch, task, num_examples, variant)
            # plot_proposer(session, method, statistics, batch, task, num_examples, variant)
            # plot_motion_model(session, method, statistics, batch, task, 10, 50, variant)
            plot_particle_filter(session, method, statistics, batch, task, num_examples, 1000, variant)

    print(predictions.keys())
    # plot_prediction(predictions['ind_e2e'], predictions['lstm'], statistics, batch, task, num_examples, variant)

    plt.pause(10000.0)

def plot_statistics():
    task = 'nav02'
    data_path = '../data/100s'
    test_data = load_data(data_path=data_path, filename=task + '_test')
    noisy_test_data = noisyfy_data(test_data)
    # noisy_test_data = noisyfy_data(test_data)
    batch_size = 32
    test_batch_iterator = make_batch_iterator(noisy_test_data, seq_len=2, batch_size=batch_size)


    filenames = {              'ind_e2e': '2017-12-22_18:30:30_compute-0-1_nav02_pf_ind_e2e_1000',
                               'ind': '2017-12-23_06:56:07_compute-0-26_nav02_pf_ind_1000',
                               'e2e': '2017-12-24_00:51:18_compute-1-0_nav02_pf_e2e_1000',
                               }

    for variant in ['ind', 'e2e', 'ind_e2e']:
        file_name = filenames[variant]

        with open('../log/lc/'+file_name, 'rb') as f:
            log = pickle.load(f)
        hyper_params = log['hyper_params'][0]
        model_path = '../models/' + log['exp_params'][0]['model_path'].split('/models/')[-1] # ['exp_params']['model_path]

        # reset tensorflow graph
        tf.reset_default_graph()

        # instantiate method
        method = DPF(**hyper_params['global'])

        with tf.Session() as session:
            # load method and apply to new data
            statistics = method.load(session, model_path)
            plot_measurement_statistics(session, method, statistics, test_batch_iterator, batch_size, variant)
            plot_motion_statistics(session, method, statistics, test_batch_iterator, task, variant)

    plt.pause(10000.0)

plot_models()
# plot_statistics()
