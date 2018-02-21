import os
import numpy as np
import sonnet as snt
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.data_utils_kitti import wrap_angle, compute_statistics, split_data, make_batch_iterator, make_repeating_batch_iterator, rotation_matrix, load_data_for_stats
from utils.method_utils import atan2, compute_sq_distance
from utils.plotting_utils import plot_maze, show_pause
from datetime import datetime

if tf.__version__ == '1.1.0-rc1' or tf.__version__ == '1.2.0':
    from tensorflow.python.framework import ops
    @ops.RegisterGradient("FloorMod")
    def _mod_grad(op, grad):
        x, y = op.inputs
        gz = grad
        x_grad = gz
        y_grad = None  # tf.reduce_mean(-(x // y) * gz, axis=[0], keep_dims=True)[0]
        return x_grad, y_grad


class DPF():

    def __init__(self, init_with_true_state, learn_odom, use_proposer, propose_ratio, proposer_keep_ratio, min_obs_likelihood, learn_gaussian_mle):
        """
        :param init_with_true_state:
        :param learn_odom:
        :param use_proposer:
        :param propose_ratio:
        :param particle_std:
        :param proposer_keep_ratio:
        :param min_obs_likelihood:
        """

        # store hyperparameters which are needed later
        self.init_with_true_state = init_with_true_state
        self.learn_odom = learn_odom
        self.use_proposer = use_proposer and not init_with_true_state  # only use proposer if we do not initializet with true state
        self.propose_ratio = propose_ratio if not self.init_with_true_state else 0.0

        # define some more parameters and placeholders
        self.state_dim = 5
        self.action_dim = 3
        self.observation_dim = 6
        self.placeholders = {'o': tf.placeholder('float32', [None, None, 50, 150, self.observation_dim], 'observations'),
                             'a': tf.placeholder('float32', [None, None, 3], 'actions'),
                             's': tf.placeholder('float32', [None, None, 5], 'states'),
                             'num_particles': tf.placeholder('float32'),
                             'keep_prob': tf.placeholder_with_default(tf.constant(1.0), []),
                             'is_training': tf.placeholder_with_default(tf.constant(False), [])
                             }
        self.num_particles_float = self.placeholders['num_particles']
        self.num_particles = tf.to_int32(self.num_particles_float)

        # build learnable modules
        self.build_modules(min_obs_likelihood, proposer_keep_ratio, learn_gaussian_mle)


    def build_modules(self, min_obs_likelihood, proposer_keep_ratio, learn_gaussian_mle):
        """
        :param min_obs_likelihood:
        :param proposer_keep_ratio:
        :return: None
        """

        # MEASUREMENT MODEL

        # conv net for encoding the image
        self.encoder = snt.Sequential([
            snt.nets.ConvNet2D([16, 16, 16, 16], [[7, 7], [5, 5], [5, 5], [5, 5]], [[1,1], [1, 2], [1, 2], [2, 2]], [snt.SAME], activate_final=True, name='encoder/convnet'),
            snt.BatchFlatten(),
            lambda x: tf.nn.dropout(x,  self.placeholders['keep_prob']),
            snt.Linear(128, name='encoder/linear'),
            tf.nn.relu
        ])

        # observation likelihood estimator that maps states and image encodings to probabilities
        self.obs_like_estimator = snt.Sequential([
            snt.Linear(128, name='obs_like_estimator/linear'),
            tf.nn.relu,
            snt.Linear(128, name='obs_like_estimator/linear'),
            tf.nn.relu,
            snt.Linear(1, name='obs_like_estimator/linear'),
            tf.nn.sigmoid,
            lambda x: x * (1 - min_obs_likelihood) + min_obs_likelihood
        ], name='obs_like_estimator')

        # motion noise generator used for motion sampling
        if learn_gaussian_mle:
            self.mo_noise_generator = snt.nets.MLP([32, 32, 4], activate_final=False, name='mo_noise_generator')
        else:
            self.mo_noise_generator = snt.nets.MLP([32, 32, 2], activate_final=False, name='mo_noise_generator')

        # odometry model (if we want to learn it)
        if self.learn_odom:
            self.mo_transition_model = snt.nets.MLP([128, 128, 128, self.state_dim], activate_final=False, name='mo_transition_model')

        # particle proposer that maps encodings to particles (if we want to use it)
        if self.use_proposer:
            self.particle_proposer = snt.Sequential([
                snt.Linear(128, name='particle_proposer/linear'),
                tf.nn.relu,
                lambda x: tf.nn.dropout(x,  proposer_keep_ratio),
                snt.Linear(128, name='particle_proposer/linear'),
                tf.nn.relu,
                snt.Linear(128, name='particle_proposer/linear'),
                tf.nn.relu,
                snt.Linear(128, name='particle_proposer/linear'),
                tf.nn.relu,
                snt.Linear(4, name='particle_proposer/linear'),
                tf.nn.tanh,
            ])

        self.noise_scaler1 = snt.Module(lambda x: x * tf.exp(10 * tf.get_variable('motion_sampler/noise_scaler1', initializer=np.array(0.0, dtype='float32'))))
        self.noise_scaler2 = snt.Module(lambda x: x * tf.exp(10 * tf.get_variable('motion_sampler/noise_scaler2', initializer=np.array(0.0, dtype='float32'))))


    def custom_build(self, inputs):
        """A custom build method to wrap into a sonnet Module."""
        outputs = snt.Conv2D(output_channels=16, kernel_shape=[7, 7], stride=[1, 1])(inputs)
        outputs = tf.nn.relu(outputs)
        outputs = snt.Conv2D(output_channels=16, kernel_shape=[5, 5], stride=[1, 2])(outputs)
        outputs = tf.nn.relu(outputs)
        outputs = snt.Conv2D(output_channels=16, kernel_shape=[5, 5], stride=[1, 2])(outputs)
        outputs = tf.nn.relu(outputs)
        outputs = snt.Conv2D(output_channels=16, kernel_shape=[5, 5], stride=[2, 2])(outputs)
        outputs = tf.nn.relu(outputs)
        outputs = tf.nn.dropout(outputs,  self.placeholders['keep_prob'])
        outputs = snt.BatchFlatten()(outputs)
        outputs = snt.Linear(128)(outputs)
        outputs = tf.nn.relu(outputs)

        return outputs

    def measurement_update(self, encoding, particles, means, stds):
        """
        Compute the likelihood of the encoded observation for each particle.

        :param encoding: encoding of the observation
        :param particles:
        :param means:
        :param stds:
        :return: observation likelihood
        """

        # prepare input (normalize particles poses and repeat encoding per particle)
        particle_input = self.transform_particles_as_input(particles, means, stds)
        encoding_input = tf.tile(encoding[:, tf.newaxis, :], [1,  tf.shape(particles)[1], 1])
        input = tf.concat([encoding_input, particle_input], axis=-1)

        # estimate the likelihood of the encoded observation for each particle, remove last dimension
        obs_likelihood = snt.BatchApply(self.obs_like_estimator)(input)[:, :, 0]

        return obs_likelihood


    def transform_particles_as_input(self, particles, means, stds):
        return ((particles - means['s']) / stds['s'])[..., 3:5]


    def propose_particles(self, encoding, num_particles, state_mins, state_maxs):
        duplicated_encoding = tf.tile(encoding[:, tf.newaxis, :], [1, num_particles, 1])
        proposed_particles = snt.BatchApply(self.particle_proposer)(duplicated_encoding)
        proposed_particles = tf.concat([
            proposed_particles[:,:,:1] * (state_maxs[0] - state_mins[0]) / 2.0 + (state_maxs[0] + state_mins[0]) / 2.0,
            proposed_particles[:,:,1:2] * (state_maxs[1] - state_mins[1]) / 2.0 + (state_maxs[1] + state_mins[1]) / 2.0,
            atan2(proposed_particles[:,:,2:3], proposed_particles[:,:,3:4])], axis=2)
        return proposed_particles


    def motion_update(self, actions, particles, means, stds, state_step_sizes, learn_gaussian_mle, stop_sampling_gradient=False):
        """
        Move particles according to odometry info in actions. Add learned noise.

        :param actions:
        :param particles:
        :param means:
        :param stds:
        :param state_step_sizes:
        :param stop_sampling_gradient:
        :return: moved particles
        """

        # 1. SAMPLE NOISY ACTIONS

        # add dimension for particles
        time_step = 0.103

        if learn_gaussian_mle:
            actions = tf.concat([particles[:, :, 3:4] - means['s'][:, :, 3:4], particles[:, :, 4:5] - means['s'][:, :, 4:5]], axis=-1)

            # prepare input (normalize actions and repeat per particle)
            action_input = actions / stds['s'][:, :, 3:5]
            input = action_input

            # estimate action noise
            delta = snt.BatchApply(self.mo_noise_generator)(input)
            delta = tf.concat([delta[:, :, 0:2] * state_step_sizes[3], delta[:, :, 2:4] * state_step_sizes[4]], axis=-1)
            if stop_sampling_gradient:
                delta = tf.stop_gradient(delta)

            action_vel_f = tf.random_normal(tf.shape(particles[:, :, 3:4]), mean = delta[:, :, 0:1], stddev = delta[:, :, 1:2])
            action_vel_rot = tf.random_normal(tf.shape(particles[:, :, 4:5]), mean = delta[:, :, 2:3], stddev = delta[:, :, 3:4])

            heading = particles[:, :, 2:3]
            sin_heading = tf.sin(heading)
            cos_heading = tf.cos(heading)

            new_x = particles[:, :, 0:1] + cos_heading * particles[:, :, 3:4] * time_step
            new_y = particles[:, :, 1:2] + sin_heading * particles[:, :, 3:4] * time_step
            new_theta = particles[:, :, 2:3] + particles[:, :, 4:5] * time_step
            wrap_angle(new_theta)
            new_v = particles[:, :, 3:4] + action_vel_f
            new_theta_dot = particles[:, :, 4:5] + action_vel_rot

            moved_particles = tf.concat([new_x, new_y, new_theta, new_v, new_theta_dot], axis=-1)

            return moved_particles, delta

        else:

            heading = particles[:, :, 2:3]
            sin_heading = tf.sin(heading)
            cos_heading = tf.cos(heading)

            random_input = tf.random_normal(tf.shape(particles[:, :, 3:5]))
            noise = snt.BatchApply(self.mo_noise_generator)(random_input)
            noise = noise - tf.reduce_mean(noise, axis=1, keep_dims=True)

            new_z = particles[:, :, 0:1] + cos_heading * particles[:, :, 3:4] * time_step
            new_x = particles[:, :, 1:2] + sin_heading * particles[:, :, 3:4] * time_step
            new_theta = wrap_angle(particles[:, :, 2:3] + particles[:, :, 4:5] * time_step)

            new_v = particles[:, :, 3:4] + noise[:, :, :1] * state_step_sizes[3]
            new_theta_dot = particles[:, :, 4:5] + noise[:, :, 1:] * state_step_sizes[4]

            moved_particles = tf.concat([new_z, new_x, new_theta, new_v, new_theta_dot], axis=-1)

            return moved_particles


    def compile_training_stages(self, sess, batch_iterators, particle_list, particle_probs_list, encodings, means, stds, state_step_sizes, state_mins, state_maxs, learn_gaussian_mle, learning_rate, plot_task):

        # TRAINING!
        losses = dict()
        train_stages = dict()
        std = 0.25

        # TRAIN ODOMETRY

        if self.learn_odom:

            # apply model
            motion_samples = self.motion_update(self.placeholders['a'][:,0],
                                                self.placeholders['s'][:, :1],
                                                means, stds, state_step_sizes,
                                                stop_sampling_gradient=True)

            # define loss and optimizer
            sq_distance = compute_sq_distance(motion_samples, self.placeholders['s'][:, 1:2], state_step_sizes)
            losses['motion_mse'] = tf.reduce_mean(sq_distance, name='loss')
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            # put everything together
            train_stages['train_odom'] = {
                         'train_op': optimizer.minimize(losses['motion_mse']),
                         'batch_iterator_names': {'train': 'train1', 'val': 'val1'},
                         'monitor_losses': ['motion_mse'],
                         'validation_loss': 'motion_mse',
                         'plot': lambda e: self.plot_motion_model(sess, next(batch_iterators['val2']), motion_samples, plot_task, state_step_sizes) if e % 1 == 0 else None
                         }

        # TRAIN MOTION MODEL

        if learn_gaussian_mle:
            motion_samples, motion_params = self.motion_update(self.placeholders['a'][:,1],
                                                tf.tile(self.placeholders['s'][:, :1], [1, 1, 1]),
                                                means, stds, state_step_sizes, learn_gaussian_mle)

            # define loss and optimizer
            diff_in_states = self.placeholders['s'][:, 1:2] - self.placeholders['s'][:, :1]
            activations_vel_f = (1 / 32) / tf.sqrt(2 * np.pi * motion_params[:, :, 1] ** 2) * tf.exp(
                -(diff_in_states[:, :, 3] - motion_params[:, :, 0]) ** 2 / (2.0 * motion_params[:, :, 1] ** 2))
            activations_vel_rot = (1 / 32) / tf.sqrt(2 * np.pi * motion_params[:, :, 3] ** 2) * tf.exp(
                -(diff_in_states[:, :, 4] - motion_params[:, :, 2]) ** 2 / (2.0 * motion_params[:, :, 3] ** 2))
            losses['motion_mle'] = tf.reduce_mean(-tf.log(1e-16 + (tf.reduce_sum(activations_vel_f, axis=-1, name='loss1') * tf.reduce_sum(activations_vel_rot, axis=-1, name='loss2'))))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            # put everything together
            train_stages['train_motion_sampling'] = {
                         'train_op': optimizer.minimize(losses['motion_mle']),
                         'batch_iterator_names': {'train': 'train2', 'val': 'val2'},
                         'monitor_losses': ['motion_mle'],
                         'validation_loss': 'motion_mle',
                         'plot': lambda e: self.plot_motion_model(sess, next(batch_iterators['val2']), motion_samples, plot_task, state_step_sizes) if e % 1 == 0 else None
                         }

        else:
            motion_samples = self.motion_update(self.placeholders['a'][:,1],
                                    tf.tile(self.placeholders['s'][:, :1], [1, self.num_particles, 1]),
                                    means, stds, state_step_sizes, learn_gaussian_mle)

            # define loss and optimizer
            sq_distance = compute_sq_distance(motion_samples, self.placeholders['s'][:, 1:2], state_step_sizes)
            activations_sample = (1 / self.num_particles_float) / tf.sqrt(2 * np.pi * std ** 2) * tf.exp(
                -sq_distance / (2.0 * std ** 2))
            losses['motion_mle'] = tf.reduce_mean(-tf.log(1e-16 + tf.reduce_sum(activations_sample, axis=-1, name='loss')))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            # put everything together
            train_stages['train_motion_sampling'] = {
                         'train_op': optimizer.minimize(losses['motion_mle']),
                         'batch_iterator_names': {'train': 'train2', 'val': 'val2'},
                         'monitor_losses': ['motion_mle'],
                         'validation_loss': 'motion_mle',
                         'plot': lambda e: self.plot_motion_model(sess, next(batch_iterators['val2']), motion_samples, plot_task, state_step_sizes) if e % 1 == 0 else None
                         }

        # TRAIN MEASUREMENT MODEL

        # apply model for all pairs of observations and states in that batch
        test_particles = tf.tile(self.placeholders['s'][tf.newaxis, :, 0], [self.batch_size, 1, 1])
        measurement_model_out = self.measurement_update(encodings[:, 0], test_particles, means, stds)

        # define loss (correct -> 1, incorrect -> 0) and optimizer
        correct_samples = tf.diag_part(measurement_model_out)
        incorrect_samples = measurement_model_out - tf.diag(tf.diag_part(measurement_model_out))
        losses['measurement_heuristic'] = tf.reduce_sum(-tf.log(correct_samples)) / tf.cast(self.batch_size, tf.float32) \
                                          + tf.reduce_sum(-tf.log(1.0 - incorrect_samples)) / tf.cast(self.batch_size * (self.batch_size - 1), tf.float32)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # put everything together
        train_stages['train_measurement_model'] = {
                     'train_op': optimizer.minimize(losses['measurement_heuristic']),
                     'batch_iterator_names': {'train': 'train1', 'val': 'val1'},
                     'monitor_losses': ['measurement_heuristic'],
                     'validation_loss': 'measurement_heuristic',
                     'plot': lambda e: self.plot_measurement_model(sess, batch_iterators['val1'], measurement_model_out) if e % 1 == 0 else None
                     }

        # TRAIN PARTICLE PROPOSER

        if self.use_proposer:

            # apply model (but only compute gradients until the encoding,
            # otherwise we would unlearn it and the observation likelihood wouldn't work anymore)
            proposed_particles = self.propose_particles(tf.stop_gradient(encodings[:, 0]), self.num_particles, state_mins, state_maxs)

            # define loss and optimizer
            std = 0.2
            sq_distance = compute_sq_distance(proposed_particles, self.placeholders['s'][:, :1], state_step_sizes)
            activations = (1 / self.num_particles_float) / tf.sqrt(2 * np.pi * std ** 2) * tf.exp(
                -sq_distance / (2.0 * std ** 2))
            losses['proposed_mle'] = tf.reduce_mean(-tf.log(1e-16 + tf.reduce_sum(activations, axis=-1)))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            # put everything together
            train_stages['train_particle_proposer'] = {
                         'train_op': optimizer.minimize(losses['proposed_mle']),
                         'batch_iterator_names': {'train': 'train1', 'val': 'val1'},
                         'monitor_losses': ['proposed_mle'],
                         'validation_loss': 'proposed_mle',
                         'plot': lambda e: self.plot_particle_proposer(sess, next(batch_iterators['val1']), proposed_particles, plot_task) if e % 10 == 0 else None
                         }


        # END-TO-END TRAINING

        # model was already applied further up -> particle_list, particle_probs_list

        # define losses and optimizer
        # first loss (which is being optimized)
        sq_distance = compute_sq_distance(particle_list[:, :, :, 3:5], self.placeholders['s'][:, :, tf.newaxis, 3:5], state_step_sizes[3:5])
        activations = particle_probs_list[:, :] / tf.sqrt(2 * np.pi * self.particle_std ** 2) * tf.exp(
            -sq_distance / (2.0 * self.particle_std ** 2))
        losses['mle'] = tf.reduce_mean(-tf.log(1e-16 + tf.reduce_sum(activations, axis=2, name='loss')))

        # second loss (which we will monitor during execution)
        pred = self.particles_to_state(particle_list, particle_probs_list)

        sq_error = compute_sq_distance(pred[:, -1, 0:2], self.placeholders['s'][:, -1, 0:2], [1., 1.])
        sq_dist = compute_sq_distance(self.placeholders['s'][:, 0, 0:2], self.placeholders['s'][:, -1, 0:2], [1., 1.])
        losses['m/m'] = tf.reduce_mean(sq_error**0.5/sq_dist**0.5)

        sq_error = compute_sq_distance(pred[:, -1, 2:3], self.placeholders['s'][:, -1, 2:3], [np.pi/180.0])
        losses['deg/m'] = tf.reduce_mean(sq_error ** 0.5 / sq_dist ** 0.5)

        # optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # put everything together
        train_stages['train_e2e'] = {
                     'train_op': optimizer.minimize(losses['mle']),
                     'batch_iterator_names': {'train': 'train', 'val': 'val'},
                     'monitor_losses': ['m/m', 'deg/m', 'mle'],
                     'validation_loss': 'deg/m',
                     'plot': lambda e: self.plot_particle_filter(sess, next(batch_iterators['val_ex']), particle_list,
                                                                 particle_probs_list, state_step_sizes, plot_task) if e % 1 == 0 else None
                     }

        return losses, train_stages


    def load(self, sess, model_path, model_file='best_validation', statistics_file='statistics.npz', connect_and_initialize=True, modules=('encoder', 'mo_noise_generator', 'mo_transition_model', 'obs_like_estimator', 'particle_proposer')):

        if type(modules) not in [type(list()), type(tuple())]:
            raise Exception('modules must be a list or tuple, not a ' + str(type(modules)))

        # build the tensorflow graph
        if connect_and_initialize:
            # load training data statistics (which are needed to build the tf graph)
            statistics = dict(np.load(os.path.join(model_path, statistics_file)))
            for key in statistics.keys():
                if statistics[key].shape == ():
                    statistics[key] = statistics[key].item()  # convert 0d array of dictionary back to a normal dictionary

            # connect all modules into the particle filter
            self.connect_modules(**statistics)
            init = tf.global_variables_initializer()
            sess.run(init)

        # load variables
        all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        vars_to_load = []
        loaded_modules = set()
        for v in all_vars:
            for m in modules:
                if m in v.name:
                    vars_to_load.append(v)
                    loaded_modules.add(m)

        print('Loading all modules')

        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(model_path, model_file))

    # def fit(self, sess, data, model_path, train_individually, train_e2e, split_ratio, seq_len, batch_size, epoch_length, num_epochs, patience, learning_rate, dropout_keep_ratio, num_particles, particle_std, plot_task=None, plot=False):
    def fit(self, sess, data, model_path, train_individually, train_e2e, split_ratio, seq_len, batch_size, epoch_length, num_epochs, patience, learning_rate, dropout_keep_ratio, num_particles, particle_std, learn_gaussian_mle, plot_task=None, plot=False):
        if plot:
            plt.ion()

        self.particle_std = particle_std

        mean_loss_for_plot = np.zeros((1,))

        means, stds, state_step_sizes, state_mins, state_maxs = compute_statistics(data)


        data = split_data(data, ratio=split_ratio)

        epoch_lengths = {'train': epoch_length, 'val': epoch_length*2}
        batch_iterators = {'train': make_batch_iterator(data['train'], seq_len=seq_len, batch_size=batch_size),
                           'val': make_repeating_batch_iterator(data['val'], epoch_lengths['val'], batch_size=batch_size, seq_len=seq_len),
                           'train_ex': make_batch_iterator(data['train'], batch_size=batch_size, seq_len=seq_len),
                           'val_ex': make_batch_iterator(data['val'], batch_size=batch_size, seq_len=seq_len),
                           'train1': make_batch_iterator(data['train'], batch_size=batch_size, seq_len=1),
                           'train2': make_batch_iterator(data['train'], batch_size=batch_size, seq_len=2),
                            'val1': make_repeating_batch_iterator(data['val'], epoch_lengths['val'], batch_size=batch_size, seq_len=1),
                            'val2': make_repeating_batch_iterator(data['val'], epoch_lengths['val'], batch_size=batch_size, seq_len=2),
                        }

        # build the tensorflow graph by connecting all modules in the particles filter
        particles, particle_probs, encodings, particle_list, particle_probs_list = self.connect_modules(means, stds, state_mins, state_maxs, state_step_sizes, learn_gaussian_mle)

        # define losses and train stages for different ways of training (e.g. training individual models and e2e training)
        losses, train_stages = self.compile_training_stages(sess, batch_iterators, particle_list, particle_probs_list,
                                                            encodings, means, stds, state_step_sizes, state_mins,
                                                            state_maxs, learn_gaussian_mle, learning_rate, plot_task)

        # initialize variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # save statistics and prepare saving variables
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        np.savez(os.path.join(model_path, 'statistics'), means=means, stds=stds, state_step_sizes=state_step_sizes,
                 state_mins=state_mins, state_maxs=state_maxs)
        saver = tf.train.Saver()
        save_path = os.path.join(model_path, 'best_validation')

        # define the training curriculum
        curriculum = []
        if train_individually:
            if self.learn_odom:
                curriculum += ['train_odom']
            curriculum += ['train_measurement_model']
            curriculum += ['train_motion_sampling']
            if self.use_proposer:
                curriculum += ['train_particle_proposer']
        if train_e2e:
            curriculum += ['train_e2e']

        # split data for early stopping
        data_keys = ['train']
        if split_ratio < 1.0:
            data_keys.append('val')

        # define log dict
        log = {c: {dk: {lk: {'mean': [], 'se': []} for lk in train_stages[c]['monitor_losses']} for dk in data_keys} for c in curriculum}

        # go through curriculum
        for c in curriculum:

            stage = train_stages[c]
            best_val_loss = np.inf
            best_epoch = 0
            epoch = 0

            if c == 'train_e2e':
                saver.save(sess, os.path.join(model_path, 'before_e2e/best_validation'))
                np.savez(os.path.join(model_path, 'before_e2e/statistics'), means=means, stds=stds, state_step_sizes=state_step_sizes,
                 state_mins=state_mins, state_maxs=state_maxs)
            while epoch < num_epochs and epoch - best_epoch < patience:
                # training
                for dk in data_keys:
                    # don't train in the first epoch, just evaluate the initial parameters
                    if dk == 'train' and epoch == 0:
                        continue
                    # set up loss lists which will be filled during the epoch
                    loss_lists = {lk: [] for lk in stage['monitor_losses']}
                    for e in range(epoch_lengths[dk]):
                        # t0 = time.time()
                        # pick a batch from the right iterator
                        batch = next(batch_iterators[stage['batch_iterator_names'][dk]])
                        # define the inputs and train/run the model
                        input_dict = {**{self.placeholders[key]: batch[key] for key in 'osa'},
                                      **{self.placeholders['num_particles']: num_particles},
                                      }
                        if dk == 'train':
                            input_dict[self.placeholders['keep_prob']] = dropout_keep_ratio
                            input_dict[self.placeholders['is_training']] = True
                        monitor_losses = {l: losses[l] for l in stage['monitor_losses']}
                        if dk == 'train':
                            s_losses, _ = sess.run([monitor_losses, stage['train_op']], input_dict)
                        else:
                            s_losses = sess.run(monitor_losses, input_dict)

                        for lk in stage['monitor_losses']:
                            loss_lists[lk].append(s_losses[lk])

                    # after each epoch, compute and log statistics
                    for lk in stage['monitor_losses']:
                        log[c][dk][lk]['mean'].append(np.mean(loss_lists[lk]))
                        log[c][dk][lk]['se'].append(np.std(loss_lists[lk], ddof=1) / np.sqrt(len(loss_lists[lk])))


                # check whether the current model is better than all previous models
                if 'val' in data_keys:
                    current_val_loss = log[c]['val'][stage['validation_loss']]['mean'][-1]
                    mean_loss_for_plot = np.append(mean_loss_for_plot,current_val_loss)
                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        best_epoch = epoch
                        # save current model
                        saver.save(sess, save_path)
                        txt = 'epoch {:>3} >> '.format(epoch)
                    else:
                        txt = 'epoch {:>3} == '.format(epoch)
                else:
                    best_epoch = epoch
                    saver.save(sess, save_path)
                    txt = 'epoch {:>3} >> '.format(epoch)

                # after going through all data sets, do a print out of the current result
                for lk in stage['monitor_losses']:
                    txt += '{}: '.format(lk)
                    for dk in data_keys:
                        if len(log[c][dk][lk]['mean']) > 0:
                            txt += '{:.2f}+-{:.2f}/'.format(log[c][dk][lk]['mean'][-1], log[c][dk][lk]['se'][-1])

                    txt = txt[:-1] + ' -- '
                print(txt)

                if plot:
                    stage['plot'](epoch)

                epoch += 1

            # after running out of patience, restore the model with lowest validation loss
            saver.restore(sess, save_path)

        return log


    def predict(self, sess, batch, return_particles=False, **kwargs):
        # define input dict, use the first state only if we do tracking
        input_dict = {self.placeholders['o']: batch['o'],
                      self.placeholders['a']: batch['a'],
                      self.placeholders['num_particles']: 100}
        if self.init_with_true_state:
            input_dict[self.placeholders['s']] = batch['s'][:, :1]

        if return_particles:
            return sess.run([self.pred_states, self.particle_list, self.particle_probs_list], input_dict)
        else:
            return sess.run(self.pred_states, input_dict)


    def connect_modules(self, means, stds, state_mins, state_maxs, state_step_sizes, learn_gaussian_mle=False):

        # get shapes
        self.batch_size = tf.shape(self.placeholders['o'])[0]
        self.seq_len = tf.shape(self.placeholders['o'])[1]
        # we use the static shape here because we need it to build the graph
        self.action_dim = self.placeholders['a'].get_shape()[-1].value

        encodings = snt.BatchApply(self.encoder)((self.placeholders['o'] - means['o']) / stds['o'])

        # initialize particles
        if self.init_with_true_state:
            # tracking with known initial state
            initial_particles = tf.tile(self.placeholders['s'][:, 0, tf.newaxis, :], [1, self.num_particles, 1])
        else:
            # global localization
            if self.use_proposer:
                # propose particles from observations
                initial_particles = self.propose_particles(encodings[:, 0], self.num_particles, state_mins, state_maxs)
            else:
                # sample particles randomly
                initial_particles = tf.concat(
                    [tf.random_uniform([self.batch_size, self.num_particles, 1], state_mins[d], state_maxs[d]) for d in
                     range(self.state_dim)], axis=-1, name='particles')

        initial_particle_probs = tf.ones([self.batch_size, self.num_particles],
                                         name='particle_probs') / self.num_particles_float

        # assumes that samples has the correct size
        def permute_batch(x, samples):
            # get shapes
            batch_size = tf.shape(x)[0]
            num_particles = tf.shape(x)[1]
            sample_size = tf.shape(samples)[1]
            # compute 1D indices into the 2D array
            idx = samples + num_particles * tf.tile(
                tf.reshape(tf.range(batch_size), [batch_size, 1]),
                [1, sample_size])
            # index using the 1D indices and reshape again
            result = tf.gather(tf.reshape(x, [batch_size * num_particles, -1]), idx)
            result = tf.reshape(result, tf.shape(x[:,:sample_size]))
            return result


        def loop(particles, particle_probs, particle_list, particle_probs_list, additional_probs_list, i):

            num_proposed_float = tf.round((self.propose_ratio ** tf.cast(i, tf.float32)) * self.num_particles_float)
            num_proposed = tf.cast(num_proposed_float, tf.int32)
            num_resampled_float = self.num_particles_float - num_proposed_float
            num_resampled = tf.cast(num_resampled_float, tf.int32)

            if self.propose_ratio < 1.0:

                # resampling
                basic_markers = tf.linspace(0.0, (num_resampled_float - 1.0) / num_resampled_float, num_resampled)
                random_offset = tf.random_uniform([self.batch_size], 0.0, 1.0 / num_resampled_float)
                markers = random_offset[:, None] + basic_markers[None, :]  # shape: batch_size x num_resampled
                cum_probs = tf.cumsum(particle_probs, axis=1)
                marker_matching = markers[:, :, None] < cum_probs[:, None, :]  # shape: batch_size x num_resampled x num_particles
                samples = tf.cast(tf.argmax(tf.cast(marker_matching, 'int32'), dimension=2), 'int32')
                standard_particles = permute_batch(particles, samples)
                standard_particle_probs = tf.ones([self.batch_size, num_resampled])
                standard_particles = tf.stop_gradient(standard_particles)
                standard_particle_probs = tf.stop_gradient(standard_particle_probs)

                # motion update
                if learn_gaussian_mle:
                    standard_particles, _ = self.motion_update(self.placeholders['a'][:, i], standard_particles, means, stds, state_step_sizes, learn_gaussian_mle)
                else:
                    standard_particles = self.motion_update(self.placeholders['a'][:, i], standard_particles, means, stds, state_step_sizes, learn_gaussian_mle)


                # measurement update
                standard_particle_probs *= self.measurement_update(encodings[:, i], standard_particles, means, stds)

            if self.propose_ratio > 0.0:

                # proposed particles
                proposed_particles = self.propose_particles(encodings[:, i], num_proposed, state_mins, state_maxs)
                proposed_particle_probs = tf.ones([self.batch_size, num_proposed])


            # NORMALIZE AND COMBINE PARTICLES
            if self.propose_ratio == 1.0:
                particles = proposed_particles
                particle_probs = proposed_particle_probs

            elif self.propose_ratio == 0.0:
                particles = standard_particles
                particle_probs = standard_particle_probs

            else:
                standard_particle_probs *= (num_resampled_float / self.num_particles_float) / tf.reduce_sum(standard_particle_probs, axis=1, keep_dims=True)
                proposed_particle_probs *= (num_proposed_float / self.num_particles_float) / tf.reduce_sum(proposed_particle_probs, axis=1, keep_dims=True)
                particles = tf.concat([standard_particles, proposed_particles], axis=1)
                particle_probs = tf.concat([standard_particle_probs, proposed_particle_probs], axis=1)

            # NORMALIZE PROBABILITIES
            particle_probs /= tf.reduce_sum(particle_probs, axis=1, keep_dims=True)

            particle_list = tf.concat([particle_list, particles[:, tf.newaxis]], axis=1)
            particle_probs_list = tf.concat([particle_probs_list, particle_probs[:, tf.newaxis]], axis=1)

            return particles, particle_probs, particle_list, particle_probs_list, additional_probs_list, i + 1

        # reshapes and sets the first shape sizes to None (which is necessary to keep the shape consistent in while loop)
        particle_list = tf.reshape(initial_particles,
                                   shape=[self.batch_size, -1, self.num_particles, self.state_dim])
        particle_probs_list = tf.reshape(initial_particle_probs, shape=[self.batch_size, -1, self.num_particles])
        additional_probs_list = tf.reshape(tf.ones([self.batch_size, self.num_particles, 4]), shape=[self.batch_size, -1, self.num_particles, 4])

        # run the filtering process
        particles, particle_probs, particle_list, particle_probs_list, additional_probs_list, i = tf.while_loop(
            lambda *x: x[-1] < self.seq_len, loop,
            [initial_particles, initial_particle_probs, particle_list, particle_probs_list, additional_probs_list,
             tf.constant(1, dtype='int32')], name='loop')

        # compute mean of particles
        self.pred_states = self.particles_to_state(particle_list, particle_probs_list)
        self.particle_list = particle_list
        self.particle_probs_list = particle_probs_list

        return particles, particle_probs, encodings, particle_list, particle_probs_list

    def particles_to_state(self, particle_list, particle_probs_list):
        mean_position = tf.reduce_sum(particle_probs_list[:, :, :, tf.newaxis] * particle_list[:, :, :, :2], axis=2)
        mean_orientation = atan2(
            tf.reduce_sum(particle_probs_list[:, :, :, tf.newaxis] * tf.cos(particle_list[:, :, :, 2:3]), axis=2),
            tf.reduce_sum(particle_probs_list[:, :, :, tf.newaxis] * tf.sin(particle_list[:, :, :, 2:3]), axis=2))
        mean_velocity = tf.reduce_sum(particle_probs_list[:, :, :, tf.newaxis] * particle_list[:, :, :, 3:5], axis=2)
        return tf.concat([mean_position, mean_orientation, mean_velocity], axis=2)


    def plot_motion_model(self, sess, batch, motion_samples, task, state_step_sizes):

        # define the inputs and train/run the model
        input_dict = {**{self.placeholders[key]: batch[key] for key in 'osa'},
                      **{self.placeholders['num_particles']: 100},
                      }

        s_motion_samples = sess.run(motion_samples, input_dict)

        plt.figure('Motion Model')
        plt.gca().clear()
        for i in range(min(len(s_motion_samples), 10)):
            plt.scatter(s_motion_samples[i, :, 3] / state_step_sizes[3], s_motion_samples[i, :, 4] / state_step_sizes[4], color='blue', s=1)
            plt.scatter(batch['s'][i, 0, 3] / state_step_sizes[3], batch['s'][i, 0, 4] / state_step_sizes[4], color='black', s=1)
            plt.scatter(batch['s'][i, 1, 3] / state_step_sizes[3], batch['s'][i, 1, 4] / state_step_sizes[4], color='red', s=3)
            plt.plot(batch['s'][i, :2, 3] / state_step_sizes[3], batch['s'][i, :2, 4] / state_step_sizes[4], color='black')

        plt.xlim([0, 200])
        plt.ylim([-50, 50])
        plt.xlabel('translational vel')
        plt.ylabel('angular vel')
        plt.gca().set_aspect('equal')
        plt.pause(0.01)


    def plot_measurement_model(self, sess, batch_iterator, measurement_model_out):

        batch = next(batch_iterator)

        # define the inputs and train/run the model
        input_dict = {**{self.placeholders[key]: batch[key] for key in 'osa'},
                      **{self.placeholders['num_particles']: 100},
                      }

        s_measurement_model_out = sess.run([measurement_model_out], input_dict)

        plt.figure('Measurement Model Output')
        plt.gca().clear()
        plt.imshow(s_measurement_model_out[0], interpolation="nearest", cmap="viridis_r", vmin=0.0, vmax=1.0)

        plt.figure('Measurement Model Input')
        plt.clf()
        plt.scatter(batch['s'][:1, 0, 3], batch['s'][:1, 0, 4], marker='x', c=s_measurement_model_out[0][0,:1], vmin=0, vmax=1.0, cmap='viridis_r')
        plt.scatter(batch['s'][1:, 0, 3], batch['s'][1:, 0, 4], marker='o', c=s_measurement_model_out[0][0,1:], vmin=0, vmax=1.0, cmap='viridis_r')
        plt.xlabel('x_dot')
        plt.ylabel('theta_dot')
        plt.pause(0.01)


    def plot_particle_proposer(self, sess, batch, proposed_particles, task):

        # define the inputs and train/run the model
        input_dict = {**{self.placeholders[key]: batch[key] for key in 'osa'},
                      **{self.placeholders['num_particles']: 100},
                      }

        s_samples = sess.run(proposed_particles, input_dict)

        plt.figure('Particle Proposer')
        plt.gca().clear()
        plot_maze(task)

        for i in range(min(len(s_samples), 10)):
            color = np.random.uniform(0.0, 1.0, 3)
            plt.quiver(s_samples[i, :, 0], s_samples[i, :, 1], np.cos(s_samples[i, :, 2]), np.sin(s_samples[i, :, 2]), color=color, width=0.001, scale=100)
            plt.quiver(batch['s'][i, 0, 0], batch['s'][i, 0, 1], np.cos(batch['s'][i, 0, 2]), np.sin(batch['s'][i, 0, 2]), color=color, scale=50, width=0.003)

        plt.pause(0.01)


    def plot_particle_filter(self, sess, batch, particle_list,
                        particle_probs_list, state_step_sizes, task):

        s_states, s_particle_list, s_particle_probs_list, \
            = sess.run([self.placeholders['s'], particle_list,
                        particle_probs_list], #self.noise_scaler1(1.0), self.noise_scaler2(2.0)],
                       {**{self.placeholders[key]: batch[key] for key in 'osa'},
                        **{self.placeholders['num_particles']: 20},
                        })
        # print('learned motion noise factors {:.2f}/{:.2f}'.format(n1, n2))

        num_steps = s_particle_list.shape[1]

        for s in range(3):

            plt.figure('particle_evolution, example {}'.format(s))
            plt.clf()

            for d in range(5):

                plt.subplot(3, 2, [1, 3, 5, 2, 4][d])

                for i in range(num_steps):

                    plt.scatter(i * np.ones_like(s_particle_list[s, i, :, d]),
                                s_particle_list[s, i, :, d] / (1 if s == 0 else state_step_sizes[d]),
                                c=s_particle_probs_list[s, i, :], cmap='viridis_r', marker='o', s=6, alpha=0.5,
                                linewidths=0.05,
                                vmin=0.0,
                                vmax=0.1)
                    current_state = batch['s'][s, i, d] / (1 if s == 0 else state_step_sizes[d])
                    plt.plot([i], [current_state], 'o', markerfacecolor='None', markeredgecolor='k',
                             markersize=2.5)

                plt.xlabel('Time')
                plt.ylabel('State {}'.format(d))

        show_pause(pause=0.01)
