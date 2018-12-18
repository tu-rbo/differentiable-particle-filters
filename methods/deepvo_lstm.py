import tensorflow as tf
import sonnet as snt

from utils.data_utils import *
from utils.method_utils import compute_sq_distance
slim = tf.contrib.slim
from utils.data_utils_tfrecord import pad, LeakyReLU

class DeepVOLSTM():
    def __init__(self, init_with_true_state=False, model='2lstm', **unused_kwargs):

        self.placeholders = {'o': tf.placeholder('float32', [None, None, 384, 1280, 3], 'observations'),
                     'a': tf.placeholder('float32', [None, None, 3], 'actions'),
                     's': tf.placeholder('float32', [None, None, 3], 'states'),
                     'keep_prob': tf.placeholder('float32')}
        self.pred_states = None
        self.init_with_true_state = init_with_true_state
        self.model = model


        # build models
        self.encoder = snt.Module(name='FlowNetS', build=self.custom_build)

        # <-- action
        if self.model == '2lstm':
            self.rnn1 = snt.LSTM(1000)
            self.rnn2 = snt.LSTM(1000)

        self.belief_decoder = snt.Sequential([
            snt.Linear(3)
        ])

    def custom_build(self, inputs):
        """A custom build method to wrap into a sonnet Module."""
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            # Only backprop this network if trainable
                            trainable=True,
                            # He (aka MSRA) weight initialization
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=tf.nn.relu,
                            # We will do our own padding to match the original Caffe code
                            padding='VALID'):
            weights_regularizer = slim.l2_regularizer(0.0004)
            with slim.arg_scope([slim.conv2d], weights_regularizer=weights_regularizer):
                with slim.arg_scope([slim.conv2d], stride=2):
                    conv_1 = slim.conv2d(pad(inputs, 3), 64, 7, scope='conv1')
                    conv_2 = slim.conv2d(pad(conv_1, 2), 128, 5, scope='conv2')
                    conv_3 = slim.conv2d(pad(conv_2, 2), 256, 5, scope='conv3')

                conv3_1 = slim.conv2d(pad(conv_3), 256, 3, scope='conv3_1')
                with slim.arg_scope([slim.conv2d], num_outputs=512, kernel_size=3):
                    conv4 = slim.conv2d(pad(conv3_1), stride=2, scope='conv4')
                    conv4_1 = slim.conv2d(pad(conv4), scope='conv4_1')
                    conv5 = slim.conv2d(pad(conv4_1), stride=2, scope='conv5')
                    conv5_1 = slim.conv2d(pad(conv5), scope='conv5_1')
                conv6 = slim.conv2d(pad(conv5_1), 1024, 3, stride=2, scope='conv6')
                # conv6_1 = slim.conv2d(pad(conv6), 1024, 3, scope='conv6_1')

        # outputs = tf.nn.dropout(conv6,  self.placeholders['keep_prob'])
        outputs = snt.BatchFlatten()(conv6)   #outputs
        # outputs = snt.Linear(128)(outputs)
        # outputs = tf.nn.relu(outputs)

        return outputs

    def fit(self, sess, data, model_path, split_ratio, seq_len, batch_size, epoch_length, num_epochs, patience, learning_rate, dropout_keep_ratio, **unused_kwargs):

        # preprocess data
        data = split_data(data, ratio=split_ratio)
        epoch_lengths = {'train': epoch_length, 'val': epoch_length*2}
        batch_iterators = {'train': make_batch_iterator(data['train'], batch_size=batch_size, seq_len=seq_len),
                           'val': make_repeating_batch_iterator(data['val'], epoch_lengths['val'], batch_size=batch_size, seq_len=seq_len),
                           'train_ex': make_batch_iterator(data['train'], batch_size=batch_size, seq_len=seq_len),
                           'val_ex': make_batch_iterator(data['val'], batch_size=batch_size, seq_len=seq_len)}
        means, stds, state_step_sizes, state_mins, state_maxs = compute_staticstics(data['train'])

        self.connect_modules(means, stds, state_mins, state_maxs, state_step_sizes)

        # training

        sq_dist = compute_sq_distance(self.pred_states, self.placeholders['s'], state_step_sizes)
        losses = {'mse': tf.reduce_mean(sq_dist),
                  'mse_last': tf.reduce_mean(sq_dist[:, -1])}

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        gradients = optimizer.compute_gradients(losses['mse'])
        # clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
        train_op = optimizer.apply_gradients(gradients)

        init = tf.global_variables_initializer()
        sess.run(init)

        # save statistics and prepare saving variables
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        np.savez(os.path.join(model_path, 'statistics'), means=means, stds=stds, state_step_sizes=state_step_sizes,
                 state_mins=state_mins, state_maxs=state_maxs)
        saver = tf.train.Saver()
        save_path = model_path + '/best_validation'

        loss_keys = ['mse_last', 'mse']
        if split_ratio < 1.0:
            data_keys = ['train', 'val']
        else:
            data_keys = ['train']

        log = {dk: {lk: {'mean': [], 'se': []} for lk in loss_keys} for dk in data_keys}

        best_val_loss = np.inf
        best_epoch = 0
        i = 0
        while i < num_epochs and i - best_epoch < patience:
            # training
            loss_lists = dict()
            for dk in data_keys:
                loss_lists = {lk: [] for lk in loss_keys}
                for e in range(epoch_lengths[dk]):
                    batch = next(batch_iterators[dk])
                    if dk == 'train':
                        s_losses, _ = sess.run([losses, train_op], {**{self.placeholders[key]: batch[key] for key in 'osa'},
                                                                **{self.placeholders['keep_prob']: dropout_keep_ratio}})
                    else:
                        s_losses = sess.run(losses, {**{self.placeholders[key]: batch[key] for key in 'osa'},
                                                            **{self.placeholders['keep_prob']: 1.0}})
                    for lk in loss_keys:
                        loss_lists[lk].append(s_losses[lk])
                # after each epoch, compute and log statistics
                for lk in loss_keys:
                    log[dk][lk]['mean'].append(np.mean(loss_lists[lk]))
                    log[dk][lk]['se'].append(np.std(loss_lists[lk], ddof=1) / np.sqrt(epoch_lengths[dk]))

            # check whether the current model is better than all previous models
            if 'val' in data_keys:
                if log['val']['mse_last']['mean'][-1] < best_val_loss:
                    best_val_loss = log['val']['mse_last']['mean'][-1]
                    best_epoch = i
                    # save current model
                    saver.save(sess, save_path)
                    txt = 'epoch {:>3} >> '.format(i)
                else:
                    txt = 'epoch {:>3} == '.format(i)
            else:
                best_epoch = i
                saver.save(sess, save_path)
                txt = 'epoch {:>3} >> '.format(i)

            # after going through all data sets, do a print out of the current result
            for lk in loss_keys:
                txt += '{}: '.format(lk)
                for dk in data_keys:
                    txt += '{:.2f}+-{:.2f}/'.format(log[dk][lk]['mean'][-1], log[dk][lk]['se'][-1])
                txt = txt[:-1] + ' -- '
            print(txt)

            i += 1

        saver.restore(sess, save_path)

        return log


    def connect_modules(self, means, stds, state_mins, state_maxs, state_step_sizes):

        # tracking_info_full = tf.tile(((self.placeholders['s'] - means['s']) / stds['s'])[:, :1, :], [1, tf.shape(self.placeholders['s'])[1], 1])
        tracking_info = tf.concat([((self.placeholders['s'] - means['s']) / stds['s'])[:, :1, :], tf.zeros_like(self.placeholders['s'][:,1:,:])], axis=1)
        flag = tf.concat([tf.ones_like(self.placeholders['s'][:,:1,:1]), tf.zeros_like(self.placeholders['s'][:,1:,:1])], axis=1)

        preproc_o = snt.BatchApply(self.encoder)((self.placeholders['o'] - means['o']) / stds['o'])

        if self.model == '2lstm' or self.model == '2gru':
            lstm1_out, lstm1_final_state = tf.nn.dynamic_rnn(self.rnn1, preproc_o, dtype=tf.float32)
            lstm2_out, lstm2_final_state = tf.nn.dynamic_rnn(self.rnn2, lstm1_out, dtype=tf.float32)
            belief_list = lstm2_out

        self.pred_states = snt.BatchApply(self.belief_decoder)(belief_list)
        self.pred_states = self.pred_states * stds['s'] + means['s']


    def predict(self, sess, batch, **unused_kwargs):
        return sess.run(self.pred_states, {**{self.placeholders[key]: batch[key] for key in 'osa'},
                                           **{self.placeholders['keep_prob']: 1.0}})

    def load(self, sess, model_path, model_file='best_validation', statistics_file='statistics.npz', connect_and_initialize=True):

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
        for v in all_vars:
            print("%s %r %s" % (v, v, v.shape))

        # restore variable values
        saver = tf.train.Saver()  # <- var list goes in here
        saver.restore(sess, os.path.join(model_path, model_file))

        # print('Loaded the following variables:')
        # for v in all_vars:
        #     print(v.name)
