import tensorflow as tf
import sonnet as snt

from utils.data_utils import *
from utils.method_utils import compute_sq_distance
slim = tf.contrib.slim
from utils.data_utils_tfrecord import pad, LeakyReLU, _parse_function

class DeepVOLSTM():
    def __init__(self, init_with_true_state=False, model='2lstm', **unused_kwargs):

        # self.placeholders = {'o': tf.placeholder('float32', [None, None, 384, 1280, 3], 'observations'),
        #              'a': tf.placeholder('float32', [None, None, 3], 'actions'),
        #              's': tf.placeholder('float32', [None, None, 3], 'states'),
        #              'keep_prob': tf.placeholder('float32')}
        self.pred_states = None
        self.init_with_true_state = init_with_true_state
        self.model = model

        self.image_input = tf.keras.Input(shape=(None, 384, 1280, 3), name='input_layer')
        # build models
        # self.encoder = snt.Module(name='FlowNetS', build=self.custom_build)

        # <-- action
        # if self.model == '2lstm':


        # self.output_layer = snt.Linear(output_size=3, name='LSTM_to_out')

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
                print(conv6.get_shape())
        # outputs = tf.nn.dropout(conv6,  self.placeholders['keep_prob'])
        outputs = tf.reshape(conv6, [-1, conv6.get_shape()[1]*conv6.get_shape()[2]*conv6.get_shape()[3]])   #outputs
        print(outputs.get_shape())
        # outputs = snt.Linear(128)(outputs)
        # outputs = tf.nn.relu(outputs)

        return outputs

    def fit(self, sess, split_ratio, learning_rate, dropout_keep_ratio, **unused_kwargs):

        # preprocess data

        # data = split_data(data, ratio=split_ratio)
        # epoch_lengths = {'train': epoch_length, 'val': epoch_length*2}
        # batch_iterators = {'train': make_batch_iterator(data['train'], batch_size=batch_size, seq_len=seq_len),
        #                    'val': make_repeating_batch_iterator(data['val'], epoch_lengths['val'], batch_size=batch_size, seq_len=seq_len),
        #                    'train_ex': make_batch_iterator(data['train'], batch_size=batch_size, seq_len=seq_len),
        #                    'val_ex': make_batch_iterator(data['val'], batch_size=batch_size, seq_len=seq_len)}

        ###################### Using the dataset API ##################################
        filenames = ["/mnt/StorageDevice/KITTI/kitti_4.tfrecords"]
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_function)

        # stacked_dataset = dataset.batch(2)

        # stacked_dataset = stacked_dataset.shuffle(buffer_size=10000)
        # dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(10)
        dataset = dataset.batch(2)
        iterator = dataset.make_initializable_iterator()

        next_image, next_state = iterator.get_next()
        self.image_input = tf.keras.Input(tensor=next_image)
        # means, stds, state_step_sizes, state_mins, state_maxs = compute_staticstics(data['train'])   #Needs to be changed

        ###################### Connecting model and computing loss ##########################
        # sq_dist = tf.keras.losses.mean_squared_error(self.pred_states, next_state)
        # losses = {'mse': tf.reduce_mean(sq_dist),
        #           'mse_last': tf.reduce_mean(sq_dist[:, -1])}

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # gradients = optimizer.compute_gradients(losses['mse'])
        # clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
        # train_op = optimizer.apply_gradients(gradients)

        self.model = self.connect_modules()
        self.model.compile(optimizer=optimizer, loss=tf.keras.losses.mean_squared_error, target_tensors=[next_state], metrics=['mae'])

        init = tf.global_variables_initializer()
        sess.run(init)

        ################### Defining saving parameters #########################################
        saver = tf.train.Saver()
        save_path = './models/tmp' + '/best_validation'

        loss_keys = ['mse_last', 'mse']
        if split_ratio < 1.0:
            data_keys = ['train', 'val']
        else:
            data_keys = ['train']

        log = {lk: {'mean': [], 'se': []} for lk in loss_keys}

        best_val_loss = np.inf
        best_epoch = 0
        # save statistics and prepare saving variables
        # if not os.path.exists(model_path):
        #     os.makedirs(model_path)
        # np.savez(os.path.join(model_path, 'statistics'), means=means, stds=stds, state_step_sizes=state_step_sizes,
        #          state_mins=state_mins, state_maxs=state_maxs)

        # Compute for 100 epochs.
        for _ in range(1):
            sess.run(iterator.initializer)
            loss_lists = {lk: [] for lk in loss_keys}
            batches_length = 0
            while True:
                # try:
                # a = sess.run(next_image)
                # b = sess.run(next_state)
                self.model.fit(steps_per_epoch=1, epochs=5, verbose=2)
        #           s_losses, _ = sess.run([losses, train_op])
        #             for lk in loss_keys:
        #                 loss_lists[lk].append(s_losses[lk])
        #                 batches_length += 1
        #         except tf.errors.OutOfRangeError:
        #             break
        #     log[lk]['mean'].append(np.mean(loss_lists[lk]))
        #     log[lk]['se'].append(np.std(loss_lists[lk], ddof=1) / np.sqrt(batches_length))
        #
        #     txt = ''
        #     for lk in loss_keys:
        #         txt += '{}: '.format(lk)
        #         for dk in data_keys:
        #             txt += '{:.2f}+-{:.2f}/'.format(log[dk][lk]['mean'][-1], log[dk][lk]['se'][-1])
        #         txt = txt[:-1] + ' -- '
        #     print(txt)
        # # i = 0
        # # while i < num_epochs and i - best_epoch < patience:
        # #     # training
        # #     loss_lists = dict()
        # #     for dk in data_keys:
        # #         loss_lists = {lk: [] for lk in loss_keys}
        # #         for e in range(epoch_lengths[dk]):
        # #             batch = next(batch_iterators[dk])
        # #             if dk == 'train':
        # #                 s_losses, _ = sess.run([losses, train_op], {**{self.placeholders[key]: batch[key] for key in 'osa'},
        # #                                                         **{self.placeholders['keep_prob']: dropout_keep_ratio}})
        # #             else:
        # #                 s_losses = sess.run(losses, {**{self.placeholders[key]: batch[key] for key in 'osa'},
        # #                                                     **{self.placeholders['keep_prob']: 1.0}})
        # #             for lk in loss_keys:
        # #                 loss_lists[lk].append(s_losses[lk])
        # #         # after each epoch, compute and log statistics
        # #         for lk in loss_keys:
        # #             log[dk][lk]['mean'].append(np.mean(loss_lists[lk]))
        # #             log[dk][lk]['se'].append(np.std(loss_lists[lk], ddof=1) / np.sqrt(epoch_lengths[dk]))
        # #
        # #     # check whether the current model is better than all previous models
        # #     if 'val' in data_keys:
        # #         if log['val']['mse_last']['mean'][-1] < best_val_loss:
        # #             best_val_loss = log['val']['mse_last']['mean'][-1]
        # #             best_epoch = i
        # #             # save current model
        # #             saver.save(sess, save_path)
        # #             txt = 'epoch {:>3} >> '.format(i)
        # #         else:
        # #             txt = 'epoch {:>3} == '.format(i)
        # #     else:
        # #         best_epoch = i
        # #         saver.save(sess, save_path)
        # #         txt = 'epoch {:>3} >> '.format(i)
        # #
        # #     # after going through all data sets, do a print out of the current result
        # #     for lk in loss_keys:
        # #         txt += '{}: '.format(lk)
        # #         for dk in data_keys:
        # #             txt += '{:.2f}+-{:.2f}/'.format(log[dk][lk]['mean'][-1], log[dk][lk]['se'][-1])
        # #         txt = txt[:-1] + ' -- '
        # #     print(txt)
        # #
        # #     i += 1
        #
        # saver.restore(sess, save_path)

        return log


    def connect_modules(self):

        # encoder_output = self.custom_build(inputs)
        #
        # conv_flat = tf.keras.layers.Flatten()(encoder_output)
        #
        # rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [1000, 1000]]
        #
        # # create a RNN cell composed sequentially of a number of RNNCells
        # lstm_core = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
        #
        # output_sequence, final_state = tf.nn.dynamic_rnn(
        #     cell=lstm_core,
        #     inputs=encoder_output,
        #     time_major=False,
        #      dtype=tf.float32)
        #
        # self.pred_states = tf.layers.dense(output_sequence, units=3)
        # self.pred_states = self.pred_states * stds['s'] + means['s']

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='valid'))
        self.model.add(tf.keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='valid'))
        self.model.add(tf.keras.layers.Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding='valid'))
        self.model.add(tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        self.model.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='valid'))
        self.model.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        self.model.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='valid'))
        self.model.add(tf.keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
        self.model.add(tf.keras.layers.Conv2D(1024, kernel_size=(3, 3), strides=(2, 2), padding='valid'))
        self.model.add(tf.keras.layers.Flatten())

        time_distribute = tf.keras.layers.TimeDistributed(tf.keras.layers.Lambda(lambda x: self.model(x)))(self.image_input)

        lstm1 = tf.keras.layers.LSTM(1000, return_sequences=True)(time_distribute)
        lstm2 = tf.keras.layers.LSTM(1000, return_sequences=True)(lstm1)

        self.pred_states = tf.keras.layers.Dense(3, activation='relu')(lstm2)

        model = tf.keras.Model(inputs=[self.image_input], outputs=[self.pred_states])

        return model


    def predict(self, sess, batch, **unused_kwargs):
        return sess.run(self.pred_states, batch)

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
