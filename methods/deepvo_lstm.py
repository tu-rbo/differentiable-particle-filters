import tensorflow as tf
import sonnet as snt

from utils.data_utils import *
from utils.method_utils import compute_sq_distance
slim = tf.contrib.slim
from utils.data_utils_tfrecord import pad, LeakyReLU, _parse_function, concat_datasets
from utils.adam_accumulate import Adam_accumulate
from tensorflow.contrib.data.python.ops import sliding
from utils.data_utils_kitti import compute_statistics, load_kitti_sequences
import random
import math
import keras
from keras import backend as K

class DeepVOLSTM():
    def __init__(self, init_with_true_state=False, model='2lstm', **unused_kwargs):

        self.pred_states = None
        self.init_with_true_state = init_with_true_state
        self.model = model

    def conv_model(self, inputs):
        """A custom build method to wrap into a sonnet Module."""
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            # Only backprop this network if trainable
                            trainable=True,
                            # He (aka MSRA) weight initialization
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=tf.nn.relu,
                            # We will do our own padding to match the original Caffe code
                            padding='VALID',
                            reuse=tf.AUTO_REUSE):
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

        outputs = tf.layers.Flatten()(conv6)

        return outputs

    def fit(self, sess, learning_rate, batch_seq_len, num_of_samples, epoch_length, num_epochs, patience, batch_size, **unused_kwargs):

        training_sequences = [0, 2, 8, 9]
        full_seq_len = [4540, 4660, 4070, 1590]   # Sequence lengths for the training trajectories
        test_sequences = [10]

        training_filenames = ["../data/kitti_tf_records/kitti_{}.tfrecords".format(i) for i in training_sequences]
        test_filenames = ["../data/kitti_tf_records/kitti_{}.tfrecords".format(i) for i in test_sequences]

        ###### Training dataset creation ######

        training_dataset = self.generate_dataset(training_filenames, seq_len=full_seq_len, batch_seq_len=batch_seq_len,
                                                 num_of_samples=num_of_samples)
        iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
        train_init_op = iterator.make_initializer(training_dataset)

        ###### Test dataset creation ######
        test_dataset = self.generate_val_dataset(test_filenames, seq_len=[1590], batch_seq_len=batch_seq_len,
                                                 num_of_samples=100)
        test_init_op = iterator.make_initializer(test_dataset)

        self.image, self.state = iterator.get_next()

        ###### Model creation and defining inputs and outputs ######
        self.image_input = keras.Input(shape=(batch_seq_len-1, 384, 1280, 6), tensor=self.image[:, 1:, :, :, :])
        self.connect_modules()

        ###### Variables to restore for pre-trained Flownet
        vars_to_restore = [v for v in tf.global_variables() if "conv" in v.name]
        vars_to_restore_old = []
        dic_for_name_matching = {"conv":"FlowNetS/conv", "kernel": "weights", "bias":"biases", ":0":""}
        for count, v in enumerate(vars_to_restore):
            vars_to_restore_old.append(v.name)
            for i,j in dic_for_name_matching.items():
                vars_to_restore_old[count] = vars_to_restore_old[count].replace(i, j)
        vars_to_restore_dic = dict(zip(vars_to_restore_old, vars_to_restore))
        saver = tf.train.Saver(vars_to_restore_dic)
        saver.restore(sess, '/home/robotics/flownet2-tf/checkpoints/FlowNetS/flownet-S.ckpt-0')

        ###### Setup training loss with gradient averaging
        self.setup_train(average_gradients=batch_size, lr=learning_rate)
        sess.run(tf.global_variables_initializer())

        ###### Defining saving parameters ######
        saver = tf.train.Saver()
        save_path = '../models/tmp' + '/best_deepvo_model_loss_last_10_step_dpf_theta_Apr_4'

        loss_keys = ['mse_last', 'mse']
        # if split_ratio < 1.0:
        #     data_keys = ['train', 'val']
        # else:
        #     data_keys = ['train']

        log = {lk: {'mean': [], 'se': []} for lk in loss_keys}

        loss_mse = 10000.0
        patience_counter = 0
        epochs = 0
        best_loss = 10000
        while epochs<num_epochs and patience_counter<patience:
            epoch_lengths = 0
            sess.run(train_init_op)
            loss_degm = []
            loss_mm = []
            while epoch_lengths<epoch_length:
                for _ in range(batch_size):
                    tmp = self.train(sess)
                    loss_degm.append(tmp[0])
                    loss_mm.append(tmp[1])
                training_dataset = self.generate_dataset(training_filenames, seq_len=full_seq_len, batch_seq_len=batch_seq_len,
                                                         num_of_samples=num_of_samples)
                train_init_op = iterator.make_initializer(training_dataset)

                epoch_lengths += 1
            print ("Epoch:", epochs, " ------ ", "deg/m:", (sum(loss_degm)/len(loss_degm)), " m/m:", (sum(loss_mm)/len(loss_mm)))
            epochs += 1

            ###### Evaluating the model ######
            print("Test epoch")
            for _ in range(1):
                test_loss_mm= []
                test_loss_degm = []
                sess.run(test_init_op)
                while True:
                    try:
                        tmp = (sess.run([self._loss_op, self._loss]))
                        test_loss_degm.append(tmp[0])
                        test_loss_mm.append(tmp[1])
                    except tf.errors.OutOfRangeError:
                        break
                test_loss_mm = sum(test_loss_mm)/len(test_loss_mm)
                test_loss_degm = sum(test_loss_degm)/len(test_loss_degm)
                print ("Test epoch ------", "deg/m:", test_loss_degm, " m/m:", test_loss_mm)

                if test_loss_degm < best_loss:
                    print("Model saved")
                    saver.save(sess, save_path)
                    best_loss = test_loss_degm
                    patience_counter = 0
                else:
                    patience_counter += 1

        return log

    def train(self, session):
        feed_dict = dict()
        if self._average_gradients == 1:
            loss_degm, loss_mm, _ = session.run([self._loss_op, self._loss, self._train_op])
        else:
            loss_degm, loss_mm, grads = session.run([self._loss_op, self._loss, self._grad_op])
            self._gradients.append(grads)
            if len(self._gradients) == self._average_gradients:
                for i, placeholder in enumerate(self._grad_placeholders):
                    feed_dict[placeholder] = np.stack([g[i] for g in self._gradients], axis=0).mean(axis=0)
                session.run(self._train_op, feed_dict)
                self._gradients = []
        return loss_degm, loss_mm

    def setup_train(self, average_gradients=1, lr=1e-3):
        self._average_gradients = average_gradients
        sq_error_trans = tf.square(self.pred_states[:, -10:, 0:2] - self.state[:, -10:, 0:2])
        sq_dist = tf.reduce_sum(tf.square(self.state[:, :1, 0:2] - self.state[:, -1:, 0:2]))
        self._loss = tf.reduce_mean(sq_error_trans ** 0.5 / sq_dist ** 0.5)
        sq_error_rot = tf.square(wrap_angle(self.state[:, -10:, 2:3] - self.pred_states[:, -10:, 2:3]))*(180/np.pi)
        self._loss_op = tf.reduce_mean(sq_error_rot ** 0.5 / sq_dist ** 0.5)
        self._loss_final = tf.add(self._loss, self._loss_op)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        if average_gradients == 1:
            # This 'train_op' computes gradients and applies them in one step.
            self._train_op = optimizer.minimize(self._loss_op)
        else:
            # here 'train_op' only applies gradients passed via placeholders stored
            # in 'grads_placeholders. The gradient computation is done with 'grad_op'.
            grads_and_vars = optimizer.compute_gradients(self._loss_final)
            avg_grads_and_vars = []
            self._grad_placeholders = []
            for grad, var in grads_and_vars:
                grad_ph = tf.placeholder(grad.dtype, grad.shape)
                self._grad_placeholders.append(grad_ph)
                avg_grads_and_vars.append((grad_ph, var))
            self._grad_op = [x[0] for x in grads_and_vars]
            self._train_op = optimizer.apply_gradients(avg_grads_and_vars)
            self._gradients = []  # list to store gradients

    def connect_modules(self):

        conv_model = keras.Sequential()

        conv_model.add(keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='valid', name='conv1', input_shape=(384, 1280, 6)))
        conv_model.add(keras.layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='valid', name='conv2'))
        conv_model.add(keras.layers.Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding='valid', name='conv3'))
        conv_model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='conv3_1'))
        conv_model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='valid', name='conv4'))
        conv_model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='conv4_1'))
        conv_model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='valid', name='conv5'))
        conv_model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='conv5_1'))
        conv_model.add(keras.layers.Conv2D(1024, kernel_size=(3, 3), strides=(2, 2), padding='valid', name='conv6'))
        conv_model.add(keras.layers.Flatten())


        time_distribute = keras.layers.TimeDistributed(keras.layers.Lambda(lambda x: conv_model(x)))(self.image_input)

        lstm1 = keras.layers.CuDNNLSTM(1000, return_sequences=True)(time_distribute)
        lstm2 = keras.layers.CuDNNLSTM(1000, return_sequences=True)(lstm1)

        self.pred_states = keras.layers.Dense(3, activation='linear')(lstm2)
        self.pred_states = self.pred_states + self.state[:, :1, :]


    def generate_dataset(self, filenames, seq_len=[4540, 4660, 4070, 1590], num_of_samples=50, batch_seq_len=32):

        dataset = []
        for c, value in enumerate(filenames):
            dataset.append(tf.data.TFRecordDataset(value))   # Add all files to the dataset
            seq_skip = random.randint(1, seq_len[c] - (batch_seq_len*num_of_samples))        # Get a random integer
            shift = random.randint(1, math.floor((seq_len[c] - seq_skip) / num_of_samples))  # Compute the shift using the random integer
            dataset[c] = dataset[c].skip(seq_skip)            # Skip the first 'seq_skip' elements of the sequence
            dataset[c] = dataset[c].map(_parse_function)      # Extract the image and true state
            dataset[c] = dataset[c].apply(sliding.sliding_window_batch(batch_seq_len, shift))  # Apply sliding window operation
            dataset[c] = dataset[c].take(num_of_samples)  # Take 'num_of_samples' from each training sequence
        random.shuffle(dataset)        # Shuffle the dataset list

        ds0 = dataset[0]               # Concatenate the datasets
        for i in dataset[1:]:
            ds0 = ds0.concatenate(i)

        dataset = ds0.take(num_of_samples*len(seq_len))
        dataset = dataset.shuffle(buffer_size=20)
        dataset = dataset.batch(1)
        dataset = dataset.repeat()

        return dataset

    def generate_val_dataset(self, filenames, seq_len=[800, 270], num_of_samples=50, batch_seq_len=32):

        dataset = []
        for c, value in enumerate(filenames):
            dataset.append(tf.data.TFRecordDataset(value))
            seq_end = random.randint(1, seq_len[c])
            dataset[c] = dataset[c].take(seq_end)
            dataset[c] = dataset[c].map(_parse_function)
            shift = max(math.floor(seq_end / num_of_samples), 1)
            dataset[c] = dataset[c].apply(sliding.sliding_window_batch(batch_seq_len, shift))

        ds0 = dataset[0]
        for i in dataset[1:]:
            ds0 = ds0.concatenate(i)

        ds0 = ds0.take(num_of_samples*len(seq_len))
        dataset = ds0.batch(1)

        return dataset

    def generate_test_dataset(self, filenames, batch_seq_len = 32):

        dataset = []
        for c, value in enumerate(filenames):
            dataset.append(tf.data.TFRecordDataset(value))   # Add all files to the dataset
            dataset[c] = dataset[c].map(_parse_function)     # Extract the image and true state
            dataset[c] = dataset[c].apply(sliding.sliding_window_batch(batch_seq_len, 100))  # Apply sliding window operation

        ds0 = dataset[0]                                     # Concatenate the datasets
        for i in dataset[1:]:
            ds0 = ds0.concatenate(i)

        dataset = ds0.batch(1)                                # Batch the dataset with batch size of 1
        return dataset

    def predict(self, sess, handle, test_handle):
        prediction, true_state = sess.run([self.pred_states, self.state], feed_dict={handle: test_handle})
        return prediction, true_state


    def load(self, sess, model_path, batch_seq_len, num_of_samples, **unused_kwargs):

        # build the tensorflow graph

        self.image_input = keras.Input(shape=(batch_seq_len-1, 384, 1280, 6), tensor=self.image[:, 1:, :, :, :])
        self.connect_modules()
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        return sess

