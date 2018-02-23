import numpy as np
import matplotlib.pyplot as plt
import os
import math
import glob
from time import time
from PIL import Image
from utils.plotting_utils import plot_trajectories, plot_maze, plot_observations

def wrap_angle(angle):
    return ((angle - np.pi) % (2 * np.pi)) - np.pi

def rotation_matrix(x):
    rot_psi = np.array([[math.cos(x[2]), -math.sin(x[2]), 0], [math.sin(x[2]), math.cos(x[2]), 0], [0, 0, 1]])
    rot_theta = np.array([[math.cos(x[1]), 0, math.sin(x[1])], [0, 1, 0], [-math.sin(x[1]), 0, math.cos(x[1])]])
    rot_phi = np.array([[1, 0, 0], [0, math.cos(x[0]), -math.sin(x[0])], [0, math.sin(x[0]), math.cos(x[0])]])
    R = np.dot(rot_psi,np.dot(rot_theta,rot_phi))
    return R

def read_oxts_data(oxts, oxts_prev, oxts_init):

    with open(oxts, 'r') as f:
        oxts_data = np.loadtxt(f)

    with open(oxts_init, 'r') as f:
        oxts_init = np.loadtxt(f)

    with open(oxts_prev, 'r') as f:
        oxts_prev = np.loadtxt(f)

    north  = (oxts_data[0] - oxts_init[0]) * 6378137 * math.pi / 180
    east = (oxts_data[1] - oxts_init[1]) * 6378137 * math.pi / 180 * math.cos(oxts_init[0] * math.pi / 180)
    alpha = (oxts_data[22] - oxts_prev[22])/0.103
    state = np.array([east, north, -oxts_data[5], oxts_data[8], -oxts_data[22]])
    action = np.array([oxts_data[14], oxts_data[15], alpha])

    return state, action

def load_image(img_file):
    return np.asarray(Image.open(img_file), 'float32')

def image_input(img1, img2):
    return np.concatenate((img1, img1-img2), axis=2)

def load_data_for_stats(oxts_data, images, diff_images, seq_num, base_frame):

    state = np.zeros((len(oxts_data), 6))
    action = np.zeros((len(oxts_data), 3))
    with open(base_frame, 'r') as f:
        data = np.loadtxt(f)
        base_lat = data[0]
        base_long = data[1]

    for ii in range(len(oxts_data)):
        with open(oxts_data[ii], 'r') as f:
            data = np.loadtxt(f)
        # if ii==0: #or ii in seq_num[:-1]:
        #     base_lat = data[0]
        #     base_long = data[1]
        north = (data[0] - base_lat) * 6378137 * math.pi / 180
        east = (data[1] - base_long) * 6378137 * math.pi / 180 * math.cos(base_lat * math.pi / 180)
        state[ii,:] = np.array([north, east, data[5], data[6], data[7], data[22]])
        action[ii,:] = np.array([data[8], data[14], data[15]])

    images_per_seq = 100
    obs = np.zeros((len(seq_num) * images_per_seq, 50, 150, 6))
    for ii in range(1, len(seq_num)-1):
        for jj in range(images_per_seq):
            img1 = load_image(images[seq_num[ii - 1] + jj])
            obs[images_per_seq*(ii-1)+jj,:,:,:3] = img1
            img2 = load_image(diff_images[seq_num[ii - 1] + jj])
            obs[images_per_seq*(ii-1)+jj,:,:,3:6] = img2


    data_for_stats = {'s': state, 'a': action, 'o': obs}

    return data_for_stats

# loading all sequences for KITTI
def load_kitti_sequences(sequence_list=None):

    print('Loading KITTI DATA')
    t1 = time()
    try:
        if sequence_list is None:
            print('Trying to load from cache ... ')
            data = dict(np.load('../data/kitti.npz'))
            t2 = time()
            print('Done! ({:.2f}s)'.format(t2-t1))
        else:
            raise Exception

    except:

        if sequence_list is None:
            sequence_list = list(range(11))

        print('Cache not found, loading from KITTI_dataset')
        path = "../data/kitti"

        image_seq_1_full_path = ["{}/{:02d}/image_2".format(path, x) for x in sequence_list]
        image_seq_2_full_path = ["{}/{:02d}/image_3".format(path, x) for x in sequence_list]

        # Extract original image and difference image
        input_image_file = []
        seq_num = []
        for ii in range(len(sequence_list)):
            for name in glob.glob('{}/image*.png'.format(image_seq_1_full_path[ii])):
                input_image_file = input_image_file + [name]
            for name in glob.glob('{}/image*.png'.format(image_seq_2_full_path[ii])):
                input_image_file = input_image_file + [name]

        input_image_file.sort()
        # print(len(input_image_file))

        oxts_seq_1 = ["%.2d_image1.txt" % i for i in sequence_list]
        oxts_seq_1 = oxts_seq_1 + ["%.2d_image2.txt" % i for i in sequence_list]
        oxts_seq_1.sort()
        oxts_seq_1_full_path = ["{}/{}".format(path, x) for x in oxts_seq_1]
        output_oxts_file = oxts_seq_1_full_path

        sequence_starts_ends = [[0, 4540], [0, 1100], [0, 4660], [0, 800], [0, 270], [0, 2760], [0, 1100], [0, 1100], [1100, 5170], [0, 1590],
         [0, 1200]]
        data_values = np.array([sequence_starts_ends[i] for i in sequence_list])
        seq_num = np.zeros((2*data_values.shape[0],))
        weights = np.zeros((2*data_values.shape[0],))

        for ii in range(data_values.shape[0]):
            if ii == 0:
                seq_num[0] = data_values[ii,1] - data_values[ii,0]
                seq_num[1] = seq_num[0] + data_values[ii,1] - data_values[ii,0]
                weights[0] = weights[1] = data_values[ii,1] - data_values[ii,0]
            else:
                seq_num[2*ii] = seq_num[2*ii-1] + data_values[ii, 1] - data_values[ii, 0]
                seq_num[2*ii+1] = seq_num[2*ii] + data_values[ii, 1] - data_values[ii, 0]
                weights[2*ii] = weights[2*ii+1] = data_values[ii, 1] - data_values[ii, 0]

        # seq_num is an array of the cumulative sequence lengths, e.g. [100, 300, 350] for sequences of length 100, 200, 50
        seq_num = seq_num.astype(int)
        weights = weights/seq_num[-1]
        print(seq_num, weights)

        o = np.zeros((seq_num[-1], 50, 150, 6))
        count = 0
        # for all sequences
        for ii in range(len(seq_num)):
            # find out the start and end of the current sequence
            if ii == 0:
                start = 1
            else:
                start = seq_num[ii-1]+ii+1

            # load first image
            prev_image = load_image(input_image_file[start-1])
            # for all time steps
            for jj in range(start, seq_num[ii]+ii+1):
                # load next image
                cur_image = load_image(input_image_file[jj])
                # observation from current and last image
                o[count, :, :, :] = image_input(cur_image, prev_image)
                prev_image = cur_image
                count += 1

        a = np.zeros((seq_num[-1], 3))
        s = np.zeros((seq_num[-1], 5))
        for ii in range(len(output_oxts_file)):

            # load text file
            with open(output_oxts_file[ii], 'r') as f:
                tmp = np.loadtxt(f)

            start = 0 if ii == 0 else seq_num[ii-1]

            x = tmp[:, 11]
            y = -tmp[:, 3]
            theta = -np.arctan2(-tmp[:, 8], tmp[:, 10])
            s[start:seq_num[ii], 0] = x[1:]  # x
            s[start:seq_num[ii], 1] = y[1:]  # y
            s[start:seq_num[ii], 2] = theta[1:]  # angle
            s[start:seq_num[ii], 3] = np.sqrt((y[1:] - y[:-1]) ** 2 + (x[1:] - x[:-1]) ** 2) / 0.103  # forward vel
            s[start:seq_num[ii], 4] = wrap_angle(theta[1:] - theta[:-1])/0.103  # angular vel

        t2 = time()
        print('Done! ({:.2f}s)'.format(t2 - t1))
        print('By default not saving data to cache ... ')
        # if len(sequence_list) == 11:
        #     print('Saving data to cache in ../data/kitti')
        #     np.savez('../data/kitti', s=s, a=a, o=o, seq_num=seq_num, weights=weights)

        print(s.shape, a.shape, o.shape, seq_num.shape, weights.shape)

        data = {'s': s,
                'a': a,
                'o': o,
                'seq_num': seq_num,
                'weights': weights
                }

    for key in 'osa':
        # add dimension to be consistent with the batch x seq x dim convention
        data[key] = data[key][:, np.newaxis, :]

    return add_mirrored_data(data)


def load_data(data_path='data/100s', filename='nav01_train', steps_per_episode=100, num_episodes=None):

    data = dict(np.load(os.path.join(data_path, '100s', filename + '.npz')))
    data = dict(np.load(os.path.join(data_path, filename + '.npz')))

    # reshape data
    for key in data.keys():
        # 'vel': (100, 1000, 3), 'rgbd': (100, 1000, 32, 32, 4), 'pose': (100, 1000, 3)
        if num_episodes is not None:
            data[key] = data[key][:num_episodes*steps_per_episode]
        data[key] = np.reshape(data[key], [-1, steps_per_episode] + list(data[key].shape[1:])).astype('float32')

    # convert degrees into radients and
    for key in ['pose', 'vel']:
        data[key][:, :, 2] *= np.pi / 180
    # angles should be between -pi and pi
    data['pose'][:, :, 2] = wrap_angle(data['pose'][:, :, 2])

    abs_d_x = (data['pose'][:, 1:, 0:1] - data['pose'][:, :-1, 0:1])
    abs_d_y = (data['pose'][:, 1:, 1:2] - data['pose'][:, :-1, 1:2])
    d_theta = wrap_angle(data['pose'][:, 1:, 2:3] - data['pose'][:, :-1, 2:3])
    s = np.sin(data['pose'][:, :-1, 2:3])
    c = np.cos(data['pose'][:, :-1, 2:3])
    rel_d_x = c * abs_d_x + s * abs_d_y
    rel_d_y = s * abs_d_x - c * abs_d_y

    # define observations, states, and actions for the filter, use current and previous velocity measurement as action
    # and ignore the 0th timestep because we don't have the previous velocity of that step
    return {'o': data['rgbd'][:, 1:, :, :, :3],
            's': data['pose'][:, 1:, :],
            'a': np.concatenate([rel_d_x, rel_d_y, d_theta], axis=-1)}
            # 'a': np.concatenate([data['vel'][:, :-1, None, :], data['vel'][:, 1:, None, :]], axis=-2)}

def compute_statistics(data):
    means = dict()
    stds = dict()
    state_step_sizes = []
    state_mins = []
    state_maxs = []

    for key in 'osa':
        # compute means
        axis = tuple(range(len(data[key].shape) - 1))  # means std by averaging over all but the last dimension
        means[key] = np.mean(data[key], axis=axis, keepdims=True)

        # compute stds
        axis = tuple(range(len(data[key].shape) - 1))  # compute std by averaging over all but the last dimension
        stds[key] = np.std(data[key] - means[key], axis=axis, keepdims=True)

    # compute average step size in x, y, and theta for the distance metric
    for i in range(5):
        for j in range(len(data['seq_num'])):
            if j == 0:
                steps = np.reshape(data['s'][1:data['seq_num'][j], :, i] - data['s'][0:data['seq_num'][j]-1, :, i], [-1])
            else:
                steps = np.append(steps, np.reshape(data['s'][data['seq_num'][j-1]+1:data['seq_num'][j], :, i] - data['s'][data['seq_num'][j-1]:data['seq_num'][j]-1, :, i], [-1]))
            if i == 2:
                steps = wrap_angle(steps)
        state_step_sizes.append(np.mean(abs(steps)))
    state_step_sizes[0] = state_step_sizes[1] = (state_step_sizes[0] + state_step_sizes[1]) / 2
    state_step_sizes = np.array(state_step_sizes)

    # compute min and max in x, y and theta
    for i in range(5):
        state_mins.append(np.min(data['s'][:, :, i]))
        state_maxs.append(np.max(data['s'][:, :, i]))
    state_mins = np.array(state_mins)
    state_maxs = np.array(state_maxs)

    return means, stds, state_step_sizes, state_mins, state_maxs


def split_data(data, ratio=0.8, categories=['train', 'val']):
    split_data = {categories[0]: dict(), categories[1]: dict()}
    split_point_seq = math.floor(data['seq_num'].shape[0] * ratio)
    split_point_data = data['seq_num'][split_point_seq-1]
    for key in data.keys():
        if key == 'seq_num':
            split_data[categories[0]][key] = data[key][:split_point_seq]
            split_data[categories[1]][key] = data[key][split_point_seq:] - data[key][split_point_seq-1]
        elif key == 'weights':
            split_data[categories[0]][key] = data[key][:split_point_seq]
            split_data[categories[0]][key] = split_data[categories[0]][key]/np.sum(split_data[categories[0]][key])
            split_data[categories[1]][key] = data[key][split_point_seq:]
            split_data[categories[1]][key] = split_data[categories[1]][key]/np.sum(split_data[categories[1]][key])
        else:
            split_data[categories[0]][key] = data[key][:split_point_data]
            split_data[categories[1]][key] = data[key][split_point_data:]
    for key in split_data.keys():
        print('SPLIT --> {}: {}'.format(key, len(split_data[key]['seq_num'])))
    return split_data


def reduce_data(data, num_episodes):
    new_data = dict()
    for key in 'osa':
        new_data[key] = data[key][:num_episodes]
    return new_data

def shuffle_data(data):
    new_data = dict()
    shuffled_indices = np.random.permutation(len(data['o']))
    for key in 'osa':
        new_data[key] = data[key][shuffled_indices]
    return new_data

def remove_state(data, provide_initial_state=False):
    new_data = dict()
    new_data['o'] = data['o']
    new_data['a'] = data['a']
    if provide_initial_state:
        new_data['s'] = data['s'][..., :1, :]
    return new_data


def noisify_data_condition(data, condition):
    print('condition', condition)
    if condition == 'odom0_imgTG':
        return noisyfy_data(data, odom_noise_factor=0.0)
    elif condition == 'odom5_imgTG':
        return noisyfy_data(data, odom_noise_factor=0.5)
    elif condition == 'odom10_imgTG':
        return noisyfy_data(data)
    elif condition == 'odom20_imgTG':
        return noisyfy_data(data, odom_noise_factor=2.0)
    elif condition == 'odomX_imgTG':
        data = noisyfy_data(data, odom_noise_factor=0.0)
        # shuffle actions to basically make them meaningless
        shape = data['a'].shape
        a = np.reshape(data['a'], [-1, shape[-1]])
        np.random.shuffle(a)
        data['a'] = np.reshape(a, shape)
        return data
    elif condition == 'odom10_imgC':
        return noisyfy_data(data, img_noise_factor=0.0, img_random_shift=False)
    elif condition == 'odom10_imgG':
        return noisyfy_data(data, img_noise_factor=1.0, img_random_shift=False)
    elif condition == 'odom10_imgT':
        return noisyfy_data(data, img_noise_factor=0.0, img_random_shift=True)
    elif condition == 'odom10_imgX':
        data = noisyfy_data(data, img_noise_factor=0.0, img_random_shift=False)
        # shuffle observations to basically make them meaningless
        shape = data['o'].shape
        o = np.reshape(data['o'], [-1, shape[-1]])
        np.random.shuffle(o)
        data['o'] = np.reshape(o, shape)
        return data

def noisyfy_data(data, odom_noise_factor=1.0, img_noise_factor=1.0, img_random_shift=True):
    print("noisyfying data ... ")
    new_data = dict()
    new_data['s'] = data['s']
    new_data['a'] = data['a'] * np.random.normal(1.0, 0.1 * odom_noise_factor, data['a'].shape)
    new_data['seq_num'] = data['seq_num']
    new_data['o'] = data['o']
    return new_data

def make_batch_iterator(data, batch_size=32, seq_len=10, use_mirrored_data=True):

    while True:
        o = np.zeros((batch_size, seq_len, 50, 150, 6))
        a = np.zeros((batch_size, seq_len, 3))
        s = np.zeros((batch_size, seq_len, 5))
        for ii in range(batch_size):
            trajectory = np.random.choice(len(data['seq_num']), p = data['weights'])

            start = 0 if trajectory == 0 else data['seq_num'][trajectory-1]
            start_steps = np.random.random_integers(start, data['seq_num'][trajectory] - seq_len - 1)
            key_append = '-m' if use_mirrored_data and ii >= batch_size / 2 else ''
            o[ii, :, :, :, :] = data['o'+key_append][start_steps:start_steps + seq_len, 0]
            a[ii, :, :] = data['a'][start_steps:start_steps + seq_len, 0]
            s[ii, :, :] = data['s'+key_append][start_steps:start_steps + seq_len, 0]

        batches =  {'o': o, 'a': a, 's': s}
        yield batches

def make_repeating_batch_iterator(data, epoch_len, batch_size=32, seq_len=10, use_mirrored_data=True):

    o = np.zeros((batch_size, seq_len, 50, 150, 6))
    a = np.zeros((batch_size, seq_len, 3))
    s = np.zeros((batch_size, seq_len, 5))
    start_steps = np.zeros((epoch_len, batch_size))
    trajectory = np.random.random_integers(0, len(data['seq_num']) - 1, size=[epoch_len, batch_size])
    for kk in range(epoch_len):
        for ii in range(batch_size):
            start = 0 if trajectory[kk, ii] == 0 else data['seq_num'][trajectory[kk, ii] - 1]
            start_steps[kk, ii] = np.random.random_integers(start, data['seq_num'][trajectory[kk, ii]] - seq_len - 1)

    start_steps = start_steps.astype(int)

    while True:
        for kk in range(epoch_len):
            for ii in range(batch_size):
                ssteps = start_steps[kk, ii]
                key_append = '-m' if use_mirrored_data and ii >= batch_size / 2 else ''
                o[ii, :, :, :, :] = data['o' + key_append][ssteps:ssteps + seq_len, 0]
                a[ii, :, :] = data['a'][ssteps:ssteps + seq_len, 0]
                s[ii, :, :] = data['s' + key_append][ssteps:ssteps + seq_len, 0]
            batches =  {'o': o, 'a': a, 's': s}
            yield batches


def make_complete_batch_iterator(data, batch_size=1000, seq_len=10):
    num_episodes = len(data['s'])
    num_start_steps = len(data['s'][0]) - seq_len
    batch_indices = [(i, j) for i in range(num_episodes) for j in range(num_start_steps)]
    while batch_indices != []:
        batches = {k: np.concatenate([data[k][i:i + 1, j:j + seq_len] for (i, j) in batch_indices[:batch_size]]) for k in data.keys}
        batch_indices = batch_indices[batch_size:]
        yield batches


def make_batch_iterator_for_evaluation(data, start_step, trajectory, batch_size = 1, seq_len=10):
    while True:
        o = np.zeros((batch_size,seq_len, 50, 150, 6))
        a = np.zeros((batch_size, seq_len, 3))
        s = np.zeros((batch_size, seq_len, 5))
        for ii in range(batch_size):

            # shift start step to where the sequence begins
            if trajectory != 0:
                start_step = data['seq_num'][trajectory-1] + start_step

            for jj in range(seq_len):
                o[ii, jj, :, :, :] = data['o'][start_step+jj, :, :, :]
                a[ii, jj, :] = data['a'][start_step+jj, :]
                s[ii, jj, :] = data['s'][start_step+jj, :]

        batches =  {'o': o, 'a': a, 's': s}
        yield batches

def plot_observation_check(data, means, stds):

    observations = data['o']
    plt.ion()
    for o in observations:
        # shape(o): (1, 50, 150, 6)
        # shape(means['o']) = (1, 1, 50, 150, 6)

        norm_o = (o - means['o'][0]) / stds['o'][0]

        for d in range(o.shape[-1]):
            plt.figure(d)
            plt.clf()

            plt.imshow(norm_o[0, :, :, d], interpolation='nearest', cmap='coolwarm', vmin=-3, vmax=3)
            print('dimension {}: ({}-{})'.format(d, np.min(o[:, :, d]), np.max(o[:, :, d])))

        for d in range(2):
            plt.figure(10 + d)
            plt.clf()
            if d == 0:
                plt.imshow(np.clip(o[0, :, :, 3*d:3*(d+1)]/255.0, 0.0, 1.0), interpolation='nearest')
            else:
                plt.imshow(o[0, :, :, 3*d:3*(d+1)]/255.0/2 + 0.5, interpolation='nearest')

        d = 2
        plt.figure('means')
        plt.clf()
        plt.imshow(means['o'][0, 0, :, :, d], interpolation='nearest', cmap='coolwarm', vmin=0, vmax=255)

        plt.figure('stds')
        plt.clf()
        plt.imshow(stds['o'][0, 0, :, :, d], interpolation='nearest', cmap='coolwarm', vmin=0, vmax=255)

        plt.pause(10)

def plot_video(data):
    observations = data['o']
    plt.ion()
    for i, o in enumerate(observations):
        # shape(o): (1, 50, 150, 6)
        # shape(means['o']) = (1, 1, 50, 150, 6)

        d = 0
        plt.figure(10 + d)
        plt.clf()
        plt.imshow(np.clip(o[0, :, :, 3 * d:3 * (d + 1)] / 255.0, 0.0, 1.0), interpolation='nearest')

        plt.pause(0.05)
        print(i)

def plot_sequences(data, means, stds, state_step_sizes):

    for k, i in enumerate(data['seq_num']):
        print(i)
        if k < len(data['seq_num'])-1:
            plt.figure(k)
            for j in range(4):
                plt.subplot(4,1,j+1)
                plt.imshow(np.clip(data['o'][i-2+j, 0, :, :, :3]/255.0, 0.0, 1.0), interpolation='nearest')
                plt.xticks([])
                plt.yticks([])
                plt.ylabel(i-2+j)

    plt.figure('trajectories')
    last_seq_num = 0
    for k, i in enumerate(data['seq_num']):
        if k % 2 == 0:
            plt.plot(data['s'][last_seq_num:i, 0, 0], data['s'][last_seq_num:i, 0, 1], label="trajectory {}/{}".format(k, k+1))
            plt.quiver(data['s'][last_seq_num:i, 0, 0], data['s'][last_seq_num:i, 0, 1],
                       np.cos(data['s'][last_seq_num:i, 0, 2]), np.sin(data['s'][last_seq_num:i, 0, 2]), color='k')
        last_seq_num = i

    plt.gca().set_aspect('equal')
    plt.legend()

    plt.figure('normalized state')
    norm_states = (data['s'] - means['s'][0]) / stds['s'][0]
    for d in range(data['s'].shape[-1]):
        plt.plot(norm_states[:, 0, d], label='state dim {}'.format(d))

    plt.figure('state')
    for d in range(data['s'].shape[-1]):
        plt.plot(data['s'][:, 0, d], label='state dim {}'.format(d))

    plt.figure('scaled state')
    for d in range(data['s'].shape[-1]):
        plt.plot(data['s'][:, 0, d] / state_step_sizes[d], label='state dim {}'.format(d))
        print('dim {}: state step size: {}'.format(d, state_step_sizes[d]))
    for i in data['seq_num']:
        plt.plot([i, i], [-3, 3], 'k')

    plt.legend()
    plt.show()



def add_mirrored_data(data):
    data['o-m'] = data['o'][..., ::-1, :]
    data['s-m'] = np.concatenate([ data['s'][..., 0:1], # keep x
                                         -data['s'][..., 1:2], # invert y
                                         -data['s'][..., 2:3], # invert angle
                                          data['s'][..., 3:4], # keep foward vel
                                         -data['s'][..., 4:5], # invert angular vel
                                         ], axis=-1)
    return data

if __name__ == '__main__':

    data = load_kitti_sequences()

    print(data['o'].shape)
    means, stds, state_step_sizes, state_mins, state_maxs = compute_statistics(data)
    print(data['o'].shape)


