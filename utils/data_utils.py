import numpy as np
import matplotlib.pyplot as plt
import os

from utils.plotting_utils import plot_trajectories, plot_maze, plot_observations, plot_trajectory

def wrap_angle(angle):
    return ((angle - np.pi) % (2 * np.pi)) - np.pi

def mix_data(file_in1, file_in2, file_out, steps_per_episode=100, num_episodes=1000):
    data1 = dict(np.load(file_in1))
    data2 = dict(np.load(file_in2))
    data_mix = dict()
    for key in data1.keys():
        d1 = data1[key][:steps_per_episode*num_episodes//2]
        d2 = data2[key][:steps_per_episode*num_episodes//2]
        data_mix[key] = np.concatenate((d1, d2), axis=0)
    np.savez(file_out, **data_mix)

def average_nn(states_from, states_to, step_sizes, num_from=10, num_to=100):

    states_from = np.reshape(states_from, [-1, 3])
    states_to = np.reshape(states_to, [-1, 3])

    idx_from = np.random.choice(len(states_from), num_from)
    idx_to = np.random.choice(len(states_to), num_to)

    sum = 0.0
    for i in range(3):
        diff = states_from[idx_from, None, i] - states_to[None, idx_to, i]
        if i == 2:
            diff = wrap_angle(diff)
        sum += (diff / step_sizes[i])**2
    average_dist = np.mean(np.min(sum, axis=1) > 0.5)
    return average_dist

def load_data(data_path='../data/100s', filename='nav01_train', steps_per_episode=100, num_episodes=None):

    # data = dict(np.load(os.path.join(data_path, '100s', filename + '.npz')))
    data = dict(np.load(os.path.join(data_path, filename + '.npz')))

    # reshape data
    for key in data.keys():
        # 'vel': (100, 1000, 3), 'rgbd': (100, 1000, 32, 32, 4), 'pose': (100, 1000, 3)
        if num_episodes is not None:
            data[key] = data[key][:num_episodes*steps_per_episode]
        data[key] = np.reshape(data[key], [-1, steps_per_episode] + list(data[key].shape[1:])).astype('float32')

    # convert degrees into gradients and
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


def compute_staticstics(data):

    means = dict()
    stds = dict()
    state_step_sizes = []
    state_mins = []
    state_maxs = []

    for key in 'osa':
        # compute means
        means[key] = np.mean(data[key], axis=(0, 1), keepdims=True)
        if key == 's':
            means[key][:, :, 2] = 0  # don't touch orientation because we'll feed this into cos/sin functions
        if key == 'a':
            means[key][:, :, :] = 0  # don't change means of velocities, 0.0, positive and negative values have semantics

        # compute stds
        axis = tuple(range(len(data[key].shape) - 1))  # compute std by averaging over all but the last dimension
        stds[key] = np.std(data[key] - means[key], axis=axis, keepdims=True)
        if key == 's':
            stds[key][:, :, :2] = np.mean(stds[key][:, :, :2])  # scale x and by by the same amount
        if key == 'a':
            stds[key][:, :, :2] = np.mean(stds[key][:, :, :2])  # scale x and by by the same amount

    # compute average step size in x, y, and theta for the distance metric
    for i in range(3):
        steps = np.reshape(data['s'][:, 1:, i] - data['s'][:, :-1, i], [-1])
        if i == 2:
            steps = wrap_angle(steps)
        state_step_sizes.append(np.mean(abs(steps)))
    state_step_sizes[0] = state_step_sizes[1] = (state_step_sizes[0] + state_step_sizes[1]) / 2
    state_step_sizes = np.array(state_step_sizes)

    # compute min and max in x, y and theta
    for i in range(3):
        state_mins.append(np.min(data['s'][:, :, i]))
        state_maxs.append(np.max(data['s'][:, :, i]))
    state_mins = np.array(state_mins)
    state_maxs = np.array(state_maxs)

    return means, stds, state_step_sizes, state_mins, state_maxs


def split_data(data, ratio=0.8, categories=['train', 'val']):
    print('SPLIT {}'.format(data['s'].shape))
    split_data = {categories[0]: dict(), categories[1]: dict()}
    for key in data.keys():
        split_point = int(data[key].shape[0] * ratio)
        split_data[categories[0]][key] = data[key][:split_point]
        split_data[categories[1]][key] = data[key][split_point:]
    for key in split_data.keys():
        print('SPLIT --> {}: {}'.format(key, len(split_data[key]['s'])))
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
    new_o = np.zeros([data['o'].shape[0], data['o'].shape[1], 24, 24, 3])
    for i in range(data['o'].shape[0]):
        for j in range(data['o'].shape[1]):
            if img_random_shift:
                offsets = np.random.random_integers(0, 8, 2)
            else:
                offsets = (4, 4)
            new_o[i, j] = data['o'][i, j, offsets[0]:offsets[0]+24, offsets[1]:offsets[1]+24, :]
    new_o += np.random.normal(0.0, 20 * img_noise_factor, new_o.shape)
    # for i in range(data['o'].shape[0]):
    #     for j in range(data['o'].shape[1]):
    #         plt.figure()
    #         plt.imshow(new_o[i,j]/255, interpolation='nearest')
    #         plt.figure()
    #         plt.imshow(data['o'][i,j]/255, interpolation='nearest')
    #         plt.show()
    new_data['o'] = new_o
    return new_data


def make_batch_iterator(data, batch_size=32, seq_len=10):
    # go through data and select a subsequence from each sequence
    while True:
        episodes = np.random.random_integers(0, len(data['s']) - 1, size=batch_size)
        start_steps = np.random.random_integers(0, len(data['s'][0]) - seq_len - 1, size=batch_size)
        batches = {k: np.concatenate([data[k][i:i + 1, j:j + seq_len] for i, j in zip(episodes, start_steps)]) for k in data.keys()}
        yield batches

def make_repeating_batch_iterator(data, epoch_len, batch_size=32, seq_len=10):
    # go through data and select a subsequence from each sequence
    repeating_episodes = np.random.random_integers(0, len(data['s']) - 1, size=[epoch_len, batch_size])
    repeating_start_steps = np.random.random_integers(0, len(data['s'][0]) - seq_len - 1, size=[epoch_len, batch_size])
    while True:
        for episodes, start_steps in zip(repeating_episodes, repeating_start_steps):
            batches = {k: np.concatenate([data[k][i:i + 1, j:j + seq_len] for i, j in zip(episodes, start_steps)]) for k in data.keys()}
            yield batches

def make_complete_batch_iterator(data, batch_size=1000, seq_len=10):
    num_episodes = len(data['s'])
    num_start_steps = len(data['s'][0]) - seq_len
    batch_indices = [(i, j) for i in range(num_episodes) for j in range(num_start_steps)]
    while batch_indices != []:
        batches = {k: np.concatenate([data[k][i:i + 1, j:j + seq_len] for (i, j) in batch_indices[:batch_size]]) for k in data.keys}
        batch_indices = batch_indices[batch_size:]
        yield batches


def compare_data_coverage():

    task = 'nav02'

    data = load_data(filename=task + '_train', data_path='../data/100s_mix', steps_per_episode=100, num_episodes=100)
    means, stds, state_step_sizes, state_mins, state_maxs = compute_staticstics(data)
    states = dict()
    states['ab'] = data['s']
    data = load_data(filename=task + '_train', data_path='../data/100s_astar', steps_per_episode=100, num_episodes=100)
    states['b'] = data['s']
    data = load_data(filename=task + '_train', data_path='../data/100s', steps_per_episode=100, num_episodes=100)
    states['a'] = data['s']
    # plt.figure()
    # h, b = np.histogram(states['a'][:,:,2], bins=100)
    # plt.plot(b[1:], h)
    # h, b = np.histogram(states['b'][:,:,2], bins=100)
    # plt.plot(b[1:], h)
    # plt.show()
    for f in ['a', 'b']:
        for t in ['a', 'b', 'ab']:
            d = average_nn(states_from=states[f], states_to=states[t], step_sizes=state_step_sizes, num_from=10000, num_to=10000)
            print('{} <- {}: {}'.format(f, t, d))
            plt.pause(0.01)

if __name__ == '__main__':

    # mix_data('../data/100s/nav02_test.npz',
    #          '../data/100s_astar/nav02_test.npz',
    #          '../data/100s_mix/nav02_test')
    #
    # compare_data_coverage()

    task = 'nav03'

    # data = load_data(filename=task + '_train')
    data = load_data(filename=task + '_train', data_path='../data/100s', steps_per_episode=100, num_episodes=1000)
    # data = noisyfy_data(data)

    data = split_data(data, ratio=0.5)
    # means, stds, state_step_sizes, state_mins, state_maxs = compute_staticstics(data)

    # batch_iterator = make_batch_iterator(data['train'])
    scaling = 0.5  # 0.5
    if task == 'nav01':
        plt.figure(figsize=[10*scaling,5*scaling])
    elif task == 'nav02':
        plt.figure(figsize=[15*scaling,9*scaling])
    elif task == 'nav03':
        plt.figure(figsize=[20*scaling,13*scaling])
    # plot_trajectories(noisy_data, emphasize=2, mincolor=0.3)

    # np.random.seed(11)
    # nav02: i=108
    i = 108
    # for i in range(100, 120):
    np.random.seed(i)
    dat = shuffle_data(data['train'])
    dat = reduce_data(dat, 1)
    dat = noisyfy_data(dat)

    plot_trajectory(dat, figure_name=None, emphasize=0, mincolor=0.0, linewidth=0.5)
    plot_maze(task)
    # plot_trajectories(data['val'], figure_name='2', emphasize=None, mincolor=0.0, linewidth=0.5)
    # plot_maze(task)
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')

    plt.tight_layout()
    # plt.savefig("../plots/"+task +".png",
    #            bbox_inches='tight',
    #            transparent=False,
    #            pad_inches=0,
    #            dpi=200)
    plt.savefig("../plots/"+task +".pdf",
               bbox_inches='tight',
               transparent=False,
               pad_inches=0)

    plt.figure()
    # plot_observations(data)
    # plt.savefig("../plots/"+ task +"_obs.png",
    #            bbox_inches='tight',
    #            transparent=False,
    #            pad_inches=0,
    #            dpi=200)
    plot_observations(dat, n=5)
    plt.savefig("../plots/"+ task +"_noisy_obs.pdf",
               bbox_inches='tight',
               transparent=False,
               pad_inches=0,
               dpi=200)

    plt.show()
