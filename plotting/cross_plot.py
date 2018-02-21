import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import os

results = None

# matplotlib.rcParams.update({'font.size': 12})

color_list = plt.cm.tab10(np.linspace(0, 1, 10))
colors = {'lstm': color_list[0], 'pf_e2e': color_list[1], 'pf_ind_e2e': color_list[2], 'pf_ind': color_list[3]}
labels = {'lstm': 'LSTM', 'pf_e2e': 'DPF (e2e)', 'pf_ind_e2e': 'DPF (ind+e2e)', 'pf_ind': 'DPF (ind)', 'ff': 'FF', 'odom': 'Odom. baseline'}
# conditions = ['normal', 'no_motion_likelihood', 'learn_odom', 'no_proposer']
# conditions = ['normal', 'learn_odom', 'no_inject']
# clabels = {'normal': 'Default', 'no_motion_likelihood': 'W/o motion likelihood', 'learn_odom': 'Learned odometry', 'no_proposer': 'W/o particle proposer', 'no_inject': "No inject"}
conditions = ['lc2lc', 'pl2lc', 'mx2lc', 'lc2pl', 'pl2pl', 'mx2pl']
clabels = {'lc2lc':'lc2lc', 'lc2pl':'lc2pl', 'pl2lc':'pl2lc', 'pl2pl':'pl2pl', 'mx2lc': 'mx2lc', 'mx2pl': 'mx2pl'}
task = 'nav02'
methods = ['pf_ind', 'pf_e2e', 'pf_ind_e2e', 'lstm']
# methods = ['pf_ind_e2e', 'lstm']

# load results
results = dict()

count = 0
for cond in conditions:
    # log_path = '/home/rbo/Desktop/log/'+task+'_ab1'
    log_path = '../log/'+cond
    for filename in [f for f in os.listdir(log_path) if os.path.isfile(os.path.join(log_path, f))]:
        full_filename = os.path.join(log_path, filename)
        print('loading {}:'.format(count) + full_filename + ' ...')
        try:
            # if 'DeepThought' not in filename:
            # if 'DeepThought' in filename:
            with open(full_filename, 'rb') as f:
                result = pickle.load(f)
                # result_name = result['task'][0] + '/' + result['method'][0] + '/' + str(result['num_episodes'][0]) + '/' + result['condition'][0]
                result_name = cond + '_' + result['exp_params'][0]['file_ending'] #result['exp_params'][0]['task'] + '/' + result['exp_params'][0]['method'] + '/' + str(result['exp_params'][0]['num_episodes']) + '/' + result['exp_params'][0]['ab_cond']
                print(result_name)
                if result_name not in results.keys():
                    results[result_name] = result
                else:
                    for key in result.keys():
                        if key in results[result_name].keys():
                            results[result_name][key] += result[key]
                        else:
                            results[result_name][key] = result[key]
                        # print(result_name, key)
                count += 1
        except Exception as e:
            print(e)
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

print()
for result_name in results.keys():
    print(result_name, len(results[result_name]['test_mse']))

print('Loaded {} results'.format(count))

# print(results['test_errors'].shape, np.mean(results['test_errors']**2, axis=1))

#print('SHAPE', results['test_mse'].shape)

# plt.figure(1)
# plt.gca().set_color_cycle(None)
# for method in set(results['method']):

task = 'nav02'
# step = 30
step = 3

episodes = [1000]
# episodes = [1000]
fig_names = []

max_1 = 0
max_2 = {n: 0 for n in episodes}

means = dict()
ses = dict()

for num_episodes in episodes:

    means[num_episodes] = dict()
    ses[num_episodes] = dict()

    for method in methods:

        means[num_episodes][method] = np.zeros([len(conditions), 5])
        # means[num_episodes][method] = np.zeros([len(conditions), 50])
        ses[num_episodes][method] = np.zeros([len(conditions), 5])
        # ses[num_episodes][method] = np.zeros([len(conditions), 50])

        for c, condition in enumerate(conditions):

            result_name = condition + '_' + task + '_' + method + '_' + str(num_episodes)
            if result_name in results.keys():
                result = results[result_name]

                # means[num_episodes][method][c] = np.mean(result['test_mse'], axis=0)
                # std = np.std(result['test_mse'], axis=0, ddof=1)
                # ses[num_episodes][method][c] = std / np.sqrt(len(result['test_mse']))

                hist = np.array([[h[i] for i in range(0, 50, 10)] for h in result['test_hist']])  # result x time x sqe [.0, 0.1, .., 10.0]
                err = 1. - np.sum(hist[:,:,:10], axis=-1) # sqe < 1.0
                # err = np.sum(hist[:,:,:10], axis=-1) # sqe < 1.0
                print(result_name, err)
                means[num_episodes][method][c] = np.mean(err, axis=0)
                ses[num_episodes][method][c] = np.std(err, axis=0, ddof=1) / np.sqrt(len(err))
                print(means[num_episodes][method][c])

            else:
                # print(result_name, 0)
                means[num_episodes][method][c] *= np.nan
                ses[num_episodes][method][c] *= np.nan



    means[num_episodes]['min'] = np.stack([means[num_episodes][method] for method in methods], axis=0).min(axis=1)

    fig_name = 'ab1_{}'.format(num_episodes)
    fig = plt.figure(fig_name, [6, 3.5])
    fig_names.append(fig_name)
    ax = fig.add_subplot(111)
    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    for c, condition in enumerate(conditions):
        sax = fig.add_subplot(2, 3, c+1)
        for m, method in enumerate(methods):
            sax.bar(0.0  - 0.5 + (m+1)/len(methods)*0.8,
                            means[num_episodes][method][c, step],
                            0.8/len(methods),
                            yerr=ses[num_episodes][method][c, step],
                            color=colors[method], label=labels[method])

            text = '{:.3s}'.format('{:.2f}'.format(means[num_episodes][method][c, step])[1:])
            plt.text(0.0  - 0.5 + (m+1)/len(methods)*0.8, means[num_episodes][method][c, step] + ses[num_episodes][method][c, step] + 0.05, text, va='center', ha='center', color=colors[method], fontweight='normal')


        # sax.set_ylim([0.0, 1.05])
        sax.set_ylim([0.0, 1.0])
        sax.set_xticks([])
        sax.set_yticks([])
        # if c % 2 == 0:
        # if c >= 2:
        if 'lc2' in condition:
            xlabel = 'A'
            sax.set_ylabel(('A' if '2lc' in condition else 'B'), fontweight = 'bold')
        elif 'pl2' in condition:
            xlabel = 'B'
        elif 'mx2' in condition:
            xlabel = 'A+B'
        if '2pl' in condition:
            sax.set_xlabel(xlabel, fontweight = 'bold')
        if c == 0:
            plt.legend()

    ax.set_xlabel('Trained with policy')
    ax.set_ylabel('Error rate in test with policy\n')

    plt.tight_layout(h_pad=0.0, w_pad=0.0, pad=0.0)
    plt.savefig('../plots/cr/policy.pdf', bbox_inches="tight", transparent=True, dpi=600, frameon=True, facecolor='w', pad_inches=0.01)
    plt.show()

