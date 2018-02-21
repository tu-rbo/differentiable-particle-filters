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
labels = {'lstm': 'LSTM', 'pf_e2e': 'e2e', 'pf_ind_e2e': 'ind+e2e', 'pf_ind': 'ind'}
# conditions = ['normal', 'no_motion_likelihood', 'learn_odom', 'no_proposer']
# conditions = ['normal', 'learn_odom', 'no_inject']
# clabels = {'normal': 'Default', 'no_motion_likelihood': 'W/o motion likelihood', 'learn_odom': 'Learned odometry', 'no_proposer': 'W/o particle proposer', 'no_inject': "No inject"}
conditions = ['full', 'learn_odom', 'no_inject', 'no_proposer']
clabels = {'full': 'Full', 'learn_odom': 'Learned\nodometry', 'no_proposer': 'No particle\nproposer', 'no_inject': "No particle\ninjection"}
tasks = ['nav02']
methods = ['pf_ind', 'pf_e2e', 'pf_ind_e2e']

# load results
results = dict()

count = 0
for task in tasks:
    # log_path = '/home/rbo/Desktop/log/'+task+'_ab1'
    log_path = '../log/ab'
    for filename in [f for f in os.listdir(log_path) if os.path.isfile(os.path.join(log_path, f))]:
        full_filename = os.path.join(log_path, filename)
        print('loading {}:'.format(count) + full_filename + ' ...')
        try:
            # if 'DeepThought' not in filename:
            # if 'DeepThought' in filename:
            with open(full_filename, 'rb') as f:
                result = pickle.load(f)
                # result_name = result['task'][0] + '/' + result['method'][0] + '/' + str(result['num_episodes'][0]) + '/' + result['condition'][0]
                result_name = result['exp_params'][0]['file_ending'] #result['exp_params'][0]['task'] + '/' + result['exp_params'][0]['method'] + '/' + str(result['exp_params'][0]['num_episodes']) + '/' + result['exp_params'][0]['ab_cond']
                for ab_cond in conditions:
                    if result_name.endswith(ab_cond):
                        result['exp_params'][0]['ab_cond'] = ab_cond
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

for result_name in results.keys():
    print(result_name, len(results[result_name]['exp_params'][0]['task']))

print('Loaded {} results'.format(count))

# print(results['test_errors'].shape, np.mean(results['test_errors']**2, axis=1))

#print('SHAPE', results['test_mse'].shape)

# plt.figure(1)
# plt.gca().set_color_cycle(None)
# for method in set(results['method']):

task = 'nav02'
# step = 30
step = 2

episodes = [16, 125, 1000]
# episodes = [1000]
fig_names = []

max_1 = 0
max_2 = {n: 0 for n in episodes}

means = dict()
ses = dict()

fig_name = 'abcolorbar'
plt.figure(fig_name, [0.8,2.5])
fig_names.append(fig_name)

vmax=1.0
a = np.array([[0.0, 1.0]])
img = plt.imshow(a, cmap="viridis_r", vmin=-0.33*vmax, vmax=vmax)
plt.gca().set_visible(False)
cax = plt.axes([0.0, 0.2, 0.1, 0.65])
plt.colorbar(orientation="vertical", cax=cax, label='Error rate', boundaries=np.linspace(0,1.0,100), ticks=np.linspace(0.0, 1.0, 11))

for num_episodes in episodes:

    means[num_episodes] = dict()
    ses[num_episodes] = dict()

    for method in methods:

        means[num_episodes][method] = np.zeros([len(conditions), 5])
        # means[num_episodes][method] = np.zeros([len(conditions), 50])
        ses[num_episodes][method] = np.zeros([len(conditions), 5])
        # ses[num_episodes][method] = np.zeros([len(conditions), 50])

        for c, condition in enumerate(conditions):

            result_name = task + '_' + method + '_' + str(num_episodes) + '_' + condition
            if result_name in results.keys():
                result = results[result_name]

                # means[num_episodes][method][c] = np.mean(result['test_mse'], axis=0)
                # std = np.std(result['test_mse'], axis=0, ddof=1)
                # ses[num_episodes][method][c] = std / np.sqrt(len(result['test_mse']))

                hist = np.array([[h[i] for i in range(0, 50, 10)] for h in result['test_hist']])  # result x time x sqe [.0, 0.1, .., 10.0]
                err = 1. - np.sum(hist[:,:,:10], axis=-1) # sqe < 1.0
                means[num_episodes][method][c] = np.mean(err, axis=0)
                print(result_name)
                print(err[:, step])
                print(np.mean(err, axis=0)[step])
                print(np.std(err, axis=0, ddof=1)[step], np.sqrt(len(err)))
                ses[num_episodes][method][c] = np.std(err, axis=0, ddof=1) / np.sqrt(len(err))

            else:
                # print(result_name, 0)
                means[num_episodes][method][c] *= np.nan
                ses[num_episodes][method][c] *= np.nan


    means[num_episodes]['min'] = np.stack([means[num_episodes][method] for method in methods], axis=0).min(axis=1)

    fig_name = 'ab1_{}'.format(num_episodes)
    plt.figure(fig_name, [3,2.5])
    fig_names.append(fig_name)

    # m = means[num_episodes][method][:,step-1]
    m = np.stack([means[num_episodes][method][:, step] for method in methods], axis=0)
    s = np.stack([ses[num_episodes][method][:, step] for method in methods], axis=0)
    is_min = m == means[num_episodes]['min'][:, None, step]

    # plt.imshow((means[:,:,30-1])**0.5, interpolation='nearest', vmin=0, vmax=15)
    # plt.imshow(np.log(m.T), interpolation='nearest', vmin=-2.5, vmax=2.5, cmap='viridis')
    plt.imshow(m.T, interpolation='nearest', vmin=-0.33, vmax=1.0, cmap='viridis_r')
    # data = np.reshape(np.arange(len(conditions)*len(conditions)), [len(conditions), len(conditions)])
    # plt.imshow(data, interpolation='nearest', vmin=0, vmax=10)
    plt.yticks(np.arange(len(conditions)), [clabels[c] for c in conditions])
    plt.xticks(np.arange(len(methods)), [labels[m] for m in methods])

    # plt.xlabel('Test {}{} noise'.format(noise_in, '.' if noise_in == 'odom' else ''))
    # plt.ylabel('Training {}{} noise'.format(noise_in, '.' if noise_in == 'odom' else ''))

    # plt.colorbar()
    #text portion
    # min_val, max_val, diff = 0., len(conditions), 1.
    # N_points = (max_val - min_val) / diff
    ind_array_y = np.arange(0., len(methods), 1.)
    ind_array_x = np.arange(0., len(conditions), 1.)
    x, y = np.meshgrid(ind_array_x, ind_array_y)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        value = m[int(y_val),int(x_val)]
        s_value = s[int(y_val),int(x_val)]
        text = '{:.4s}\n+-{:.4s}'.format('{:.3f}'.format(value)[1:],'{:.2f}'.format(s_value)[1:])
        plt.text(y_val, x_val, text, va='center', ha='center', color='white', fontweight='bold' if is_min[int(y_val), int(x_val)] else 'normal')
        # plt.text(y_val, x_val, text, va='center', ha='center', color='white', fontweight='normal')

        # fig_name = 'nt_diag_{}'.format(num_episodes)
        # plt.figure(fig_name, [3,2.5])
        # fig_names.append(fig_name)
        #
        # x = np.arange(len(conditions))
        # m = means[num_episodes][method][:,:,step-1]
        # s = ses[num_episodes][method][:,:,step-1]
        # plt.plot(x[:-1], np.diag(m)[:-1], '-', color=colors[method], label=labels[method])
        # ind = -3 if noise_in == 'odom' else -2
        # plt.plot(x[ind], np.diag(m)[ind], 'x', color=colors[method])
        # plt.fill_between(x[:-1], np.diag(m-s)[:-1], np.diag(m+s)[:-1], color=colors[method], alpha=0.5, linewidth=0.0)
        # plt.xticks(np.arange(len(conditions)-1), [clabels[c] for c in conditions[:-1]])
        # if noise_in == 'odom':
        #     plt.xlabel('Gaussian odometry noise in %')
        # else:
        #     plt.xlabel('Image noise')
        #     plt.legend()
        #
        # plt.ylabel('Test MSE ({} episodes)'.format(num_episodes))
        # plt.ylim([0, 2.5])
        #
        # fig_name = 'nt_shuffle_{}'.format(num_episodes)
        # plt.figure(fig_name, [3,2.5])
        # fig_names.append(fig_name)
        #
        # if noise_in == 'odom':
        #     plt.bar(0.0  - 0.5 + (methods.index(method)+1)/len(methods)*0.8,
        #             np.diag(m)[-2],
        #             0.8/len(methods),
        #             yerr=np.diag(s)[-2],
        #             color=colors[method], label=labels[method])
        # plt.bar((1.0 if noise_in == 'odom' else 2.0)  - 0.5 + (methods.index(method)+1)/len(methods)*0.8,
        #         np.diag(m)[-1], 0.8/len(methods),
        #         yerr=np.diag(s)[-1],
        #         color=colors[method])
        # relative = np.diag(m)[-1] / np.diag(m)[-2]
        # textpos = np.diag(m)[-1] + np.diag(s)[-1] + 2
        # if num_episodes == 1000:
        #     if textpos > 10:
        #         textpos = 3
        # elif textpos > 80:
        #         textpos = 10
        # plt.text((1.0 if noise_in == 'odom' else 2.0)  - 0.5 + (methods.index(method)+1)/len(methods)*0.8,
        #          textpos, '×{:.0f}'.format(relative), va='bottom', ha='center',color='black', rotation=90)
        # plt.xticks([0, 1, 2], ['Both', 'Image', 'Odom.'])
        # plt.ylim([0,60])
        # plt.xlabel('Input')
        # plt.ylabel('Test MSE ({} episodes)'.format(num_episodes))
        # # plt.legend()


for fn in fig_names:
    plt.figure(fn)
    plt.tight_layout()
    plt.savefig('../plots/ab/{}.pdf'.format(fn), bbox_inches="tight", transparent=True, dpi=600, frameon=True, facecolor='w', pad_inches=0.01)
    # plt.savefig('../plots/ab/{}.png'.format(fn), bbox_inches="tight", transparent=True, dpi=600, frameon=True, facecolor='w', pad_inches=0.01)

            # plt.bar(np.arange(len(conditions)) + (methods.index(method)+1)/len(methods)*0.8 - 0.5, np.diag(means[:,:,30-1])/np.diag(means[:,:,30-1])[0], 0.8/len(methods), color=colors[method])


plt.show()
