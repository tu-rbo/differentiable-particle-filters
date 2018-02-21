import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

results = None



# matplotlib.rcParams.update({'font.size': 12})

color_list = plt.cm.tab10(np.linspace(0, 1, 10))
colors = {'lstm': color_list[0], 'pf_e2e': color_list[1], 'pf_ind_e2e': color_list[2], 'pf_ind': color_list[3]}
labels = {'lstm': 'LSTM', 'pf_e2e': 'DPF (e2e)', 'pf_ind_e2e': 'DPF (ind+e2e)', 'pf_ind': 'DPF (ind)'}
tasks = ['nav02']
methods = ['lstm', 'pf_ind', 'pf_e2e', 'pf_ind_e2e']

# load results
results = dict()

count = 0
for task in tasks:
    log_path = '../log/nt'
    for filename in [f for f in os.listdir(log_path) if os.path.isfile(os.path.join(log_path, f))]:
        full_filename = os.path.join(log_path, filename)
        print('loading {}:'.format(count) + full_filename + ' ...')
        try:
            # if 'DeepThought' not in filename:
            # if 'DeepThought' in filename:
            with open(full_filename, 'rb') as f:
                result = pickle.load(f)
                result_name = result['exp_params'][0]['task'] + '/' + result['exp_params'][0]['method'] + '/' + str(result['exp_params'][0]['num_episodes']) + '/' + result['exp_params'][0]['noise_condition']
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
            print('%r' % e)
            # raise e
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

for result_name in results.keys():
    print(result_name, len(results[result_name]['exp_params']))

print('Loaded {} results'.format(count))

# print(results['test_errors'].shape, np.mean(results['test_errors']**2, axis=1))

#print('SHAPE', results['test_mse'].shape)

# plt.figure(1)
# plt.gca().set_color_cycle(None)
# for method in set(results['method']):

task = 'nav02'
metric = 'err'; step = 2 # err or mse
# metric = 'mse'; step = 20 # err or mse

# episodes = [125, 1000]
episodes = [1000]
fig_names = []

for noise_in in ['odom', 'image']:

    if noise_in == 'odom':
        conditions = ['odom0_imgTG', 'odom5_imgTG', 'odom10_imgTG', 'odom20_imgTG', 'odomX_imgTG']; clabels = {'odom0_imgTG': '0', 'odom5_imgTG': '5', 'odom10_imgTG': '10', 'odom20_imgTG': '20', 'odomX_imgTG': 'X'}
        # conditions = ['odom0_imgTG', 'odom5_imgTG', 'odom10_imgTG', 'odomX_imgTG']; clabels = {'odom0_imgTG': '0', 'odom5_imgTG': '5', 'odom10_imgTG': '10', 'odomX_imgTG': 'X'}
    else:
        conditions = ['odom10_imgC', 'odom10_imgG', 'odom10_imgT', 'odom10_imgTG', 'odom10_imgX']; clabels = {'odom10_imgC': 'N', 'odom10_imgG': 'G', 'odom10_imgT': 'S', 'odom10_imgTG': 'G+S', 'odom10_imgX': 'X'}

    max_1 = 0
    max_2 = {n: 0 for n in episodes}

    means = dict()
    ses = dict()

    for num_episodes in episodes:

        means[num_episodes] = dict()
        ses[num_episodes] = dict()

        for method in methods:

            if metric == 'mse':
                means[num_episodes][method] = np.zeros([len(conditions), len(conditions), 50])
                ses[num_episodes][method] = np.zeros([len(conditions), len(conditions), 50])
            elif metric == 'err':
                means[num_episodes][method] = np.zeros([len(conditions), len(conditions), 5])
                ses[num_episodes][method] = np.zeros([len(conditions), len(conditions), 5])

            for c, condition in enumerate(conditions):

                result_name = task + '/' + method + '/' + str(num_episodes) + '/' + condition
                if result_name in results.keys():
                    result = results[result_name]
                    for ct, test_condition in enumerate(conditions):
                        if 'test_'+test_condition+'_mse' not in result.keys():
                            means[num_episodes][method][c, ct] *= np.nan
                            ses[num_episodes][method][c, ct] *= np.nan
                        else:
                            if noise_in != "odom" and num_episodes == 1000 and c == 1 and ct == 1:
                                print(method,
                                    np.array(result['test_'+test_condition+'_mse'])[:, 30])

                            if metric == 'mse':
                                means[num_episodes][method][c, ct] = np.mean(result['test_'+test_condition+'_mse'], axis=0)
                                std = np.std(result['test_'+test_condition+'_mse'], axis=0, ddof=1)
                                ses[num_episodes][method][c, ct] = std / np.sqrt(len(result['test_'+test_condition+'_mse']))
                            elif metric == 'err':
                                hist = np.array([[h[i] for i in range(0, 50, 10)] for h in result['test_'+test_condition+'_hist']])  # result x time x sqe [.0, 0.1, .., 10.0]
                                err = 1. - np.sum(hist[:,:,:10], axis=-1) # sqe < 1.0
                                means[num_episodes][method][c, ct] = np.mean(err, axis=0)
                                ses[num_episodes][method][c, ct] = np.std(err, axis=0, ddof=1) / np.sqrt(len(err))

                else:
                    # print(result_name, 0)
                    for test_condition in conditions:
                        means[num_episodes][method][c, :] *= np.nan


        # if noise_in != 'odom':
        #     print(means[1000]['pf_ind'][1,1,30])

        means[num_episodes]['min'] = np.stack([means[num_episodes][method] for method in methods], axis=0).min(axis=0)

        for method in methods:

            fig_name = 'nt_{}_{}_{}'.format(noise_in,num_episodes,method)
            plt.figure(fig_name, [3,2.5])
            fig_names.append(fig_name)

            min_val, max_val, diff = 0., len(conditions), 1.

            #imshow portion
            N_points = (max_val - min_val) / diff
            m = means[num_episodes][method][:,:,step].T
            is_min = m == means[num_episodes]['min'][:,:,step].T

            # plt.imshow((means[:,:,30-1])**0.5, interpolation='nearest', vmin=0, vmax=15)
            if metric == 'mse':
                plt.imshow(np.log(m), interpolation='nearest', vmin=-3, vmax=6, cmap='viridis')
            elif metric == 'err':
                plt.imshow(m, interpolation='nearest', vmin=-0.33, vmax=1.0, cmap='viridis_r')
            # data = np.reshape(np.arange(len(conditions)*len(conditions)), [len(conditions), len(conditions)])
            # plt.imshow(data, interpolation='nearest', vmin=0, vmax=10)
            plt.xticks(np.arange(len(conditions)), [clabels[c] for c in conditions])
            plt.yticks(np.arange(len(conditions)), [clabels[c] for c in conditions])

            plt.ylabel('Test {}{} noise'.format(noise_in, '.' if noise_in == 'odom' else ''))
            plt.xlabel('Training {}{} noise'.format(noise_in, '.' if noise_in == 'odom' else ''))

            # plt.colorbar()
            #text portion
            ind_array = np.arange(min_val, max_val, diff)
            x, y = np.meshgrid(ind_array, ind_array)

            for x_val, y_val in zip(x.flatten(), y.flatten()):
                value = m[int(y_val),int(x_val)]
                if metric == 'err':
                    text = '{:.4s}'.format('{:.3f}'.format(value)[1:])
                else:
                    text = '{:.4s}'.format('{:.2f}'.format(value))
                if x_val == y_val:
                    if x_val == 3 and noise_in == 'image' or x_val == 2 and noise_in == 'odom':
                        style = '-w'
                    else:
                        style = '--w'
                    plt.plot(np.array([x_val, x_val+1, x_val+1, x_val, x_val])-0.5, np.array([y_val, y_val, y_val+1, y_val+1, y_val])-0.5, style, linewidth=1.5)
                # if value > 0.9:
                #     color = 'black'
                # else:
                #     color = 'white'
                plt.text(x_val, y_val, text, va='center', ha='center', color='white', fontweight='bold' if is_min[int(y_val), int(x_val)] else 'normal')

            fig_name = 'nt_diag_{}_{}'.format(noise_in,num_episodes)
            # plt.figure(fig_name, [3,2.5])
            plt.figure(fig_name, [2,2.5])
            fig_names.append(fig_name)

            if noise_in == 'odom':
                x = np.array([int(clabels[c]) for c in conditions[:-1]])
            else:
                x = np.arange(len(conditions)-1)
            m = means[num_episodes][method][:,:,step]
            s = ses[num_episodes][method][:,:,step]
            plt.plot(x, np.diag(m)[:-1], '.-', color=colors[method], label=labels[method], markersize=2)
            ind = 2 if noise_in == 'odom' else 3
            plt.plot(x[ind], np.diag(m)[ind], 'x', color=colors[method], markersize=4)
            plt.fill_between(x, np.diag(m-s)[:-1], np.diag(m+s)[:-1], color=colors[method], alpha=0.5, linewidth=0.0)

            plt.xticks(x, [clabels[c] for c in conditions[:-1]])
            if noise_in == 'odom':
                plt.xlabel('Odometry noise (%)')
            else:
                plt.xlabel('Image noise')
                plt.legend()

            if metric == 'mse':
                plt.ylabel('MSE ({} episodes)'.format(num_episodes))
            else:
                plt.ylabel('Error rate ({} episodes)'.format(num_episodes))
                plt.ylim([0, 0.3])
                plt.yticks([0.0, 0.1, 0.2, 0.3])

            fig_name = 'nt_shuffle_{}'.format(num_episodes)
            # plt.figure(fig_name, [3,2.5])
            plt.figure(fig_name, [2,2.5])
            fig_names.append(fig_name)

            if noise_in == 'odom':
                plt.bar(0.0  - 0.5 + (methods.index(method)+1)/len(methods)*0.8,
                        np.diag(m)[-3],
                        0.8/len(methods),
                        yerr=np.diag(s)[-3],
                        color=colors[method], label=labels[method])
            plt.bar((1.0 if noise_in == 'odom' else 2.0)  - 0.5 + (methods.index(method)+1)/len(methods)*0.8,
                    np.diag(m)[-1], 0.8/len(methods),
                    yerr=np.diag(s)[-1],
                    color=colors[method])
            if noise_in == 'odom':
                relative = np.diag(m)[-1] / np.diag(m)[-3]
            else:
                relative = np.diag(m)[-1] / np.diag(m)[-2]
            textpos = np.diag(m)[-1] + np.diag(s)[-1] + 2
            if num_episodes == 1000:
                if textpos > 10:
                    textpos = 3
            elif textpos > 80:
                    textpos = 10
            color = 'black'
            if metric == 'err':
                textpos = 0.05
                color = 'white'
            # plt.text((1.0 if noise_in == 'odom' else 2.0)  - 0.5 + (methods.index(method)+1)/len(methods)*0.8,
            #          textpos, '×{:.0f}'.format(relative), va='bottom', ha='center',color=color, rotation=90)
            plt.xticks([0, 1, 2], ['Both', 'Image', 'Odom.'])
            plt.ylim([0,1.05])
            plt.xlabel('Input')
            if metric == 'mse':
                plt.ylabel('MSE ({} episodes)'.format(num_episodes))
            else:
                plt.ylabel('Error rate ({} episodes)'.format(num_episodes))
            # plt.legend()


for fn in fig_names:
    plt.figure(fn)
    # plt.tight_layout()
    plt.savefig('../plots/nt/{}.pdf'.format(fn), bbox_inches="tight", transparent=True, dpi=600, frameon=True, facecolor='w', pad_inches=0.01)
    # plt.savefig('../plots/nt/{}.png'.format(fn), bbox_inches="tight", transparent=True, dpi=600, frameon=True, facecolor='w', pad_inches=0.01)

    # plt.bar(np.arange(len(conditions)) + (methods.index(method)+1)/len(methods)*0.8 - 0.5, np.diag(means[:,:,30-1])/np.diag(means[:,:,30-1])[0], 0.8/len(methods), color=colors[method])


plt.show()
