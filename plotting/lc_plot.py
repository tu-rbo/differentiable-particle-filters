import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import itertools
from collections import namedtuple
import os

results = None

# matplotlib.rcParams.update({'font.size': 12})

color_list = plt.cm.tab10(np.linspace(0, 1, 10))
colors = {'lstm': color_list[0], 'pf_e2e': color_list[1], 'pf_ind_e2e': color_list[2], 'pf_ind': color_list[3], 'ff': color_list[4], 'odom': color_list[4]}
labels = {'lstm': 'LSTM', 'pf_e2e': 'DPF (e2e)', 'pf_ind_e2e': 'DPF (ind+e2e)', 'pf_ind': 'DPF (ind)', 'ff': 'FF', 'odom': 'Odom. baseline'}

def load_results(base_path='../log/', exp='lc'):
    results = dict()

    count = 0
    log_path = os.path.join(base_path, exp)
    listdir = os.listdir(log_path)
    for i, filename in enumerate(listdir):
        full_filename = os.path.join(log_path, filename)
        # if 'DeepThought' not in full_filename:
        print('loading ' + full_filename + ' ...')
        # try:
        with open(full_filename, 'rb') as f:
            result = pickle.load(f)
            print(result['exp_params'][0].keys())
            result_name = result['exp_params'][0]['task'] + '/' + result['exp_params'][0]['method'] + '/' + str(result['exp_params'][0]['num_episodes'])
            if result_name not in results.keys():
                results[result_name] = result
            else:
                for key in result.keys():
                    results[result_name][key] += result[key]
            count += 1
        # except Exception as e:
        #     print(e)

    for task in tasks:
        for method in methods:
            for num_episodes in episodes:
                result_name = task + '/' + method + '/' + str(num_episodes)
                if result_name in results.keys():
                    print(result_name, len(results[result_name]['exp_params']))
                else:
                    print(result_name, 0)

    print('Loaded {} results'.format(count))
    return results

# print(results['test_errors'].shape, np.mean(results['test_errors']**2, axis=1))

#print('SHAPE', results['test_mse'].shape)

# plt.figure(1)
# plt.gca().set_color_cycle(None)
# for method in set(results['method']):

# step = {'nav01': 20, 'nav02': 20, 'nav03': 20}

# COMPUTE STATISTICS
def compute_statistics(results):
    sqe_means = dict()
    sqe_ses = dict()
    acc_means = dict()
    acc_ses = dict()
    for task in tasks:

        sqe_means[task] = dict()
        sqe_ses[task] = dict()
        acc_means[task] = dict()
        acc_ses[task] = dict()

        for method in methods:
            sqe_means[task][method] = []
            sqe_ses[task][method] = []
            acc_means[task][method] = []
            acc_ses[task][method] = []
            # hist[task][method] = dict()
            # hist_ses[task][method] = dict()
            for num_episodes in episodes:
                result_name = task + '/' + method + '/' + str(num_episodes)
                if result_name in results.keys():
                    result = results[result_name]
                    # hist[task][method][num_episodes] = np.mean([h[step[task]] for h in result['test_hist']], axis=0)
                    # hist_ses[task][method][num_episodes] = np.std([h[step[task]] for h in result['test_hist']], axis=0, ddof=1) / np.sqrt(len(result['test_hist']))
                    sqe_means[task][method].append(np.mean(result['test_mse'], axis=0))
                    sqe_ses[task][method].append(np.std(result['test_mse'], axis=0, ddof=1) / np.sqrt(len(result['test_mse'])))

                    # hist = np.array([h[step[task]] for h in result['test_hist']])  # result x time x sqe [.0, 0.1]
                    hist = np.array([[h[i] for i in range(0, 50, 10)] for h in result['test_hist']])  # result x time x sqe [.0, 0.1, .., 10.0]
                    acc = 1. - np.sum(hist[:,:,:10], axis=-1) # sqe < 1.0
                    acc_means[task][method].append(np.mean(acc, axis=0))
                    acc_ses[task][method].append(np.std(acc, axis=0, ddof=1) / np.sqrt(len(acc)))
                else:
                    sqe_means[task][method].append([np.nan]*max_steps)
                    sqe_ses[task][method].append([np.nan]*max_steps)
                    print(num_episodes)
                    acc_means[task][method].append([np.nan] * (max_steps // 10))
                    acc_ses[task][method].append([np.nan] * (max_steps // 10))

            sqe_means[task][method] = np.array(sqe_means[task][method])
            sqe_ses[task][method] = np.nan_to_num(sqe_ses[task][method])
            acc_means[task][method] = np.array(acc_means[task][method])
            acc_ses[task][method] = np.nan_to_num(acc_ses[task][method])

    return sqe_means, sqe_ses, acc_means, acc_ses


def plot_learning_curve(means, ses, step, f=lambda x:x, ylabel_func=lambda x: '', ylim_func=None, show_legend=None, divide_by=None, save_extra=''):

    for task in tasks:

        plt.figure('lc'+ save_extra + ' for ' + task, [4,2.5])
        # plt.plot([125, 125], [0, 1000], ':', color='gray', linewidth=1)
        # plt.plot([1000, 1000], [0, 1000], ':', color='gray', linewidth=1)

        for method in methods:

            # valid = np.isnan(means[:,step-1]) == False
            # eps = np.array(episodes)

            if divide_by is None:
                plt.fill_between(episodes, (f(means[task][method])-np.array(ses[task][method]))[:,step[task]], (f(means[task][method])+np.array(ses[task][method]))[:,step[task]], color=colors[method], alpha=0.3, linewidth=0.0)
            else:
                plt.fill_between(episodes,
                                 (f(means[task][method])-np.array(ses[task][method]))[:,step[task]] / f(means[task][divide_by][:, step[task]]),
                                 (f(means[task][method])+np.array(ses[task][method]))[:,step[task]] / f(means[task][divide_by][:, step[task]]),
                                 color=colors[method], alpha=0.3, linewidth=0.0)
            # plt.plot(125, means[task][method][episodes.index(125), step[task]], 'o', color=colors[method], markersize=3, linewidth=1)
            # plt.plot(1000, means[task][method][episodes.index(1000), step[task]], 'x', color=colors[method], markersize=4, linewidth=1)

        for method in methods:
            if divide_by is None:
                plt.plot(episodes, f(means[task][method][:, step[task]]), '.-' if method != 'odom' else '--', color=colors[method], label=labels[method], markersize=2, linewidth=1)
            else:
                plt.plot(episodes, f(means[task][method][:, step[task]]) / f(means[task][divide_by][:, step[task]]), '.-' if method != 'odom' else '--', color=colors[method], label=labels[method], markersize=2, linewidth=1)



        plt.gca().set_xscale("log", nonposx='clip')
        if ylim_func is not None:
            plt.ylim(ylim_func(task))
        # plt.ylim([0, max_1])
        # plt.ylim([0, 1.0])
        plt.xticks(episodes)
        plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.gca().get_xaxis().set_tick_params(which='minor', size=0)
        plt.gca().get_xaxis().set_tick_params(which='minor', width=0)
        plt.xlabel('Training episodes (log. scale)')
        plt.ylabel(ylabel_func(step[task]))
        # plt.tight_layout()
        if show_legend is None or show_legend[task]:
            plt.legend(loc='upper right')

        # plt.figure(task + " " + str(step) + " steps")
        extra = 'lc' + save_extra
        plt.savefig('../plots/' + exp + '/'+exp+'_'+task+'_'+extra+'.pdf', bbox_inches="tight", transparent=True, dpi=600, frameon=True, facecolor='w', pad_inches=0.01)
        plt.savefig('../plots/' + exp + '/'+exp+'_'+task+'_'+extra+'.png', bbox_inches="tight", transparent=True, dpi=600, frameon=True, facecolor='w', pad_inches=0.01)


# PLOT FILTER CONVERGENCE
def plot_filter_convergence(means, ses, step, ylabel_func, ylim_func=None, save_extra=''):

    for task in tasks:
        max_2 = {n: 0 for n in episodes}

        for num_episodes in [1000]:
            i = episodes.index(num_episodes)
            plt.figure(task + " " + str(num_episodes) + " training episodes " + save_extra, [2,2.5])
            for method in methods:
                n = means[task][method].shape[1]
                # if num_episodes == 125:
                #     plt.plot([step[task]], means[task][method][i, step[task]], 'o', color=colors[method], markersize=3, linewidth=1) # label=labels[method]
                # elif num_episodes == 1000:
                #     plt.plot([step[task]], means[task][method][i, step[task]], 'x', color=colors[method], markersize=4, linewidth=1) # label=labels[method]
                plt.fill_between(np.arange(n), (np.array(means[task][method])-np.array(ses[task][method]))[i,:], (np.array(means[task][method])+np.array(ses[task][method]))[i,:], color=colors[method], alpha=0.3, linewidth=0.0)
                if method is not 'pf_ind':
                    max_2[num_episodes] = max(means[task][method][i, -1], max_2[num_episodes])
            for method in methods:
                n = means[task][method].shape[1]
                # plt.plot(np.arange(1, 20+1), means[task][method][i, :20], '--', color=colors[method], markersize=3, linewidth=1)
                # plt.plot(np.arange(20, max_steps+1), means[task][method][i, 19:], '-', color=colors[method], markersize=3, linewidth=1) # label=labels[method]
                plt.plot(means[task][method][i, :], '-' if method != 'odom' else '--', color=colors[method], markersize=3, linewidth=1) # label=labels[method]

            # plt.plot([step], [0], 'w', label=' ', linewidth=0)
            # plt.plot([step[task]], [0], '--', color='gray', label='Steps optimized during training', linewidth=1)
            # plt.plot([step[task], step[task]], [0, 5*max_2[num_episodes]], ':', color='gray', label='Step tested in learning curve', linewidth=1)
            if ylim_func is not None:
                plt.ylim(ylim_func(task))
            plt.xticks([0, 20, 40])
            plt.xlabel('Tested at step')
            plt.ylabel(ylabel_func(step[task]))
            plt.ylabel('MSE ({} episodes)'.format(num_episodes))
            plt.tight_layout()
            # if task == 'nav01':
            #     plt.legend()

            # plt.figure(task + " " + str(num_episodes) + " training episodes")
            extra = 'convrg' + save_extra
            plt.savefig('../plots/'+exp+'/'+exp+'_'+task+'_steps_'+str(num_episodes)+'_'+extra+'.pdf', bbox_inches="tight", transparent=True, dpi=600, frameon=True, facecolor='w', pad_inches=0.01)
            plt.savefig('../plots/'+exp+'/'+exp+'_'+task+'_steps_'+str(num_episodes)+'_'+extra+'.png', bbox_inches="tight", transparent=True, dpi=600, frameon=True, facecolor='w', pad_inches=0.01)


methods = ['lstm', 'pf_ind', 'pf_e2e', 'pf_ind_e2e']
episodes = [16, 32, 64, 125, 250, 500, 1000]
exp = 'lc'; tasks = ['nav01', 'nav02', 'nav03']; max_steps = 50
# exp = 'pl'; tasks = ['nav02']; max_steps = 50; #methods = ['lstm', 'pf_ind_e2e']
# exp = 'mx'; tasks = ['nav02']; max_steps = 50; #methods = ['lstm', 'pf_ind_e2e']

# exp = 'tr'; tasks = ['nav02']; methods = ['lstm', 'pf_ind', 'pf_e2e', 'pf_ind_e2e', 'odom']; max_steps = 50

plot_path = '../plots/' + exp
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

results = load_results(exp=exp)
sqe_means, sqe_ses, acc_means, acc_ses = compute_statistics(results)

# print(acc_means['nav01']['lstm'].shape)

if exp == 'lc':

    plot_learning_curve(sqe_means, sqe_ses, step = {'nav01': 20, 'nav02': 20, 'nav03': 30},
                        ylabel_func=lambda step: 'MSE (at step {})'.format(step),
                        ylim_func=lambda task: {'nav01':[0,25], 'nav02':[0,55], 'nav03':[0,110]}[task],
                        save_extra='_mse', show_legend={'nav01': True, 'nav02': False, 'nav03': False})

    plot_learning_curve(acc_means, acc_ses, f=lambda x: x, step = {'nav01': 2, 'nav02': 2, 'nav03': 3},
                        ylabel_func=lambda step: 'Error rate',
                        ylim_func=lambda task: [0.0,1.0],
                        save_extra='_er', show_legend={'nav01': True, 'nav02': False, 'nav03': False})

    plot_learning_curve(sqe_means, sqe_ses, step = {'nav01': 20, 'nav02': 20, 'nav03': 30},
                        ylabel_func=lambda step: 'MSE relative to LSTM',
                        ylim_func=lambda task: [0.0, 1.2],
                        save_extra='_mse_div', show_legend={'nav01': False, 'nav02': False, 'nav03': False}, divide_by='lstm')

    plot_learning_curve(acc_means, acc_ses, f=lambda x: x, step = {'nav01': 2, 'nav02': 2, 'nav03': 3},
                        ylabel_func=lambda step: 'Error rate relative to LSTM',
                        ylim_func=lambda task: [0.0, 1.2],
                        save_extra='_er_div', show_legend={'nav01': False, 'nav02': False, 'nav03': False}, divide_by='lstm')

    # plot_filter_convergence(sqe_means, sqe_ses, step = {'nav01': 40, 'nav02': 40, 'nav03': 40},
    #                     ylabel_func=lambda step: 'Test MSE (at step {})'.format(step),
    #                     ylim_func=lambda task: {'nav01':[0,25], 'nav02':[0,55], 'nav03':[0,110]}[task],
    #                     save_extra='_mse')
    #
    # plot_filter_convergence(acc_means, acc_ses, step = {'nav01': 2, 'nav02': 2, 'nav03': 3},
    #                     ylabel_func=lambda step: 'Test Accuracy (at step {})'.format(step*10),
    #                     ylim_func=lambda task: [0.0,1.0],
    #                     save_extra='_acc')

    # plot_filter_convergence(sqe_means, sqe_ses, step = {'nav01': 40, 'nav02': 40, 'nav03': 40},
    #                     ylabel_func=lambda step: 'Test MSE (at step {})'.format(step),
    #                     ylim_func=lambda task: {'nav01':[0,25], 'nav02':[0,1.0], 'nav03':[0,110]}[task],
    #                     save_extra='_mse')

elif exp == 'tr':

    plot_learning_curve(sqe_means, sqe_ses, step = {'nav01': 20, 'nav02': 20, 'nav03': 30},
                        ylabel_func=lambda step: 'MSE (at step {})'.format(step),
                        ylim_func=lambda task: {'nav01':[0,25], 'nav02':[0,55], 'nav03':[0,110]}[task],
                        save_extra='_mse', show_legend={'nav01': True, 'nav02': True, 'nav03': False})

    plot_learning_curve(acc_means, acc_ses, f=lambda x: x, step = {'nav01': 2, 'nav02': 2, 'nav03': 3},
                        ylabel_func=lambda step: 'Error rate',
                        ylim_func=lambda task: [0.0,1.0],
                        save_extra='_er', show_legend={'nav01': True, 'nav02': True, 'nav03': False})
    #
    # plot_learning_curve(sqe_means, sqe_ses, step = {'nav01': 20, 'nav02': 20, 'nav03': 30},
    #                     ylabel_func=lambda step: 'MSE relative to LSTM',
    #                     ylim_func=lambda task: [0.0, 1.2],
    #                     save_extra='_mse_div', show_legend={'nav01': False, 'nav02': False, 'nav03': False}, divide_by='lstm')
    #
    # plot_learning_curve(acc_means, acc_ses, f=lambda x: x, step = {'nav01': 2, 'nav02': 2, 'nav03': 3},
    #                     ylabel_func=lambda step: 'Error rate relative to LSTM',
    #                     ylim_func=lambda task: [0.0, 1.2],
    #                     save_extra='_er_div', show_legend={'nav01': False, 'nav02': False, 'nav03': False}, divide_by='lstm')


    plot_filter_convergence(sqe_means, sqe_ses, step = {'nav01': 40, 'nav02': 40, 'nav03': 40},
                        ylabel_func=lambda step: 'MSE (at step {})'.format(step),
                        ylim_func=lambda task: {'nav01':[0,25], 'nav02':[0,1.0], 'nav03':[0,110]}[task],
                        save_extra='_mse')
    #
    # plot_filter_convergence(acc_means, acc_ses, step = {'nav01': 2, 'nav02': 2, 'nav03': 3},
    #                     ylabel_func=lambda step: 'Test Accuracy (at step {})'.format(step*10),
    #                     ylim_func=lambda task: [0.0,1.0],
    #                     save_extra='_acc')

else:

    plot_learning_curve(sqe_means, sqe_ses, step = {'nav01': 20, 'nav02': 20, 'nav03': 30},
                        ylabel_func=lambda step: 'MSE (at step {})'.format(step),
                        ylim_func=lambda task: {'nav01':[0,25], 'nav02':[0,55], 'nav03':[0,110]}[task],
                        save_extra='_mse', show_legend={'nav01': True, 'nav02': False, 'nav03': False})

    plot_learning_curve(acc_means, acc_ses, f=lambda x: x, step = {'nav01': 2, 'nav02': 2, 'nav03': 3},
                        ylabel_func=lambda step: 'Error rate',
                        ylim_func=lambda task: [0.0,1.0],
                        save_extra='_er', show_legend={'nav01': True, 'nav02': False, 'nav03': False})

    plot_learning_curve(sqe_means, sqe_ses, step = {'nav01': 20, 'nav02': 20, 'nav03': 30},
                        ylabel_func=lambda step: 'MSE relative to LSTM',
                        ylim_func=lambda task: [0.0, 1.2],
                        save_extra='_mse_div', show_legend={'nav01': False, 'nav02': False, 'nav03': False}, divide_by='lstm')

    plot_learning_curve(acc_means, acc_ses, f=lambda x: x, step = {'nav01': 2, 'nav02': 2, 'nav03': 3},
                        ylabel_func=lambda step: 'Error rate relative to LSTM',
                        ylim_func=lambda task: [0.0, 1.2],
                        save_extra='_er_div', show_legend={'nav01': False, 'nav02': False, 'nav03': False}, divide_by='lstm')

    # plot_filter_convergence(sqe_means, sqe_ses, step = {'nav01': 40, 'nav02': 40, 'nav03': 40},
    #                     ylabel_func=lambda step: 'Test MSE (at step {})'.format(step),
    #                     ylim_func=lambda task: {'nav01':[0,25], 'nav02':[0,55], 'nav03':[0,110]}[task],
    #                     save_extra='_mse')
    #
    # plot_filter_convergence(acc_means, acc_ses, step = {'nav01': 2, 'nav02': 2, 'nav03': 3},
    #                     ylabel_func=lambda step: 'Test Accuracy (at step {})'.format(step*10),
    #                     ylim_func=lambda task: [0.0,1.0],
    #                     save_extra='_acc')

    # plot_filter_convergence(sqe_means, sqe_ses, step = {'nav01': 40, 'nav02': 40, 'nav03': 40},
    #                     ylabel_func=lambda step: 'Test MSE (at step {})'.format(step),
    #                     ylim_func=lambda task: {'nav01':[0,25], 'nav02':[0,1.0], 'nav03':[0,110]}[task],
    #                     save_extra='_mse')

plt.show()


