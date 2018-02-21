import pickle
import os
import numpy as np
from experiments.distributed_experiments import run_experiment, tracking_exp, planner_agent_exp, learning_curve_exp, noise_test_exp

def cross(logfile, cross_exp, exp_name='cr'):
    # load data, choose correct task, method, num_episodes, noise_cond, seq_len
    with open(logfile, 'rb') as f:
        log = pickle.load(f)
    model_path = '../models/' + log['exp_params'][0]['model_path'].split('/models/')[-1] # ['exp_params']['model_path]
    print(model_path)

    # these are actually already lists so we can pass them on directly
    task = [log['exp_params'][0]['task']]
    method = [log['exp_params'][0]['method']]
    num_episodes = [log['exp_params'][0]['num_episodes']]
    num_episodes = [log['exp_params'][0]['num_episodes']]

    # define experiment you want to run
    get_experiment_params, get_train_data_and_eval_iterator = cross_exp('../', exp_name=exp_name, id_extra='',
                                                                           tasks=task, methods=method, episodes=num_episodes,
                                                                           num_test_episodes=1000,
                                                                           run=False)

    run_experiment(get_experiment_params, get_train_data_and_eval_iterator, base_path='../', exp_name=exp_name, id_extra='', load_from_model_path=model_path)


def swapmodels(logfiles, noise_conds, exp_name='swap', flipmodules=False):

    # expect logfiles to be a dict with two keys that match the noise conditions in noise_test,
    # e.g. {'odom5_imgTG': [log1, log2], 'odom20_imgTG': [log1, log2]}
    # noise_conds should be a list of the two conditions

    # noise_conds = logfiles.keys()
    model_paths = dict()
    for c in noise_conds:
        model_paths[c] = []
        for i, logfile in enumerate(logfiles[c]):
            with open(logfile, 'rb') as f:
                log = pickle.load(f)
                model_paths[c].append('../models/' + log['exp_params'][0]['model_path'].split('/models/')[-1])
                # should be the same for all logfiles, not checked here
                task = [log['exp_params'][0]['task']]
                method = [log['exp_params'][0]['method']]
                num_episodes = [log['exp_params'][0]['num_episodes']]



    get_experiment_params, get_train_data_and_eval_iterator = noise_test_exp('../', exp_name=exp_name, id_extra='',
                                                                           tasks=task, methods=method, episodes=num_episodes,
                                                                           noise_conds=noise_conds,
                                                                           num_test_episodes=1000,
                                                                           run=False)

    modules0 = ('mo_noise_generator', 'mo_transition_model')
    modules1 = ('encoder', 'obs_like_estimator', 'particle_proposer')

    if flipmodules:
        modules0, modules1 = modules1, modules0

    for variant, (path, module) in {
        'orig_'+noise_conds[0]: (model_paths[noise_conds[0]][0], None),
        '%s_%s' % (noise_conds[0], noise_conds[0]): (model_paths[noise_conds[0]], [modules0, modules1]),
        '%s_%s' % (noise_conds[0], noise_conds[1]): ([model_paths[noise_conds[0]][0], model_paths[noise_conds[1]][0]], [modules0, modules1]),
        'orig_'+noise_conds[1]: (model_paths[noise_conds[1]][0], None),
        '%s_%s' % (noise_conds[1], noise_conds[1]): (model_paths[noise_conds[1]], [modules0, modules1]),
        '%s_%s' % (noise_conds[1], noise_conds[0]): ([model_paths[noise_conds[1]][0], model_paths[noise_conds[0]][0]], [modules0, modules1]),
        }.items():
        print('!!! %s %s %s' % (variant, path, module))
        run_experiment(get_experiment_params, get_train_data_and_eval_iterator, base_path='../', exp_name=exp_name+'/'+variant, id_extra='', load_from_model_path=path, load_modules=module)


def get_all_logs(path, file_ending):
    return [os.path.join(path, filename) for filename in os.listdir(path)
              if os.path.isfile(os.path.join(path, filename))
              # and filename.endswith(file_ending)]
              and file_ending in filename]

def cross_lc2pl(method):
    # for f in get_all_logs('../log/lc', 'nav02_'+method+'_1000'):
    for f in get_all_logs('../log/lc', 'nav02_'+method+'_'):
        cross(f, learning_curve_exp, 'lc2lc1')
        cross(f, planner_agent_exp, 'lc2pl1')

def cross_pl2lc(method):
    # for f in get_all_logs('../log/pl', 'nav02_'+method+'_1000'):
    for f in get_all_logs('../log/pl', 'nav02_'+method):
        cross(f, learning_curve_exp, 'pl2lc1')
        cross(f, planner_agent_exp, 'pl2pl1')

def cross_mx(method):
    for f in get_all_logs('../log/mx', 'nav02_'+method+'_1000'):
    # for f in get_all_logs('../log/mx', 'nav02_'+method):
        cross(f, learning_curve_exp, 'mx2lc')
        cross(f, planner_agent_exp, 'mx2pl')

def swap_motion(method):
    noise_conds = ['odom5_imgTG', 'odom10_imgTG']
    logs = dict()
    for c in noise_conds:
        logs[c] = [f for f in get_all_logs('../log/nt', 'nav02_'+method+'_1000_'+c)]
        i, j = np.random.choice(len(logs[c]), 2, False)
        logs[c] = [logs[c][i], logs[c][j]]
    swapmodels(logs, noise_conds, 'swapmo')

def swap_measurement(method):
    noise_conds = ['odom10_imgG', 'odom10_imgTG']
    logs = dict()
    for c in noise_conds:
        logs[c] = [f for f in get_all_logs('../log/nt', 'nav02_'+method+'_1000_'+c)][:2]
        i, j = np.random.choice(len(logs[c]), 2, False)
        logs[c] = [logs[c][i], logs[c][j]]
    swapmodels(logs, noise_conds, 'swapme', flipmodules=True)

# if __name__ == '__main__':
