import tensorflow as tf
import numpy as np
import pickle
import os
import time
import itertools

from utils.exp_utils import get_default_hyperparams, add_to_log, exp_variables_to_name, print_msg_and_dict, sample_exp_variables
from utils.data_utils import load_data, noisify_data_condition, compute_staticstics, make_batch_iterator, reduce_data, shuffle_data
from utils.method_utils import compute_sq_distance
from methods.dpf import DPF
from methods.rnn import RNN
from methods.odom import OdometryBaseline

def meta_exp(base_path, id_extra):

    min_counts = []
    exp_names = ['lc', 'tr', 'nt', 'ab', 'pl', 'mx']
    funcs = [learning_curve_exp, tracking_exp, noise_test_exp, ablation_test_exp, planner_agent_exp, mix_agent_exp]

    for exp_name, f in zip(exp_names, funcs):

        get_experiment_params, get_train_data_and_eval_iterator = f(base_path, run=False)
        # check progress for that experiment
        log_base_path = os.path.join(base_path, 'log', exp_name)
        min_counts.append(get_experiment_params(log_base_path)[-1])
        print('Experiment', exp_name, 'has min_count', min_counts[-1])

    min_min_count = np.min(min_counts)
    sample_list = []
    for i in range(len(exp_names)):
        sample_list += [i] * max(0, (min_min_count + 2) - min_counts[i]) * (3 if 'lc' in exp_names[i] else 1)
    if sample_list == []:
        sample_list = range(len(exp_names))

    i = sample_list[np.random.choice(len(sample_list))]
    exp_name = exp_names[i]
    f = funcs[i]

    print('--> META EXPERIMENT CHOOSES ', exp_name)
    f(base_path, exp_name, id_extra)


def learning_curve_exp(base_path='', exp_name='lc', id_extra='', tracking=False,
                       tasks=('nav01', 'nav02', 'nav03'),
                       methods=('pf_ind', 'pf_e2e', 'pf_ind_e2e', 'lstm', 'ff'),
                       episodes = (16, 32, 64, 125, 250, 500, 1000), data_dir='100s',
                       num_test_episodes=20000, run=True):

    def get_experiment_params(base_path):

        variables, min_count = sample_exp_variables(base_path, [tasks, methods, episodes])
        task, method, num_episodes = variables

        exp_params = {
            'exp': exp_name,
            'task': task,
            'method': method,
            'num_episodes': num_episodes,
            'noise_condition': 'odom10_imgTG',
            'tracking': tracking,
            'computer': os.uname()[1],
            'num_test_episodes': num_test_episodes,
            'eval_batch_size': 16,
            'eval_seq_len': 50,
            'data_dir': data_dir,
            'file_ending': exp_variables_to_name(variables)
        }

        # match sequence length to task
        if exp_params['task'] == 'nav01':
            exp_params['seq_len'] = 20
        elif exp_params['task'] == 'nav02':
            exp_params['seq_len'] = 20
        elif exp_params['task'] == 'nav03':
            exp_params['seq_len'] = 30

        return exp_params, get_default_hyperparams(), min_count

    def get_train_data_and_eval_iterator(data, exp_params):

        # noisify
        for k in ['train', 'test']:
            data[k] = noisify_data_condition(data[k], exp_params['noise_condition'])

        # form batches
        eval_batch_iterators = {k: make_batch_iterator(data[k], batch_size=exp_params['eval_batch_size'], seq_len=exp_params['eval_seq_len']) for k in ['test']}

        return data['train'], eval_batch_iterators

    if run:
        # run an experiment with these two functions
        return run_experiment(get_experiment_params, get_train_data_and_eval_iterator, base_path, exp_name, id_extra)
    else:
        return get_experiment_params, get_train_data_and_eval_iterator


def tracking_exp(base_path='', exp_name='tr', id_extra='',
                       tasks=('nav02',),
                       methods=('pf_ind', 'pf_e2e', 'pf_ind_e2e', 'lstm', 'odom'),
                       episodes = (16, 32, 64, 125, 250, 500, 1000), data_dir='100s', num_test_episodes=20000, run=True):

    return learning_curve_exp(base_path, exp_name, id_extra, True, tasks, methods, episodes, data_dir, num_test_episodes, run)

def planner_agent_exp(base_path='', exp_name='pl', id_extra='',
                       tasks=('nav02',),
                       methods=('pf_ind', 'pf_e2e', 'pf_ind_e2e', 'lstm'),
                       episodes = (16, 32, 64, 125, 250, 500, 1000), data_dir='100s_astar', num_test_episodes=20000, run=True):

    return learning_curve_exp(base_path, exp_name, id_extra, False, tasks, methods, episodes, data_dir, num_test_episodes, run)


def mix_agent_exp(base_path='', exp_name='mx', id_extra='',
                       tasks=('nav02',),
                       methods=('pf_ind', 'pf_e2e', 'pf_ind_e2e', 'lstm'),
                       episodes = (16, 32, 64, 125, 250, 500, 1000), data_dir='100s_mix', num_test_episodes=1000, run=True):

    return learning_curve_exp(base_path, exp_name, id_extra, False, tasks, methods, episodes, data_dir, num_test_episodes, run)


def noise_test_exp(base_path='', exp_name='nt', id_extra='', tracking=False,
                   tasks=('nav02',),
                   methods=('pf_ind', 'pf_e2e', 'pf_ind_e2e', 'lstm'),
                   episodes = (16, 125, 1000),
                   noise_conds=('odom0_imgTG', 'odom5_imgTG', 'odom10_imgTG', 'odom20_imgTG', 'odomX_imgTG',
                                  'odom10_imgC', 'odom10_imgG', 'odom10_imgT', 'odom10_imgX'),
                   data_dir='100s', num_test_episodes=1000,
                   run=True,
                   ):

    def get_experiment_params(base_path):

        variables, min_count = sample_exp_variables(base_path, [tasks, methods, episodes, noise_conds])
        task, method, num_episodes, noise_cond = variables

        exp_params = {
            'exp': exp_name,
            'task': task,
            'method': method,
            'num_episodes': num_episodes,
            'noise_condition': noise_cond,
            'tracking': tracking,
            'computer': os.uname()[1],
            'num_test_episodes': num_test_episodes,
            'eval_batch_size': 16,
            'eval_seq_len': 50,
            'data_dir': data_dir,
            'file_ending': exp_variables_to_name(variables),
            'seq_len': 20,
        }

        return exp_params, get_default_hyperparams(), min_count

    def get_train_data_and_eval_iterator(data, exp_params):

        # noisify training data according to sampled noise condition
        data['train'] = noisify_data_condition(data['train'], exp_params['noise_condition'])

        # create eval batch iterators for every noise condition
        eval_batch_iterators = dict()
        for condition in noise_conds:
            key = 'test' + '_' + condition
            data[key] = noisify_data_condition(data['test'], condition)
            eval_batch_iterators[key] = make_batch_iterator(data[key], batch_size=exp_params['eval_batch_size'], seq_len=exp_params['eval_seq_len'])

        return data['train'], eval_batch_iterators

    if run:
        # run an experiment with these two functions
        return run_experiment(get_experiment_params, get_train_data_and_eval_iterator, base_path, exp_name, id_extra)
    else:
        return get_experiment_params, get_train_data_and_eval_iterator


def ablation_test_exp(base_path='', exp_name='ab', id_extra='', tracking=False,
                   tasks=('nav02',),
                   methods=('pf_ind', 'pf_e2e', 'pf_ind_e2e'),
                   episodes=(16, 125, 1000),
                   ab_conds=('full', 'learn_odom', 'no_proposer', 'no_inject'),
                   data_dir='100s',
                   run=True
                   ):
    def get_experiment_params(base_path):
        variables, min_count = sample_exp_variables(base_path, [tasks, methods, episodes, ab_conds])
        task, method, num_episodes, ab_cond = variables

        exp_params = {
            'exp': exp_name,
            'task': task,
            'method': method,
            'num_episodes': num_episodes,
            'noise_condition': 'odom10_imgTG',
            'tracking': tracking,
            'computer': os.uname()[1],
            'num_test_episodes': 20000,
            'eval_batch_size': 16,
            'eval_seq_len': 50,
            'data_dir': data_dir,
            'file_ending': exp_variables_to_name(variables),
            'seq_len': 20,
        }

        hyper_params = get_default_hyperparams()
        if ab_cond == 'learn_odom':
            hyper_params['global']['learn_odom'] = True
        elif ab_cond == 'no_proposer':
            hyper_params['global']['use_proposer'] = False
            hyper_params['global']['propose_ratio'] = 0.0
        elif ab_cond == 'no_inject':
            hyper_params['global']['propose_ratio'] = 0.0

        return exp_params, hyper_params, min_count

    def get_train_data_and_eval_iterator(data, exp_params):

        # noisify
        for k in ['train', 'test']:
            data[k] = noisify_data_condition(data[k], exp_params['noise_condition'])

        # form batches
        eval_batch_iterators = {k: make_batch_iterator(data[k], batch_size=exp_params['eval_batch_size'], seq_len=exp_params['eval_seq_len']) for k in ['test']}

        return data['train'], eval_batch_iterators

    if run:
        # run an experiment with these two functions
        return run_experiment(get_experiment_params, get_train_data_and_eval_iterator, base_path, exp_name, id_extra)
    else:
        return get_experiment_params, get_train_data_and_eval_iterator


def run_experiment(get_experiment_params, get_train_data_and_eval_iterator, base_path, exp_name, id_extra='',
                   load_from_model_path=None, load_modules=None):

    # construct base paths
    log_base_path = os.path.join(base_path, 'log', exp_name)
    if not os.path.exists(log_base_path):
        os.makedirs(log_base_path)
    model_base_path = os.path.join(base_path, 'models', exp_name)

    # sample experiment parameters by checking the log for what is most urgent right now
    exp_params, hyperparams, min_count = get_experiment_params(log_base_path)
    data_path = os.path.join(base_path, 'data', exp_params['data_dir'])

    id = exp_params['id'] = time.strftime('%Y-%m-%d_%H:%M:%S_') + exp_params['computer'] + str(id_extra) + '_' + exp_params['file_ending']
    log_path = os.path.join(log_base_path, id)
    model_path = exp_params['model_path'] = os.path.join(model_base_path, id)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # load data
    data = {k: load_data(data_path=data_path, filename=exp_params['task'] + '_' + k) for k in ['train', 'test']}
    means, stds, state_step_sizes, state_mins, state_maxs = compute_staticstics(data['train'])

    data['train'] = shuffle_data(data['train'])
    data['train'] = reduce_data(data['train'], exp_params['num_episodes'])

    data['train'], eval_batch_iterators = get_train_data_and_eval_iterator(data, exp_params)

    log = dict()

    # SET THINGS UP
    tf.reset_default_graph()

    print_msg_and_dict('STARTING EXPERIMENT', exp_params)

    hyperparams['global']['init_with_true_state'] = exp_params['tracking']

    if 'pf' in exp_params['method']:
        method = DPF(**hyperparams['global'])
        hyperparams['train']['train_e2e'] = 'e2e' in exp_params['method']
        hyperparams['train']['train_individually'] = 'ind' in exp_params['method']

    elif 'lstm' in exp_params['method']:
        method = RNN(**hyperparams['global'])
    elif 'ff' in exp_params['method']:
        method = RNN(model='ff', **hyperparams)
    elif 'odom' in exp_params['method']:
        method = OdometryBaseline(**hyperparams)
    else:
        print('I DONT KNOW THIS METHOD', exp_params['method'])

    with tf.Session() as session:

        t0 = time.time()
        if load_from_model_path is None:
            training_log = method.fit(session, data['train'], model_path, **hyperparams['train'])
        elif type(load_from_model_path) == type([]):
            for i, (path, modules) in enumerate(zip(load_from_model_path, load_modules)):
                print('Loading %s from %s' % (modules, path))
                method.load(session, path, modules=modules, connect_and_initialize=(i==0))
            training_log = None
        else:
            print('Loading model')
            if load_modules is None:
                method.load(session, load_from_model_path)
            else:
                method.load(session, load_from_model_path, modules=load_modules)
            training_log = None

        t1 = time.time()
        add_to_log(log, {'training_duration': t1 - t0})

        print_msg_and_dict('RESULTS after {}s'.format(log['training_duration'][-1]), exp_params)

        for k in sorted(eval_batch_iterators.keys()):
            results = {'mse': []}
            result_hist = dict()
            for i in range(0, exp_params['eval_seq_len'], 10):
                result_hist[i] = np.zeros(100)

            for eval_batch in eval_batch_iterators[k]:

                predicted_states = method.predict(session, eval_batch, **hyperparams['test'])
                squared_errors = compute_sq_distance(predicted_states, eval_batch['s'], state_step_sizes)

                for i in result_hist.keys():
                    result_hist[i] += np.histogram(squared_errors[:, i], bins=100, range=[0.0, 10.0])[0]
                results['mse'].append(np.mean(squared_errors, axis=0))
                if len(results['mse']) * exp_params['eval_batch_size'] >= exp_params['num_test_episodes']:
                   break

            for i in result_hist.keys():
                result_hist[i] /= len(results['mse']) * exp_params['eval_batch_size']
            mse = np.stack(results['mse'], axis=0)

            add_to_log(log, {k + '_hist': result_hist,
                            k + '_mse': np.mean(mse, axis=0),
                            k + '_mse_se':  np.std(mse, ddof=1, axis=0) / np.sqrt(len(mse))})
            for i in range(0, len(log[k+'_mse'][-1]), 5):
                print('{:>10} step {} !! mse: {:.4f}+-{:.4f}'.format(k, i, log[k+'_mse'][-1][i], log[k+'_mse_se'][-1][i]))

    add_to_log(log, {'hyper_params': hyperparams})
    add_to_log(log, {'exp_params': exp_params})
    add_to_log(log, {'training': training_log})

    # save result
    print('Saved log as ', log_path)
    with open(log_path, 'wb') as f:  # Just use 'w' mode in 3.x
        pickle.dump(log, f)
