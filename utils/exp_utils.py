import os
import itertools
import numpy as np

def get_default_hyperparams():

    # default hyperparams
    return {
        'global': {
            'init_with_true_state': False,
            'learn_odom': False,
            'use_proposer': True,
            'propose_ratio': 0.7,
            'proposer_keep_ratio': 0.15,
            'min_obs_likelihood': 0.004,
        },
        'train': {
            'train_individually': True,
            'train_e2e': True,
            'split_ratio': 0.9,
            'seq_len': 20,
            'batch_size': 32,
            'epoch_length': 50,
            'num_epochs': 10000,
            'patience': 200,
            'learning_rate': 0.0003,
            'dropout_keep_ratio': 0.3,
            'num_particles': 100,
            'particle_std': 0.2,
        },
        'test' : {
            'num_particles': 1000,
        }
    }

def exp_variables_to_name(x):
    return '_'.join(map(str, x))


def sample_exp_variables(path, exp_variables):

    # compute all combinations of the experiment variables
    product = list(itertools.product(*exp_variables))
    n = len(product)
    # turn them into filename endings
    file_endings = list(map(exp_variables_to_name, product))
    # count how often each ending appears, i.e. how often each experiment has been run
    counts = [0] * n
    try:
        for filename in os.listdir(path):
            if os.path.isfile(os.path.join(path, filename)):
                for i in range(n):
                    if filename.endswith(file_endings[i]):
                        counts[i] += 1
    except FileNotFoundError:
        pass

    # compute a sample list with samples according to which experimental variables need more examples
    min_count = np.min(counts)
    sample_list = []
    for i in range(n):
        sample_list += [product[i]] * max(0, (min_count + 2) - counts[i])
    if sample_list == []:
        sample_list = product

    # sample from this list
    print('sampling from:', sample_list)
    sample = sample_list[np.random.choice(len(sample_list))]
    print('--> ', sample)
    return sample, min_count


def print_msg_and_dict(msg, d):
    keys = sorted(list(d.keys()))
    msg += ' '
    for k in keys:
        msg += '\n{}: {}'.format(k, d[k])
    print('########################################################')
    print(msg)
    print('########################################################')


def add_to_log(log, d):
    for k in d.keys():
        if k not in log.keys():
            log[k] = []
        log[k].append(d[k])
    return log
