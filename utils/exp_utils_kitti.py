def get_default_hyperparams():

    # default hyperparams
    return {
        'global': {
            'init_with_true_state': True,
            'learn_odom': False,
            'use_proposer': False,
            'propose_ratio': 0.7,
            'proposer_keep_ratio': 0.15,
            'min_obs_likelihood': 0.004,
            'learn_gaussian_mle': False

        },
        'train': {
            'train_individually': True,
            'train_e2e': True,
            'split_ratio': 0.95,
            'seq_len': 50,
            'batch_size': 32,
            'epoch_length': 50,
            'num_epochs': 10000,
            'patience': 200,
            'learning_rate': 0.0003,
            'dropout_keep_ratio': 0.3,
            'num_particles': 100,
            'particle_std': 2.0,
            'learn_gaussian_mle': False
        },
        'test' : {
            'num_particles': 400
        }
    }