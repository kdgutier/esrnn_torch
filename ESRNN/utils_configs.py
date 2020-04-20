def get_config(dataset_name):
    """
    Returns dict config

    Parameters
    ----------
    dataset_name: str
    """
    allowed_dataset_names = ('Yearly', 'Monthly', 'Weekly', 'Hourly', 'Quarterly', 'Daily')
    if dataset_name not in allowed_dataset_names:
        raise ValueError(f'kind must be one of {allowed_kinds}')

    if dataset_name == 'Yearly':
        return YEARLY
    elif dataset_name == 'Monthly':
        return MONTHLY
    elif dataset_name == 'Weekly':
        return WEEKLY
    elif dataset_name == 'Hourly':
        return HOURLY
    elif dataset_name == 'Quarterly':
        return QUARTERLY
    elif dataset_name == 'Daily':
        return DAILY

YEARLY = {
    'device': 'cuda',
    'train_parameters': {
        'max_epochs': 25,
        'batch_size': 4,
        'freq_of_test': 5,
        'learning_rate': '1e-4',
        'lr_scheduler_step_size': 10,
        'lr_decay': 0.1,
        'per_series_lr_multip': 0.8,
        'gradient_clipping_threshold': 50,
        'rnn_weight_decay': 0,
        'noise_std': 0.001,
        'level_variability_penalty': 100,
        'testing_percentile': 50,
        'training_percentile': 50,
        'ensemble': False
    },
    'data_parameters': {
        'max_periods': 25,
        'seasonality': [],
        'input_size': 4,
        'output_size': 6,
        'frequency': 'Y'
    },
    'model_parameters': {
        'cell_type': 'LSTM',
        'state_hsize': 40,
        'dilations': [[1], [6]],
        'add_nl_layer': False,
        'random_seed': 117982
    }
}

MONTHLY = {
    'device': 'cuda',
    'train_parameters': {
        'max_epochs': 15,
        'batch_size': 64,
        'freq_of_test': 4,
        'learning_rate': '7e-4',
        'lr_scheduler_step_size': 12,
        'lr_decay': 0.2,
        'per_series_lr_multip': 0.5,
        'gradient_clipping_threshold': 20,
        'rnn_weight_decay': 0,
        'noise_std': 0.001,
        'level_variability_penalty': 50,
        'testing_percentile': 50,
        'training_percentile': 45,
        'ensemble': False
    },
    'data_parameters': {
        'max_periods': 36,
        'seasonality': [12],
        'input_size': 12,
        'output_size': 18,
        'frequency': 'M'
    },
    'model_parameters': {
        'cell_type': 'LSTM',
        'state_hsize': 50,
        'dilations': [[1, 3, 6, 12]],
        'add_nl_layer': False,
        'random_seed': 1
    }
}


WEEKLY = {
    'device': 'cuda',
    'train_parameters': {
        'max_epochs': 50,
        'batch_size': 32,
        'freq_of_test': 10,
        'learning_rate': '1e-2',
        'lr_scheduler_step_size': 10,
        'lr_decay': 0.5,
        'per_series_lr_multip': 1.0,
        'gradient_clipping_threshold': 20,
        'rnn_weight_decay': 0,
        'noise_std': 0.001,
        'level_variability_penalty': 100,
        'testing_percentile': 50,
        'training_percentile': 50,
        'ensemble': True
    },
    'data_parameters': {
        'max_periods': 31,
        'seasonality': [],
        'input_size': 10,
        'output_size': 13,
        'frequency': 'W'
    },
    'model_parameters': {
        'cell_type': 'ResLSTM',
        'state_hsize': 40,
        'dilations': [[1, 52]],
        'add_nl_layer': False,
        'random_seed': 2
    }
}

HOURLY = {
    'device': 'cuda',
    'train_parameters': {
        'max_epochs': 20,
        'batch_size': 32,
        'freq_of_test': 5,
        'learning_rate': '1e-2',
        'lr_scheduler_step_size': 7,
        'lr_decay': 0.5,
        'per_series_lr_multip': 1.0,
        'gradient_clipping_threshold': 50,
        'rnn_weight_decay': 0,
        'noise_std': 0.001,
        'level_variability_penalty': 30,
        'testing_percentile': 50,
        'training_percentile': 50,
        'ensemble': True
    },
    'data_parameters': {
        'max_periods': 371,
        'seasonality': [24, 168],
        'input_size': 24,
        'output_size': 48,
        'frequency': 'H'
    },
    'model_parameters': {
        'cell_type': 'LSTM',
        'state_hsize': 40,
        'dilations': [[1, 4, 24, 168]],
        'add_nl_layer': False,
        'random_seed': 1
    }
}

QUARTERLY = {
    'device': 'cuda',
    'train_parameters': {
        'max_epochs': 30,
        'batch_size': 16,
        'freq_of_test': 5,
        'learning_rate': '5e-4',
        'lr_scheduler_step_size': 10,
        'lr_decay': 0.5,
        'per_series_lr_multip': 1.0,
        'gradient_clipping_threshold': 20,
        'rnn_weight_decay': 0,
        'noise_std': 0.001,
        'level_variability_penalty': 100,
        'testing_percentile': 50,
        'training_percentile': 50,
        'ensemble': False
    },
    'data_parameters': {
        'max_periods': 20,
        'seasonality': [4],
         'input_size': 4,
         'output_size': 8,
         'frequency': 'Q'
    },
    'model_parameters': {
        'cell_type': 'LSTM',
        'state_hsize': 40,
        'dilations': [[1, 2, 4, 8]],
        'add_nl_layer': False,
        'random_seed': 3
    }
}

DAILY = {
    'device': 'cuda',
    'train_parameters': {
        'max_epochs': 20,
        'batch_size': 64,
        'freq_of_test': 2,
        'learning_rate': '1e-2',
        'lr_scheduler_step_size': 4,
        'lr_decay': 0.3333,
        'per_series_lr_multip': 0.5,
        'gradient_clipping_threshold': 50,
        'rnn_weight_decay': 0,
        'noise_std': 0.0001,
        'level_variability_penalty': 100,
        'testing_percentile': 50,
        'training_percentile': 65,
        'ensemble': False
    },
    'data_parameters': {
        'max_periods': 15,
        'seasonality': [7],
        'input_size': 7,
        'output_size': 14,
        'frequency': 'D'
    },
    'model_parameters': {
        'n_models': 5,
        'n_top': 4,
        'cell_type':
        'LSTM',
        'state_hsize': 40,
        'dilations': [[1, 7, 28]],
        'add_nl_layer': True,
        'random_seed': 1
    }
}
