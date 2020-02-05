class ModelConfig(object):
  def __init__(self, max_epochs, batch_size, 
               learning_rate, lr_scheduler_step_size,
               per_series_lr_multip, gradient_eps, gradient_clipping_threshold,
               rnn_weight_decay,
               noise_std,
               level_variability_penalty, tau,
               state_hsize, dilations, add_nl_layer, seasonality, input_size, output_size, 
               frequency, max_periods, device, root_dir):

    # Train Parameters
    self.max_epochs = max_epochs
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.lr_scheduler_step_size = lr_scheduler_step_size
    self.per_series_lr_multip = per_series_lr_multip
    self.gradient_eps = gradient_eps
    self.gradient_clipping_threshold = gradient_clipping_threshold
    self.rnn_weight_decay = rnn_weight_decay
    self.noise_std = noise_std
    self.level_variability_penalty = level_variability_penalty
    self.tau = tau
    self.device = device

    # Model Parameters
    self.state_hsize = state_hsize
    self.dilations = dilations
    self.add_nl_layer = add_nl_layer

    # Data Parameters
    self.seasonality = seasonality
    self.input_size = input_size
    self.input_size_i = self.input_size
    self.output_size = output_size
    self.output_size_i = self.output_size
    self.frequency = frequency
    self.min_series_length = self.input_size_i + self.output_size_i# + self.min_inp_seq_length + 2
    self.max_series_length = (max_periods * self.seasonality) + self.min_series_length
    self.root_dir = root_dir

    #self.dataset_name = config['dataset_name']
    #self.freq_of_test = config['train_parameters']['freq_of_test']
    #self.numeric_threshold = float(config['train_parameters']['numeric_threshold'])
    #self.rnn_weight_decay = config['train_parameters']['rnn_weight_decay']
    #self.percentile = config['train_parameters']['percentile']
    #self.training_percentile = config['train_parameters']['training_percentile']
    #self.training_tau = self.training_percentile / 100.
    #self.lback = config['model_parameters']['lback']
    #self.attention_hsize = self.state_hsize
    #self.exogenous_size = config['data_parameters']['exogenous_size']
    #self.min_inp_seq_length = config['data_parameters']['min_inp_seq_length']
    #self.num_series = config['data_parameters']['num_series']
    #self.output_dir = config['data_parameters']['output_dir']