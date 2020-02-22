class ModelConfig(object):
  def __init__(self, max_epochs, batch_size, batch_size_test, freq_of_test,
               learning_rate, lr_scheduler_step_size,
               per_series_lr_multip, gradient_eps, gradient_clipping_threshold,
               rnn_weight_decay,
               noise_std,
               level_variability_penalty,
               percentile, training_percentile,
               cell_type,
               state_hsize, dilations, add_nl_layer, seasonality, input_size, output_size, 
               frequency, max_periods, random_seed, device, root_dir):

    # Train Parameters
    self.max_epochs = max_epochs
    self.batch_size = batch_size
    self.batch_size_test = batch_size_test
    self.freq_of_test = freq_of_test
    self.learning_rate = learning_rate
    self.lr_scheduler_step_size = lr_scheduler_step_size
    self.per_series_lr_multip = per_series_lr_multip
    self.gradient_eps = gradient_eps
    self.gradient_clipping_threshold = gradient_clipping_threshold
    self.rnn_weight_decay = rnn_weight_decay
    self.noise_std = noise_std
    self.level_variability_penalty = level_variability_penalty
    self.percentile = percentile
    self.training_percentile = training_percentile
    self.device = device

    # Model Parameters
    self.cell_type = cell_type
    self.state_hsize = state_hsize
    self.dilations = dilations
    self.add_nl_layer = add_nl_layer
    self.random_seed = random_seed

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

    #self.numeric_threshold = float(config['train_parameters']['numeric_threshold'])
    #self.attention_hsize = self.state_hsize
    #self.min_inp_seq_length = config['data_parameters']['min_inp_seq_length']
    #self.num_series = config['data_parameters']['num_series']
    #self.output_dir = config['data_parameters']['output_dir']