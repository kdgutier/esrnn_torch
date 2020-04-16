#This code can be used to reproduce the forecasts of the M4 Competition STATISTICAL Benchmarks and evaluate their accuracy

library(forecast) #Requires v8.2
library(dplyr)
library(readr)
library(optparse)

#################################################################################
# R Naive2 implementation
#################################################################################

naive_seasonal <- function(input, fh){
  #Used to estimate Seasonal Naive
  frcy <- frequency(input)
  frcst <- naive(input, h=fh)$mean
  if (frcy>1){
    frcst <- head(rep(as.numeric(tail(input,frcy)), fh), fh) + frcst - frcst
  }
  return(frcst)
}

SeasonalityTest <- function(input, ppy){
  #Used to determine whether a time series is seasonal
  tcrit <- 1.645
  if (length(input)<3*ppy){
    test_seasonal <- FALSE
  }else{
    xacf <- acf(input, plot = FALSE)$acf[-1, 1, 1]
    clim <- tcrit/sqrt(length(input)) * sqrt(cumsum(c(1, 2 * xacf^2)))
    test_seasonal <- ( abs(xacf[ppy]) > clim[ppy] )

    if (is.na(test_seasonal)==TRUE){ test_seasonal <- FALSE }
  }

  return(test_seasonal)
}

Naive2Benchmark <- function(input, fh){
  #Used to estimate the statistical benchmarks of the M4 competition

  #Estimate seasonaly adjusted time series
  ppy <- frequency(input) ; ST <- F
  if (ppy>1){ ST <- SeasonalityTest(input,ppy) }
  if (ST==T){
    Dec <- decompose(input,type="multiplicative")
    des_input <- input/Dec$seasonal
    SIout <- head(rep(Dec$seasonal[(length(Dec$seasonal)-ppy+1):length(Dec$seasonal)], fh), fh)
  }else{
    des_input <- input ; SIout <- rep(1, fh)
  }

  f1 <- naive(input, h=fh)$mean #Naive
  f2 <- naive_seasonal(input, fh=fh) #Seasonal Naive
  f3 <- naive(des_input, h=fh)$mean*SIout #Naive2

  return(f3)
}

#################################################################################
# R Naive2 predictions
#################################################################################

seas_dict = list(Hourly=list(seasonality=24, input_size=24, output_size=48, freq='H'),
                 Daily=list(seasonality=7, input_size=7, output_size=14, freq='D'),
                 Weekly=list(seasonality=52, input_size=52, output_size=13, freq='W'),
                 Monthly=list(seasonality=12, input_size=12, output_size=18, freq='M'),
                 Quarterly=list(seasonality=4, input_size=4, output_size=8, freq='Q'),
                 Yearly=list(seasonality=1, input_size=4, output_size=6, freq='Y'))

option_list = list(make_option(c("--dataset_name"), type="character", default=NULL,
                               help="dataset name (Hourly, Daily, Weekly, Monthly, Yearly)",
                               metavar="character"),
                   make_option(c("--directory"), type="character", default=NULL,
                               help="Custom directory where data will be saved."),
                   make_option(c("--num_obs"), type="integer", default=1000000,
                               help="Number of time series to consider"))

opt_parser = OptionParser(option_list=option_list)

readDataset <- function(dataset_name, directory, num_obs){
  train_file <- paste0(directory, '/m4/Train/', dataset_name, '-train.csv')
  test_file <- paste0(directory, '/m4/Test/', dataset_name, '-test.csv')

  data_train <- data.table::fread(train_file, fill=TRUE)
  data_test <- data.table::fread(train_file, fill=TRUE)

  data_train <- head(data_train, num_obs)
  data_test <- head(data_test, num_obs)

  data_train <- arrange(data_train, V1)
  data_test <- arrange(data_test, V1)
  data <- list(data_train=data_train, data_test=data_test)
  return(data)
}

getNaive2Benchmark <- function(opt){
  # Parse arguments
  dataset_name <- opt$dataset_name
  directory <- opt$directory
  num_obs <- opt$num_obs
  fh <- seas_dict[[dataset_name]]$output_size
  frq <- seas_dict[[dataset_name]]$seasonality
  print(paste("dataset_name:", dataset_name))
  print(paste("fh:", fh))
  print(paste("frq:", frq))

  # Read data
  data <- readDataset(dataset_name, directory, num_obs)
  data_train <- data$data_train
  data_test <- data$data_test

  # Naive2 predictions
  Naive2_forecasts <- array(NA, dim = c(nrow(data_train), fh))
  for (i in 1:nrow(data_train)){
    unique_id <- data_train[i, 1]
    train_row <-data_train[i, -1]
    #print(paste(paste("Row:", as.character(i)), paste("Unique_id:", unique_id)))

    insample <- unlist(c(train_row))
    insample <- ts(insample[!is.na(insample)], frequency=frq)

    naive2_forecast <- Naive2Benchmark(input=insample, fh=fh)
    Naive2_forecasts[i,] <- naive2_forecast
  }

  Naive2_forecasts <- cbind(as.vector(data_train$V1), Naive2_forecasts)
  Naive2_forecasts <- as.data.frame(Naive2_forecasts)

  # Save predictions
  output_file <- paste0(directory, '/results/', dataset_name,'-naive2predictions_', num_obs, '_r_raw.csv')
  write_csv(Naive2_forecasts, output_file)
}

opt = parse_args(opt_parser)
getNaive2Benchmark(opt)
