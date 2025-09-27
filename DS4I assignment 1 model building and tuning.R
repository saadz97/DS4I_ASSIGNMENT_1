###__________________________________________________________________________###

###              Model building and parameter tuning                         ###

###__________________________________________________________________________###

library(keras)
library(kerastuneR)
library(tensorflow)
library(dplyr)
library(tidyr)
library(reticulate)


## data description ##

# Date = the date that the forecast was made
# Area = one of six forecasting region
# FAH = the forecast avalanche hazard for the following day
# OAH = the observed avalanche hazard on the following day (observation made the following day)
# longitude:Incline: position and topography at forecast location (predictor set)
# Air.Temp:Summit.Wind.Speed = weather in the vicinity of the forecast location at the
# time the forecast was made (predictor set 2)
# Max.Temp.Grad:Snow.Temp = results of a ”snow pack test” of the integrity of snow at
# the forecast location (predictor set 3)


#colnames(select(data, longitude:Incline))
#colnames(select(data, Air.Temp:Summit.Wind.Speed))
#colnames(select(data, Max.Temp.Grad:Snow.Temp))

load('data/training_data.RData')
load('data/testing_data.RData')

load('data/y_train.RData')
load('data/y_test.RData')

y_train = to_categorical(y_train, num_classes = 5)
y_test  = to_categorical(y_test, num_classes = 5)

comp_mat = matrix(c(colSums(y_train) / nrow(y_train), 
                    colSums(y_test) / nrow(y_test)),
                  byrow = T, nrow = 2)
colnames(comp_mat) = paste('category ', 0:4)
rownames(comp_mat) = c('train', 'test')
#comp_mat

#set.seed(2025)

#input = layer_input(shape = c(10))

#output = input %>% 
#  layer_dense(units = 100, activation = 'relu') %>%
#  layer_dropout(rate = 0.5, seed = 2025) %>%
#  layer_dense(units = 50, activation = 'relu') %>%
#  layer_dropout(rate = 0.5, seed = 2026) %>%
#  layer_dense(units = 15, activation = 'relu') %>%
#  layer_dropout(rate = 0.5, seed = 2027) %>%
#  layer_dense(units = 5, activation = 'softmax')

#model = keras_model(inputs = input, outputs = output)

#model %>% compile(loss = 'categorical_crossentropy', 
#                  optimizer = optimizer_adam(learning_rate = 0.005),
#                  metrics = c('accuracy'))
  
#history = model %>% fit(predictor_set_3_train,
#                        y_train,
#                        epochs = 100, batch_size = 5, 
#                        validation_split = 0.2, shuffle = TRUE)     

#plot(history)
     
model_builder = function(hp){
  
  n_layers = hp$Int('number_of_layers', min_value = 1, max_value = 5, step = 1)
  lr = hp$Choice('learning_rate', values = seq(from = 1e-2, to = 1e-4, length.out = 5))  
  
  n_x   = ncol(x_train)
  input = layer_input(shape = c(n_x))

  x = input
  for (i in 1:n_layers){
    
    n_nodes = hp$Int(paste0('nodes_layer_', i),
                      min_value = 30, max_value = 50, step = 10)
    
    x = x %>%
      layer_dense(units = n_nodes, activation = 'relu') %>%
      layer_dropout(rate = 0.1)
  }
  output = x %>% 
    layer_dense(units = 5, activation = 'softmax')
  
  model = keras_model(inputs = input, outputs = output)
  
  model %>% compile(loss = 'categorical_crossentropy', 
                    optimizer = optimizer_adam(learning_rate = lr),
                    metrics = c(metric_categorical_accuracy()))
  
  return(model)
}

# this will create a new folder called tuning inside your project folder
# in that folder it will contain the information about the trials
# this should probably be converted into a proper function  
for (i in 1:4){
  
  freqs = colSums(y_train) / nrow(y_train)
  weights = sqrt(1 / freqs)
  class_weights = dict()
  for (k in 0:(length(weights)-1)) {
    class_weights[[k]] = weights[k + 1]
  }
    
  x_train = training_data[[i]]
  
  tuner_randomsearch = kerastuneR::RandomSearch(hypermodel = model_builder,
                                                objective = 'val_categorical_accuracy',
                                                max_trials = 75, executions_per_trial = 3,
                                                directory = 'tuning',
                                                project_name = paste('randomsearch results', i),
                                                overwrite = TRUE)
  
  tuner_randomsearch %>% fit_tuner(x = x_train,
                                   y = y_train,
                                   epochs = 100,
                                   batch_size = 32,
                                   class_weight = class_weights,
                                   validation_split = 0.2,
                                   shuffle = TRUE)
}

###__________________________________________________________________________###

###                                  End                                     ###

###__________________________________________________________________________###
