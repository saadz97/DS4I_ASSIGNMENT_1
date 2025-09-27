###__________________________________________________________________________###

###              Model building and parameter tuning                         ###

###__________________________________________________________________________###

library(keras)
library(kerastuneR)
library(tensorflow)
library(dplyr)
library(tidyr)
library(reticulate)

# ensure that inside the folder you have the project in you also have a folder
# called data that contains the data. 
data = read.csv('./data/scotland_avalanche_forecasts_2009_2025.csv')

# this is a very general solution to the problem of missing entries
# maybe imputation?
data = drop_na(data)
data = filter(data, FAH != '', OAH != '', Precip.Code != '')

data = filter(data, Alt <= 1300,  Alt >= 0)
data = filter(data, Aspect <= 360, Aspect >= 0)
data = filter(data, Incline <= 90, Incline >= 0)
data = filter(data, Wind.Dir <= 360, Wind.Dir >= 0)
data = filter(data, Wind.Speed >= 0)
data = filter(data, Cloud >= 0)
# Drift needs to be a factor
data = filter(data, Total.Snow.Depth >= 0)
data = filter(data, Foot.Pen >= 0)
data = filter(data, Ski.Pen >= 0)
# Rain.at.900 needs to be a factor
data = filter(data, Summit.Wind.Speed >= 0)
data = filter(data, Summit.Wind.Dir <= 360, Summit.Wind.Dir >= 0)
#data = filter(data, Max.Temp.Grad >= 10)
# Max.Hardness.Grad is a categorical
data = filter(data, Snow.Temp <= 5)


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

#unique(data['Area'])
#unique(data['Obs'])
#unique(data['FAH'])
#unique(data['OAH'])
#unique(data['Precip.Code'])

load('data/train_scaled.RData')
load('data/test_scaled.RData')

colnames(select(data, longitude:Incline))
colnames(select(data, Air.Temp:Summit.Wind.Speed))
colnames(select(data, Max.Temp.Grad:Snow.Temp))

# we removed lattitude and longitude and incline
# why is area now conitnuous and scaled instead of a category?
# need to remove the FAH_factor

test_scaled  = test_scaled[, -22]
train_scaled = train_scaled[, -22]

set.seed(2025)
training_indices = sample(1:nrow(data), floor(nrow(data) * 0.7), replace = F)

data$FAH         = as.integer(factor(x = data$FAH)) - 1
data$OAH         = as.integer(factor(x = data$OAH)) - 1
data$Precip.Code = as.integer(factor(data$Precip.Code)) - 1

predictor_set_1       = select(data, c('longitude':'Incline'))
predictor_set_1_train = mutate(predictor_set_1[training_indices, ], 
                               across(c(longitude : Incline), scale))
predictor_set_1_test  = mutate(predictor_set_1[-training_indices, ], 
                               across(c(longitude : Incline), scale))
predictor_set_1_train = as.matrix(predictor_set_1_train)
predictor_set_1_test  = as.matrix(predictor_set_1_test)

predictor_set_2       = select(data, c('Air.Temp':'Summit.Wind.Speed'))
predictor_set_2_train = mutate(predictor_set_2[training_indices, ],
                               across(c(Air.Temp : Summit.Wind.Speed, - Precip.Code), scale))
predictor_set_2_test  = mutate(predictor_set_2[-training_indices, ],
                               across(c(Air.Temp : Summit.Wind.Speed, - Precip.Code), scale))
predictor_set_2_train = as.matrix(predictor_set_2_train)
predictor_set_2_test  = as.matrix(predictor_set_2_test)

predictor_set_3       = select(data, c('Max.Temp.Grad':'Snow.Temp'))
predictor_set_3_train = mutate(predictor_set_3[training_indices, ],
                               across(c(Max.Temp.Grad : Snow.Temp), scale))
predictor_set_3_test  = mutate(predictor_set_3[-training_indices, ],
                               across(c(Max.Temp.Grad : Snow.Temp), scale))
predictor_set_3_train = as.matrix(predictor_set_3_train)
predictor_set_3_test  = as.matrix(predictor_set_3_test)

predictor_set_4_train = cbind(predictor_set_1_train, predictor_set_2_train,
                              predictor_set_3_train)
predictor_set_4_test  = cbind(predictor_set_1_test, predictor_set_2_test,
                              predictor_set_3_test)

  
#training_data_list = list(predictor_set_1_train, predictor_set_2_train,
#                          predictor_set_3_train, predictor_set_4_train)
#testing_data_list = list(predictor_set_1_test, predictor_set_2_test,
#                         predictor_set_3_test, predictor_set_4_test)

y = data$FAH
y = to_categorical(y, num_classes = 5)

#y_train = y[training_indices, ]
#y_test  = y[-training_indices, ]


y_train = to_categorical(train_scaled$FAH, num_classes = 5)
y_test  = to_categorical(test_scaled$FAH, num_classes = 5)
train_scaled = train_scaled[, -1]
test_scaled  = test_scaled[, -1]
training_data_list = list(train_scaled)
testing_data_list. = list(test_scaled)

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
    
  x_train = training_data_list[[i]]
  
  tuner_randomsearch = kerastuneR::RandomSearch(hypermodel = model_builder,
                                                objective = 'val_categorical_accuracy',
                                                max_trials = 75, executions_per_trial = 1,
                                                directory = 'tuning xp',
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
