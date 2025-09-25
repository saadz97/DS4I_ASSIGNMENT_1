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
# FAH = the forecast avalanche hazard for the following da
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


data$FAH         = as.integer(factor(x = data$FAH)) - 1
data$OAH         = as.integer(factor(x = data$OAH)) - 1
data$Precip.Code = as.integer(factor(data$Precip.Code)) - 1

predictor_set_1 = select(data, c('longitude':'Incline'))
predictor_set_1 = mutate(predictor_set_1, across(c(longitude : Incline), scale))
predictor_set_1 = as.matrix(predictor_set_1)

predictor_set_2 = select(data, c('Air.Temp':'Summit.Wind.Speed'))
predictor_set_2 = mutate(predictor_set_2,
                         across(c(Air.Temp : Summit.Wind.Speed, - Precip.Code), scale))
predictor_set_2 = as.matrix(predictor_set_2)

predictor_set_3 = select(data, c('Max.Temp.Grad':'Snow.Temp'))
predictor_set_3 = mutate(predictor_set_3, across(c(Max.Temp.Grad : Snow.Temp), scale))
predictor_set_3 = as.matrix(predictor_set_3)

predictor_set_4 = cbind(predictor_set_1, predictor_set_2, predictor_set_3)

set.seed(2025)
training_indices = runif(n = floor(nrow(data) * 0.7), min = 1, max = nrow(data))

predictor_set_1_train = predictor_set_1[training_indices, ] 
predictor_set_1_test  = predictor_set_1[-training_indices, ]

predictor_set_2_train = predictor_set_2[training_indices, ] 
predictor_set_2_test  = predictor_set_2[-training_indices, ]

predictor_set_3_train = predictor_set_3[training_indices, ] 
predictor_set_3_test  = predictor_set_3[-training_indices, ]

predictor_set_4_train = predictor_set_4[training_indices, ]
predictor_set_4_test  = predictor_set_4[-training_indices, ]
  
training_data_list = list(predictor_set_1_train, predictor_set_2_train,
                          predictor_set_3_train, predictor_set_4_train)
testing_data_list = list(predictor_set_1_test, predictor_set_2_test,
                         predictor_set_3_test, predictor_set_4)

y = data$FAH
y = to_categorical(y, num_classes = 5)

y_train = y[training_indices, ]
y_test  = y[-training_indices, ]

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
  
  n_layers = hp$Int('number_of_layers', min_value = 1, max_value = 3, step = 1)
  lr = hp$Choice('learning_rate', values = c(1e-2, 1e-3, 1e-4))  
  
  n_x   = ncol(x_train)
  input = layer_input(shape = c(n_x))

  x = input
  for (i in 1:n_layers){
    
    n_nodes = hp$Int(paste0('nodes_layer_', i),
                      min_value = 30, max_value = 50, step = 10)
    
    x = x %>%
      layer_dense(units = n_nodes, activation = 'relu') %>%
      layer_dropout(rate = 0.15)
  }
  output = x %>% 
    layer_dense(units = 5, activation = 'softmax')
  
  model = keras_model(inputs = input, outputs = output)
  
  model %>% compile(loss = 'categorical_crossentropy', 
                    optimizer = optimizer_adam(learning_rate = lr),
                    metrics = c('accuracy'))
  
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
                                                objective = 'val_accuracy',
                                                max_trials = 50, executions_per_trial = 1,
                                                directory = 'tuning xp',
                                                project_name = paste('randomsearch results', i),
                                                overwrite = TRUE)
  
  tuner_randomsearch %>% fit_tuner(x = x_train,
                                   y = y_train,
                                   epochs = 200,
                                   batch_size = 32,
                                   class_weight = class_weights,
                                   validation_split = 0.2,
                                   shuffle = TRUE)
  
  #results_summary(tuner = tuner_randomsearch, num_trials = 5)
  
  #tuner_hyperband = kerastuneR::Hyperband(hypermodel = model_builder,
  #                                        objective = 'val_accuracy',
  #                                        directory = 'tuning',
  #                                        project_name = paste('hyperband results', i),
  #                                        max_epochs = 50,
  #                                        hyperband_iterations = 50,
  #                                        seed = 2025, overwrite = TRUE)
  
  #tuner_hyperband %>% fit_tuner(x = x_train,
  #                              y = y_train,
  #                              epochs = 50,
  #                              validation_split = 0.2, shuffle = TRUE)
  
  #results_summary(tuner = tuner_hyperband, num_trials = 5)
}


input = layer_input(shape = c(27), name = 'Data')

output = input %>% 
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dropout(rate = 0.15, seed = 2025) %>%
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dropout(rate = 0.15, seed = 2025) %>%
  layer_dense(units = 5, activation = 'softmax', name = 'Output-Layer')

model = keras_model(inputs = input, outputs = output)

model %>% compile(loss = 'categorical_crossentropy', 
                  optimizer = optimizer_adam(learning_rate = 0.001),
                  metrics = c(metric_categorical_accuracy()))

class_weights = list(
  '0' = (colSums(y_train)/dim(y_train)[1])[1]^(-1),
  '1' = (colSums(y_train)/dim(y_train)[1])[2]^(-1),
  '2' = (colSums(y_train)/dim(y_train)[1])[3]^(-1),
  '3' = (colSums(y_train)/dim(y_train)[1])[4]^(-1),
  '4' = (colSums(y_train)/dim(y_train)[1])[5]^(-1)
)

history = model %>% fit(predictor_set_4_train,
                        y_train,
                        epochs = 200, batch_size = 32, 
                        validation_split = 0.2, shuffle = TRUE,
                        class_weight = class_weights) 

freqs = colSums(y_train) / nrow(y_train)
weights = sqrt(1 / freqs)
#names(weights) = 0:4   # keys are now 0,1,2,3,4
class_weights <- dict()
for (k in 0:(length(weights)-1)) {
  class_weights[[k]] <- weights[k + 1]
}

library(yardstick)

# 1. Get predictions on validation data
pred_probs = model %>% predict(predictor_set_4_train[750:2750, ])

# Convert softmax outputs to class indices (0–4)
pred_classes = max.col(pred_probs) - 1

# Convert one-hot encoded y_valid to class indices
true_classes = max.col(y_train[750:2750, ]) - 1

# 2. Put into a tibble for yardstick
results = tibble(truth = factor(true_classes), 
                 estimate = factor(pred_classes))

# 3. Confusion matrix
conf_mat(results, truth, estimate)

# 4. Accuracy
accuracy(results, truth, estimate)

# 5. Per-class precision, recall, F1
precision(results, truth, estimate)
recall(results, truth, estimate)
f_meas(results, truth, estimate, beta = 1)  # F1

# 6. Macro-averaged metrics (treat all classes equally)
results %>%
  group_by(truth) %>%
  f_meas(truth, estimate, beta = 1) %>%
  summarise(macro_f1 = mean(.estimate))

# 7. Weighted (support-weighted) metrics if needed
results %>%
  group_by(truth) %>%
  summarise(n = n()) %>%
  left_join(
    results %>%
      group_by(truth) %>%
      f_meas(truth, estimate, beta = 1),
    by = "truth"
  ) %>%
  summarise(weighted_f1 = weighted.mean(.estimate, n))

###__________________________________________________________________________###

###                                  End                                     ###

###__________________________________________________________________________###
