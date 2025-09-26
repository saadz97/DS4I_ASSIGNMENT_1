###__________________________________________________________________________###

###                 Interpreting the results from tuning                     ###

###__________________________________________________________________________###

library(dplyr)
library(keras)
library(tensorflow)
library(reticulate)
library(caret)

files = c('tuning/randomsearch results 1/summary.RData',
          'tuning/randomsearch results 2/summary.RData',
          'tuning/randomsearch results 3/summary.RData',
          'tuning/randomsearch results 4/summary.RData')
var_names = c('rs_results_1', 'rs_results_2', 'rs_results_3', 'rs_results_4')

for (i in seq_along(files)){
  temp_env = new.env()                       # temporary environment
  load(files[i], envir = temp_env)           # load into temp
  assign(var_names[i], temp_env$results_df)  # assign with custom name
}

results = list(rs_results_1, rs_results_2, rs_results_3, rs_results_4)
df = data.frame()
for (i in 1:4){
  results_df = results[[i]]
  top_3_models = results_df[1:3, ]
  top_3_models = mutate(top_3_models, across(c(Val_accuracy, LR), round, 5))  
  colnames(top_3_models) = c('Validation accuracy',
                             'Learning rate',
                             'Number of layers',
                             paste('nodes on layer ', 1:5)) # make sure to make this the maximum layers
  df = bind_rows(df, top_3_models)
}
df = mutate(df, 'Predictor set' = rep(1:4, each = 3), .before = 'Number of layers')

save(df, file = 'tuning/tuning_summary_table.RData')
load('tuning/tuning_summary_table.RData')


### Build the ideal model in here and save the image that keras plot produces 

input = layer_input(shape = c(27), name = 'Data')

output = input %>% 
  layer_dense(units = 20, activation = 'relu', name = 'Hidden-Layer') %>%
  layer_dropout(rate = 0.5, seed = 2025, name = 'Dropout-Layer-0.5') %>%
  layer_dense(units = 5, activation = 'softmax', name = 'Output-Layer')

model = keras_model(inputs = input, outputs = output)

tf = tensorflow::tf
plot_model = tf$keras$utils$plot_model

plot_model(model, to_file = 'model.png',  show_shapes = TRUE,
           show_layer_names = TRUE, expand_nested = FALSE,
           show_layer_activations = TRUE, dpi = 500)

### End of the plotting

### Extracting the optimal model now

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
                    metrics = c(metric_categorical_accuracy()))
  
  return(model)
}

tuner = kerastuneR::RandomSearch(hypermodel = model_builder, 
                                 objective = 'val_accuracy',
                                 max_trials = 1, # not used, just needed to init
                                 executions_per_trial = 1,
                                 directory = 'tuning xp',
                                 project_name = 'randomsearch results 4')

# Reload past results
tuner$reload()

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

set.seed(2025)
training_indices = sample(1:nrow(data), floor(nrow(data) * 0.7), replace = F)

data$FAH         = as.integer(factor(x = data$FAH)) - 1
data$OAH         = as.integer(factor(x = data$OAH)) - 1
data$Precip.Code = as.integer(factor(data$Precip.Code)) - 1

predictor_set_1 = select(data, c('longitude':'Incline'))
predictor_set_1_train = mutate(predictor_set_1[training_indices, ], 
                               across(c(longitude : Incline), scale))
predictor_set_1_test  = mutate(predictor_set_1[-training_indices, ], 
                               across(c(longitude : Incline), scale))
predictor_set_1_train = as.matrix(predictor_set_1_train)
predictor_set_1_test  = as.matrix(predictor_set_1_test)

predictor_set_2 = select(data, c('Air.Temp':'Summit.Wind.Speed'))
predictor_set_2_train = mutate(predictor_set_2[training_indices, ],
                               across(c(Air.Temp : Summit.Wind.Speed, - Precip.Code), scale))
predictor_set_2_test  = mutate(predictor_set_2[-training_indices, ],
                               across(c(Air.Temp : Summit.Wind.Speed, - Precip.Code), scale))
predictor_set_2_train = as.matrix(predictor_set_2_train)
predictor_set_2_test = as.matrix(predictor_set_2_test)

predictor_set_3 = select(data, c('Max.Temp.Grad':'Snow.Temp'))
predictor_set_3_train = mutate(predictor_set_3[training_indices, ],
                               across(c(Max.Temp.Grad : Snow.Temp), scale))
predictor_set_3_test  = mutate(predictor_set_3[-training_indices, ],
                               across(c(Max.Temp.Grad : Snow.Temp), scale))
predictor_set_3_train = as.matrix(predictor_set_3_train)
predictor_set_3_test = as.matrix(predictor_set_3_test)

predictor_set_4_train = cbind(predictor_set_1_train, predictor_set_2_train,
                              predictor_set_3_train)
predictor_set_4_test = cbind(predictor_set_1_test, predictor_set_2_test,
                             predictor_set_3_test)
y = data$FAH
y = to_categorical(y, num_classes = 5)

y_train = y[training_indices, ]
y_test  = y[-training_indices, ]

best_hp = tuner$get_best_hyperparameters(num_trials = as.integer(1))[[1]]
x_train = predictor_set_4_train
x_test  = predictor_set_4_test
model = model_builder(best_hp)


freqs                = colSums(y_train) / nrow(y_train)
weights              = sqrt(1 / freqs)
class_weights        = as.list(weights)
names(class_weights) = as.character(0:(length(weights)-1))

# retrain on full train set
history = model %>% fit(x = x_train,
                        y = y_train,
                        epochs = 200,
                        batch_size = 32,
                        validation_split = 0.2,
                        class_weight = class_weights,
                        shuffle = TRUE)

# --- STEP 3: Predictions on test set ---
pred_probs   = model %>% predict(x_test)
pred_classes = apply(pred_probs, 1, which.max) - 1   # keras is 0-based
true_classes = apply(y_test, 1, which.max) - 1

# --- STEP 4: Confusion matrix ---
confusionMatrix(factor(pred_classes, levels = 0:4),
                factor(true_classes, levels = 0:4))

###__________________________________________________________________________###

###                             End                                          ###

###__________________________________________________________________________###