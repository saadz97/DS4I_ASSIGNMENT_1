###__________________________________________________________________________###

###                 Interpreting the results from tuning                     ###

###__________________________________________________________________________###

library(dplyr)
library(keras)
library(tensorflow)
library(reticulate)
library(caret)
library(knitr)
library(kableExtra)

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
df = mutate(df, 'Predictor set' = rep(1:4, each = 3), .before = 'Validation accuracy')

save(df, file = 'tuning/tuning_summary_table.RData')
load('tuning/tuning_summary_table.RData')

summary_table = kable(df, format = 'html', booktabs = TRUE) %>% 
  kable_styling(full_width = FALSE, position = 'center') %>%
  collapse_rows(columns = 1, valign = 'middle')

for (i in seq(3, nrow(df), by = 3)) {
  summary_table = row_spec(summary_table, i, 
                           extra_css = "border-bottom: 2px solid black;")
}

summary_table

### Extract the ideal model in here and save the image that keras plot produces 

load('data/training_data.RData')
load('data/testing_data.RData')

load('data/y_train.RData')
load('data/y_test.RData')

y_train = to_categorical(y_train, num_classes = 5)
y_test  = to_categorical(y_test, num_classes = 5)

x_train = training_data[[4]]
x_test  = testing_data[[4]]

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

tuner = kerastuneR::RandomSearch(hypermodel = model_builder, 
                                 objective = 'val_categorical_accuracy',
                                 max_trials = 1, 
                                 executions_per_trial = 1,
                                 directory = 'tuning',
                                 project_name = 'randomsearch results 4')

tuner$reload()
best_model   = tuner$get_best_models(num_models = as.integer(1))[[1]] 

tf = tensorflow::tf
plot_model = tf$keras$utils$plot_model
plot_model(best_model, show_shapes = TRUE, show_layer_names = TRUE,
           expand_nested = FALSE,
           show_layer_activations = TRUE,
           dpi = 500,
           to_file = 'best_model_plot.png')

### End of the plotting

### Need to make sure that the training and test data is saved into separate files
### that i can jut read in

load('data/training_data.RData')
load('data/testing_data.RData')

load('data/y_train.RData')
load('data/y_test.RData')

y_train = to_categorical(y_train, num_classes = 5)
y_test  = to_categorical(y_test, num_classes = 5)

x_train = as.matrix(training_data[[4]])
x_test  = as.matrix(testing_data[[4]])

results      = best_model %>% evaluate(x_test, y_test)
y_pred_probs = best_model %>% predict(x_test)
y_pred       = max.col(y_pred_probs) - 1
true_classes = apply(y_test, 1, which.max) - 1 # it was one-hot encoded so i changed it back
metrics_list = confusionMatrix(factor(y_pred), factor(true_classes))

con_mat = as.data.frame(metrics_list$table)
con_mat = rename(con_mat, Predicted = Prediction, Reference = Reference, Count = Freq)
con_mat =  tidyr::pivot_wider(con_mat, names_from = Reference, values_from = Count,
                              values_fill = 0)
metrics_mat = metrics_list$byClass
save(con_mat, file = 'test/con_mat.RData')
save(metrics_mat, file = 'test/metrics_mat.RData')


load('test/con_mat.RData')
kable(con_mat, format = 'html', caption = 'Confusion matrix') %>%
  kable_styling(full_width = FALSE, position = 'center') %>%
  add_header_above(c(" " = 1, "Reference" = 5))


load('test/metrics_mat.RData')
kable(metrics_mat, format = 'html', caption = 'Class metrics', digits = 3) %>%
  kable_styling(full_width = FALSE, position = 'center')


###__________________________________________________________________________###

###                             End                                          ###

###__________________________________________________________________________###