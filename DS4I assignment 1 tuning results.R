###__________________________________________________________________________###

###                 Interpreting the results from tuning                     ###

###__________________________________________________________________________###

library(dplyr)
library(keras)
library(tensorflow)
library(reticulate)

files = c('tuning/randomsearch results 1/summary.RData',
          'tuning/randomsearch results 2/summary.RData',
          'tuning/randomsearch results 3/summary.RData',
          'tuning/randomsearch results 4/summary.RData',
          'tuning/hyperband results 1/summary.RData',
          'tuning/hyperband results 2/summary.RData',
          'tuning/hyperband results 3/summary.RData',
          'tuning/hyperband results 4/summary.RData')
var_names = c('rs_results_1', 'rs_results_2', 'rs_results_3', 'rs_results_4',
              'hb_results_1', 'hp_results_2', 'hb_results_3', 'hb_results_4')

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
  top_3_models = mutate(top_3_models, across(c(LR, Val_accuracy), round, 4))  
  colnames(top_3_models) = c('Number of layers', 'Learning rate', 'Validation accuracy')
  df = bind_rows(df, top_3_models)
}
df = mutate(df, 'Predictor set' = rep(1:4, each = 3), .before = 'Number of layers')

save(df, file = 'tuning/tuning_summary_table.RData')
load('tuning/tuning_summary_table.RData')



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



###__________________________________________________________________________###

###                             End                                          ###

###__________________________________________________________________________###