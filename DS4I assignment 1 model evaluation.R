###__________________________________________________________________________###

###                 Compiling the results from the tuning                    ###

###__________________________________________________________________________###

library(dplyr)
library(tidyr)
library(jsonlite)

tuning_results = function(directory, max_layers){
  
  json_path = paste0(directory, '/trial.json')
  tuning_results = fromJSON(txt = json_path)
  hp               = tuning_results$hyperparameters$values
  number_of_layers = hp$number_of_layers
  learning_rate    = hp$learning_rate
  score            = tuning_results$score
  
  layers_nodes     = matrix(rep(NA, 3), nrow = 1)
  colnames(layers_nodes) = c('nodes on layer 1', 'nodes on layer 2', 'nodes on layer 3')
  for (i in 1:number_of_layers) {
    node_name = paste0('nodes_layer_', i)
    layers_nodes[1, i] = ifelse(!is.null(hp[[node_name]]), hp[[node_name]], NA)
  }
  
  row_df = data.frame( Val_accuracy = score,  LR = learning_rate, 
                       layers = number_of_layers, layers_nodes, 
                       check.names = FALSE)
  return(row_df)
}

directories              = list.dirs('tuning xp', recursive = FALSE)
hyperband_search         = 'hyperband'
randomsearch_search      = 'randomsearch'
hyperband_directories    = grep(hyperband_search, directories, value = TRUE)
randomsearch_directories = grep(randomsearch_search, directories, value = TRUE)

results_compiler = function(method_directory){

  for (i in 1:length(method_directory)){
    trial_directories = list.dirs(method_directory[i])
    results_df = do.call(rbind, lapply(trial_directories[-1], tuning_results, max_layers = 3))
    results_df = arrange(results_df, desc(Val_accuracy))
    save(results_df, file = paste0(method_directory[i], '/summary.RData'))
  }
}

# the results automatically get saved locally, inside the same tuning folder
#results_compiler(hyperband_directories)
results_compiler(randomsearch_directories)

# use this line to get the saved data. Change the randomsearch for 
# hyperband and the 1 for 2 or 3.
# load('tuning/randomsearch results 1/summary.RData')

load('tuning xp/randomsearch results 4/summary.RData')
###__________________________________________________________________________###

###                             End                                          ###

###__________________________________________________________________________###