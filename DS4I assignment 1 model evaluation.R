### Compiling the results from the tuning

library(dplyr)
library(tidyr)
library(jsonlite)

tuning_results = function(directory){
  
  json_path = paste0(directory, '/trial.json')
  tuning_results = fromJSON(txt = json_path)
  
  number_of_layers = tuning_results$hyperparameters$values$number_of_layers
  learning_rate    = tuning_results$hyperparameters$values$learning_rate
  score            = tuning_results$score
  
  row_df = data.frame(layers = number_of_layers, LR = learning_rate, Val_accuracy = score)
  return(row_df)
}

directories = list.dirs('tuning', recursive = FALSE)
hyperband_search    = 'hyperband'
randomsearch_search = 'randomsearch'
hyperband_directories    = grep(hyperband_search, directories, value = TRUE)
randomsearch_directories = grep(randomsearch_search, directories, value = TRUE)

results_compiler = function(method_directory){

  for (i in 1:length(method_directory)){
    trial_directories = list.dirs(method_directory[i])
    results_df = do.call(rbind, lapply(trial_directories[-1], tuning_results))
    results_df = arrange(results_df, desc(Val_accuracy))
    save(results_df, file = paste0(method_directory[i], '/summary.RData'))
  }
}

results_compiler(hyperband_directories)
results_compiler(randomsearch_directories)


# use this line to get the saved data. Change the randomsearch for 
# hyperband and the 1 for 2 or 3 
# load('tuning/randomsearch results 1/summary.RData')

###__________________________________________________________________________###