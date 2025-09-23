###__________________________________________________________________________###

###                 Interpreting the results from tuning                     ###

###__________________________________________________________________________###


library(dplyr)

load('tuning/randomsearch results 3/summary.RData')
#load('tuning/randomsearch results 2/summary.RData')

top_5_models = results_df[1:5, ]
top_5_models = mutate(top_5_models, across(c(LR, Val_accuracy), round, 3))
colnames(top_5_models) = c('Number of layers', 'Learning rate', 'Validation accuracy')




###__________________________________________________________________________###

###                             End                                          ###

###__________________________________________________________________________###