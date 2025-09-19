library(keras)
library(tensorflow)

#install_keras()

mnist <- dataset_mnist()
x_train <- mnist$train$x / 255
y_train <- to_categorical(mnist$train$y, 10)

model <- keras_model_sequential() |>
  layer_flatten(input_shape = c(28, 28)) |>
  layer_dense(units = 128, activation = "relu") |>
  layer_dense(units = 10, activation = "softmax")

model %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 3, batch_size = 128,
  validation_split = 0.2
)
