library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(lightgbm)
library(bonsai)

# Read in the data
train <- vroom('train.csv')
test <- vroom('test.csv')

# Make target a factor
train <- train %>% 
  mutate(target = as.factor(target))

# # Make small training and testing set
# train_small <- train %>% sample_n(1000)
# test_small <- test %>% sample_n(300)

# Create recipe
boost_recipe <- recipe(target ~ ., data = train) %>% 
  step_normalize(all_numeric_predictors())

# Create model 
boost_model <- boost_tree(tree_depth = tune(),
                          trees = 1000,
                          learn_rate = tune()) %>% 
  set_engine('lightgbm') %>% 
  set_mode('classification')

# Create workflow
boost_wf <- workflow() %>% 
  add_recipe(boost_recipe) %>% 
  add_model(boost_model)

# Grid of values to tune over
tuning_grid <- grid_regular(tree_depth(), learn_rate = 0.1, levels = 5)

# Split data for CV
folds <- vfold_cv(train, v = 10, repeats = 1)

# Run the CV
cv_results <- boost_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(mn_log_loss))

# Find best tuning params
best_tuning_params <- cv_results %>% 
  select_best(metric = 'mn_log_loss')

# Finalize workflow
final_wf <- boost_wf %>% 
  finalize_workflow(best_tuning_params) %>% 
  fit(data = train)

# Predict and format for submission
boost_preds <- predict(final_wf, new_data = test, type = "prob") %>% 
  bind_cols(., test) %>% 
  select(id, everything())

colnames(boost_preds) <- gsub('.pred_', '', colnames(boost_preds))

# Write out file
vroom_write(x = boost_preds, file = "submission.csv", delim = ",")
