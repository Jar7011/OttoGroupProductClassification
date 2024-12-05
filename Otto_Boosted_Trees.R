library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(xgboost)

# Read in the data
train <- vroom('train.csv')
test <- vroom('test.csv')

# Create recipe
boost_recipe <- recipe(target ~ ., data = train) %>% 
  step_mutate(target = as.factor(target)) %>% 
  step_normalize(all_numeric_predictors())
