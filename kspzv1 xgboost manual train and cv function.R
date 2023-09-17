xgb_train_and_cv <- function(train_features, validation_features){
  #prepare training set
  train_dat <- train_features %>%
    t() %>%
    data.frame(check.names = FALSE) %>%
    t() %>%
    data.frame(check.names = FALSE) %>%
    mutate(class = factor(class)) %>%
    mutate_at(c(2:ncol(.)), as.numeric) 
  #convert data frame to data table and preserve rownames
  setDT(train_dat, keep.rownames = TRUE) 
  train_dat_samplenames <- train_dat$rn
  train_dat <- train_dat[,-1]
  #prepare validation set
  validation_dat <- validation_features %>%
    t() %>%
    data.frame(check.names = FALSE) %>%
    t() %>%
    data.frame(check.names = FALSE) %>%
    mutate(class = factor(class)) %>%
    mutate_at(c(2:ncol(.)), as.numeric)
  #convert data frame to data table and preserve rownames
  setDT(validation_dat, keep.rownames = TRUE) 
  validation_dat_samplenames <- validation_dat$rn
  validation_dat <- validation_dat[,-1]
  
  #sanity check
  if(all(colnames(train_dat) == colnames(train_features)) &
     all(train_dat_samplenames == rownames(train_features))){
    print(paste0("training set for run ", i," good to go!"))
    } else {
      print("please check to see if training samples and features match.")
      }
  if(all(colnames(validation_dat) == colnames(validation_features)) &
     all(validation_dat_samplenames == rownames(validation_features))){
    print(paste0("validation set for run ", i," good to go!"))
    } else {
      print("please check to see if validation samples and features match.")
    }
  
  print(paste0("Training set reduced to ", nrow(train_features), " samples and ", ncol(train_features), " features."))
  print(paste0("Validation set reduced to ", nrow(validation_features), " samples and ", ncol(validation_features), " features."))
  
  #convert characters to factors
  fact_col <- colnames(train_dat)[sapply(train_dat,is.character)]
  for(k in fact_col) set(train_dat, j=k, value = factor(train_dat))
  for(k in fact_col) set(validation_dat, j=k, value = factor(validation_dat))
  
  #make dataframe to link original colnames to syntactical colnames; allows you to always map back to original names
  colname_key_df <- data.frame(og_colname = colnames(train_dat),
                               syntactic_colname = make.names(colnames(train_dat), unique = TRUE))
  if(all(colnames(train_dat) == colname_key_df$og_colname) &
     all(colnames(train_dat) == colnames(validation_dat))){
    colnames(train_dat) <- colname_key_df$syntactic_colname  #make colnames syntactic for downstream ML functions
    colnames(validation_dat) <- colname_key_df$syntactic_colname #make colnames syntactic for downstream ML functions
    } else {
      print("names don't match")
      }
  #create tasks
  traintask <- mlr::makeClassifTask(data = as.data.frame(train_dat), target = "class")
  validationtask <- mlr::makeClassifTask(data = as.data.frame(validation_dat), target = "class")
  
  #do one hot encoding`<br/> 
  traintask <- createDummyFeatures (obj = traintask) 
  validationtask <- createDummyFeatures (obj = validationtask)
  
  #assign data and labels using one hot encoding
  train_labels <- train_dat$class
  new_train <- model.matrix(~.+0,data = train_dat[,-c("class"),with=F])
  colnames(new_train) <- gsub("\\`","",colnames(new_train))
  #convert factor to numeric 
  train_labels <- as.numeric(train_labels)-1 #note that 0=infected (not protected) and 1 = protected
  #prepare matrix
  dtrain <- xgb.DMatrix(data = new_train, label = train_labels)
  #match and replace synctactic colnames with original names
  colnames(dtrain) <- colname_key_df$og_colname[match(colnames(dtrain), colname_key_df$syntactic_colname)]
  #parameter tuning
  print(paste0("Begin run number ", i, " of ", runs, " total runs."))
  print(paste0("Seed for run and splitting train and validation set was ", run_seed, "."))
  print(paste0("Max iterations for hyperparameter tuning: ", maxiterations))
  print(paste0("Running hyperparameter tuning on run number ", i, " of ", runs, " total runs."))
  tic(msg = paste0("hyperparameter tuning for run ", i))
  #set parameter space
  #https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/
  #https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning
  #create learner
  lrn <- makeLearner("classif.xgboost",
                     objective="binary:logistic",
                     nrounds=1000,
                     early_stopping_rounds = 100,
                     eval_metric="error",
                     predict.type = "response")
  params <- makeParamSet( makeDiscreteParam("booster", values = c("gbtree")),
                          makeIntegerParam("gamma",lower = 0L,upper = 3L),
                          makeIntegerParam("max_depth",lower = 2L,upper = 5L),
                          makeNumericParam("eta",lower = 0.01,upper = 0.2),
                          makeNumericParam("min_child_weight",lower = 0L,upper = 4L),
                          makeNumericParam("subsample",lower = 0.75,upper = 0.9),
                          makeNumericParam("lambda",lower = 0, upper = 1),
                          makeNumericParam("alpha",lower = 0, upper = 1),
                          makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))
  
  #set resampling strategy
  rdesc <- makeResampleDesc(method = "CV",
                            predict = "test",
                            iters = 4,
                            stratify = T)
  tune_fs_res <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params, control = ctrl, show.info = T)
  
  #set hyperparameters
  #search strategy
  lrn_tune <- setHyperPars(lrn, par.vals = tune_fs_res$x)
  best_hyper_pars <- data.frame(iterations = maxiterations, lrn_tune$par.vals)
  tuned_params <- list(booster = best_hyper_pars$booster,
                       objective = best_hyper_pars$objective,
                       eta = best_hyper_pars$eta,
                       gamma = best_hyper_pars$gamma,
                       max_depth = best_hyper_pars$max_depth,
                       min_child_weight = best_hyper_pars$min_child_weight,
                       subsample = best_hyper_pars$subsample,
                       colsample_bytree = best_hyper_pars$colsample_bytree)
  toc()
  #retrain model on best hyperparameters
  print(paste0("Training on best hyperparameters on run number ", i, " of ", runs, " total runs."))
  tic(msg = paste0("training on best hyperparameters for run ", i))
  xgb1 <- xgb.train(params = tuned_params,
                    data = dtrain,
                    nrounds = 100,
                    watchlist = list(train=dtrain),
                    print_every_n = 10,
                    early_stopping_rounds = 20,
                    maximize = F ,
                    eval_metric = "error")
  #determine feature importance
  mat1 <- xgb.importance(feature_names = xgb1$feature_names, model = xgb1)
  print(paste0("Top 5 features of run number ", i, " of ", runs, " total runs are ",
               mat1$Feature[1], ", ",
               mat1$Feature[2], ", ",
               mat1$Feature[3], ", ",
               mat1$Feature[4], ", ",
               mat1$Feature[5], "."))
  
  feature_importance <- as.data.frame(mat1) %>%
    rownames_to_column(var = "rank") %>%
    drop_na() %>%
    # left_join(., colname_key_df %>%
    #             dplyr::rename(Feature = "syntactic_colname"),
    #           by = "Feature") %>%
    # dplyr::select(-Feature) %>%
    # dplyr::rename(Feature = "og_colname") %>%
    dplyr::select(rank, Feature, everything())
  
  top_ten_features <- feature_importance$Feature[1:10][!is.na(feature_importance$Feature[1:10])] #get top 10 most important features
  toc()
  tic(msg = paste0("feature selection for ", runs, " total runs"))
  #parameter tuning
  print(paste0("Begin second step training on downselected features for number ", i, " of ", runs, " total runs."))
  n_feat_sampled_random <- sample(n_feat_sampled, 1)
  if(length(top_ten_features) >= n_feat_sampled_random){
    new_train_feat <- c("class", sample(top_ten_features, n_feat_sampled_random, replace=FALSE))
  }else{
      new_train_feat <- c("class", sample(top_ten_features, length(top_ten_features), replace=FALSE))}
  train_dat_temp <- t(train_features[,new_train_feat]) %>%
    data.frame(check.names = FALSE) %>%
    t() %>%
    data.frame(check.names = FALSE) %>%
    mutate(class = factor(class)) %>%
    mutate_at(c(2:ncol(.)), as.numeric)
  #convert data frame to data table and preserve rownames
  setDT(train_dat_temp, keep.rownames = TRUE, check.names=FALSE) 
  train_dat_temp_samplenames <- train_dat_temp$rn
  train_dat_temp <- train_dat_temp[,-1]
  #sanity check
  if(all(colnames(train_dat_temp) == new_train_feat) &
     all(train_dat_temp_samplenames == rownames(train_features))){
    print("downselected training set good to go!")
  } else {
    print("please check to see if training samples and features match.")
  }
  #assign data and labels using one hot encoding 
  train_labels <- train_dat_temp$class
  new_train <- model.matrix(~.+0, data = train_dat_temp[,-c("class"),with=F])
  colnames(new_train) <- gsub("\\`","",colnames(new_train))
  #convert factor to numeric 
  train_labels <- as.numeric(train_labels)-1 #note that 0=infected (not protected) and 1 = protected
  #prepare matrix
  dtrain <- xgb.DMatrix(data = new_train, label = train_labels)
  #convert characters to factors
  fact_col <- colnames(train_dat_temp)[sapply(train_dat_temp,is.character)]
  for(k in fact_col) set(train_dat_temp, j=k, value = factor(train_dat_temp[[k]]))
  #make dataframe to link original colnames to syntactical colnames; allows you to always map back to original names
  colname_key_df <- data.frame(og_colname = colnames(train_dat_temp),
                               syntactic_colname = make.names(colnames(train_dat_temp), unique = TRUE))
  if(all(colnames(train_dat_temp) == colname_key_df$og_colname)){
    colnames(train_dat_temp) <- colname_key_df$syntactic_colname
  } else {
    print("names don't match")
  }
  #create tasks
  traintask <- mlr::makeClassifTask(data = as.data.frame(train_dat_temp), target = "class")
  #do one hot encoding`<br/> 
  traintask <- createDummyFeatures (obj = traintask)
  #search strategy
  ctrl <- makeTuneControlRandom(maxit = maxiterations)
  print(paste0("Running hyperparameter tuning on run number ", i, " of ", runs, " total runs."))
  print(paste0("maxiterations for hyperparameter tuning: ", maxiterations))
  tic(msg = paste0("hyperparameter tuning for run ", i))
  #create learner
  lrn <- makeLearner("classif.xgboost",
                     objective="binary:logistic",
                     nrounds=1000,
                     early_stopping_rounds = 100,
                     eval_metric="error",
                     predict.type = "response")
  tune_ds_res <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params, control = ctrl, show.info = T)
  #set hyperparameters
  lrn_tune_ds <- setHyperPars(lrn, par.vals = tune_ds_res$x)
  best_hyper_pars_ds <- data.frame(iterations = maxiterations, lrn_tune_ds$par.vals)
  tuned_params_ds <- list(booster = best_hyper_pars_ds$booster,
                       objective = best_hyper_pars_ds$objective,
                       eta = best_hyper_pars_ds$eta,
                       gamma = best_hyper_pars_ds$gamma,
                       max_depth = best_hyper_pars_ds$max_depth,
                       min_child_weight = best_hyper_pars_ds$min_child_weight,
                       subsample = best_hyper_pars_ds$subsample,
                       colsample_bytree = best_hyper_pars_ds$colsample_bytree)
  toc()
  #retrain model with cv on best hyperparameters
  print(paste0("Training on best hyperparameters on run number ", i, " of ", runs, " total runs."))
  tic(msg = paste0("training on best hyperparameters for run ", i))
  xgb2 <- xgb.train(params = tuned_params_ds,
                    data = dtrain,
                    nrounds = 100,
                    watchlist = list(train=dtrain),
                    print_every_n = 10,
                    early_stopping_rounds = 20,
                    maximize = F ,
                    eval_metric = "error")
  #get variable importance
  mat2 <- xgb.importance(feature_names = xgb2$feature_names, model = xgb2)
  print(paste0("Top 3 downselected features of run number ", i, " of ", runs, " total runs are ",
               mat2$Feature[1], ", ",
               mat2$Feature[2], ", ",
               mat2$Feature[3], "."))
  #xgb.plot.importance (importance_matrix = mat2[1:20]) 
  features_ds <- as.data.frame(mat2) %>%
    rownames_to_column(var = "rank") %>%
    drop_na()
  xgb2_best_scores <- data.frame("features" = paste(sort(features_ds$Feature), collapse = '; '),
                                      "error" = xgb2$best_score) 
  toc()
  #test on validation set
  validation_dat_temp <- t(validation_features[,new_train_feat]) %>%
    data.frame(check.names = FALSE) %>%
    t() %>%
    data.frame(check.names = FALSE) %>%
    mutate(class = factor(class)) %>%
    mutate_at(c(2:ncol(.)), as.numeric)
  #convert data frame to data table and preserve rownames
  setDT(validation_dat_temp, keep.rownames = TRUE, check.names=FALSE)
  validation_dat_temp_samplenames <- validation_dat_temp$rn
  validation_dat_temp <- validation_dat_temp[,-1]
  #sanity check
  if(all(colnames(validation_dat_temp) == new_train_feat) &
     all(validation_dat_temp_samplenames == rownames(validation_features))){
    print(paste0("validation set good to go!"))
  } else {
    print("please check to see if validation samples and features match.")
  }
  #assign data and labels using one hot encoding
  validation_labels <- validation_dat$class
  new_validation <- model.matrix(~.+0,data = validation_dat[,-c("class"),with=F])
  colnames(new_validation) <- gsub("\\`","",colnames(new_validation))
  #convert factor to numeric 
  validation_labels <- as.numeric(validation_labels)-1 #note that 0=infected (not protected) and 1 = protected
  #prepare matrix
  dvalidation <- xgb.DMatrix(data = new_validation, label = validation_labels)
  ds_dvalidation <- xgb.DMatrix(data = as.matrix(validation_dat_temp[,-c("class")]), label = validation_labels, nthread = 8)
  xgbpred1 <- predict(xgb2, ds_dvalidation)
  xgbpred1 <- ifelse(xgbpred1 > 0.5,1,0)
  xgbpred1 <- factor(ifelse(xgbpred1 == 1, "protected", "not protected")) #remember that we are going with the og high-dose designation: infected=0,protected=1
  validation_lab_confusion_matrix <- factor(ifelse(validation_labels==1, "protected", "not protected"))
  #save confusion matrix for each into a list
  confusion_mat <- caret::confusionMatrix(xgbpred1, validation_lab_confusion_matrix)
  #run predictions for ROC
  response_labs <- factor(ifelse(as.character(validation_dat$class) == "infected", "not protected", "protected"))
  xgbpred2_prob <- predict(xgb2, ds_dvalidation)
  xgbpred2 <- ifelse(xgbpred2_prob > 0.5,1,0)
  xgbpred2 <- factor(ifelse(xgbpred2 == 1, "protected", "infected")) #remember that we are going with the og high-dose designation: infected=0, protected=1
  roc_data <- cbind(validation_features[,new_train_feat],
                    "pred.prob" = xgbpred2_prob,
                    "predictor" = xgbpred2) %>%
    mutate(class1 = as.integer(class)-1) %>%
    mutate(predictor1 = as.integer(predictor)-1)
  i_wanna_roc <- pROC::roc(data = roc_data, response = "class1", predictor = "predictor1")
  validation_res <- data.frame("Features" = paste(sort(features_ds$Feature), collapse = '; '),
                                    "AUC" = as.numeric(i_wanna_roc$auc),
                                    unlist(t(confusion_mat$overall)))
  newlist <- list("feature_selection_tune_results" = tune_fs_res,
                  "feature_selection_xgb_results" = xgb1,
                  "feature_importance" = feature_importance,
                  "downselected_tune_results" = tune_ds_res,
                  "downselected_xgb_results" = xgb2,
                  "downselected_xgb_lowest_train_error" = xgb2_best_scores,
                  "top_ten_features_per_run" = top_ten_features,
                  "predictor_data" = roc_data,
                  "roc_results" = i_wanna_roc,
                  "confusion_matrix_results" = confusion_mat,
                  "validation_results" = validation_res)
  return(newlist)
}

xgb_train_and_cv_on_selected_features <- function(train_features, validation_features, features_to_test){
  #prepare training set
  train_dat <- train_features %>%
    t() %>%
    data.frame(check.names = FALSE) %>%
    t() %>%
    data.frame(check.names = FALSE) %>%
    mutate(class = factor(class)) %>%
    mutate_at(c(2:ncol(.)), as.numeric) 
  #convert data frame to data table and preserve rownames
  setDT(train_dat, keep.rownames = TRUE) 
  train_dat_samplenames <- train_dat$rn
  train_dat <- train_dat[,-1]
  #prepare validation set
  validation_dat <- validation_features %>%
    t() %>%
    data.frame(check.names = FALSE) %>%
    t() %>%
    data.frame(check.names = FALSE) %>%
    mutate(class = factor(class)) %>%
    mutate_at(c(2:ncol(.)), as.numeric)
  #convert data frame to data table and preserve rownames
  setDT(validation_dat, keep.rownames = TRUE) 
  validation_dat_samplenames <- validation_dat$rn
  validation_dat <- validation_dat[,-1]
  
  #sanity check
  if(all(colnames(train_dat) == colnames(train_features)) &
     all(train_dat_samplenames == rownames(train_features))){
    print(paste0("training set for run ", i," good to go!"))
  } else {
    print("please check to see if training samples and features match.")
  }
  if(all(colnames(validation_dat) == colnames(validation_features)) &
     all(validation_dat_samplenames == rownames(validation_features))){
    print(paste0("validation set for run ", i," good to go!"))
  } else {
    print("please check to see if validation samples and features match.")
  }

  print(paste0("Training set reduced to ", nrow(train_features), " samples and ", ncol(train_features), " features."))
  print(paste0("Validation set reduced to ", nrow(validation_features), " samples and ", ncol(validation_features), " features."))

  #convert characters to factors
  fact_col <- colnames(train_dat)[sapply(train_dat,is.character)]
  for(k in fact_col) set(train_dat, j=k, value = factor(train_dat))
  for(k in fact_col) set(validation_dat, j=k, value = factor(validation_dat))

  #make dataframe to link original colnames to syntactical colnames; allows you to always map back to original names
  colname_key_df <- data.frame(og_colname = colnames(train_dat),
                               syntactic_colname = make.names(colnames(train_dat), unique = TRUE))
  if(all(colnames(train_dat) == colname_key_df$og_colname) &
     all(colnames(train_dat) == colnames(validation_dat))){
    colnames(train_dat) <- colname_key_df$syntactic_colname  #make colnames syntactic for downstream ML functions
    colnames(validation_dat) <- colname_key_df$syntactic_colname #make colnames syntactic for downstream ML functions
  } else {
    print("names don't match")
  }
  #create tasks
  traintask <- mlr::makeClassifTask(data = as.data.frame(train_dat), target = "class")
  validationtask <- mlr::makeClassifTask(data = as.data.frame(validation_dat), target = "class")

  #do one hot encoding`<br/>
  traintask <- createDummyFeatures (obj = traintask)
  validationtask <- createDummyFeatures (obj = validationtask)

  #assign data and labels using one hot encoding
  train_labels <- train_dat$class
  new_train <- model.matrix(~.+0,data = train_dat[,-c("class"),with=F])
  colnames(new_train) <- gsub("\\`","",colnames(new_train))
  #convert factor to numeric
  train_labels <- as.numeric(train_labels)-1 #note that 0=infected (not protected) and 1 = protected
  #prepare matrix
  dtrain <- xgb.DMatrix(data = new_train, label = train_labels)
  #match and replace synctactic colnames with original names
  colnames(dtrain) <- colname_key_df$og_colname[match(colnames(dtrain), colname_key_df$syntactic_colname)]
  #parameter tuning
  print(paste0("Begin run number ", i, " of ", runs, " total runs."))
  print(paste0("Seed for run and splitting train and validation set was ", run_seed, "."))
  print(paste0("Max iterations for hyperparameter tuning: ", maxiterations))
  print(paste0("Running hyperparameter tuning on run number ", i, " of ", runs, " total runs."))
  tic(msg = paste0("hyperparameter tuning for run ", i))
  #set parameter space
  #https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/
  #https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning
  #create learner
  lrn <- makeLearner("classif.xgboost",
                     objective="binary:logistic",
                     nrounds=1000,
                     early_stopping_rounds = 100,
                     eval_metric="error",
                     predict.type = "response")
  params <- makeParamSet( makeDiscreteParam("booster", values = c("gbtree")),
                          makeIntegerParam("gamma",lower = 0L,upper = 3L),
                          makeIntegerParam("max_depth",lower = 2L,upper = 5L),
                          makeNumericParam("eta",lower = 0.01,upper = 0.2),
                          makeNumericParam("min_child_weight",lower = 0L,upper = 4L),
                          makeNumericParam("subsample",lower = 0.75,upper = 0.9),
                          makeNumericParam("lambda",lower = 0, upper = 1),
                          makeNumericParam("alpha",lower = 0, upper = 1),
                          makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))
  
  #set resampling strategy
  rdesc <- makeResampleDesc(method = "CV",
                            predict = "test",
                            iters = 4,
                            stratify = T)

  #parameter tuning
  print(paste0("Begin  training on downselected features for number ", i, " of ", runs, " total runs."))
  new_train_feat <- features_to_test
  train_dat_temp <- t(train_features[,c("class",new_train_feat)]) %>%
    data.frame(check.names = FALSE) %>%
    t() %>%
    data.frame(check.names = FALSE) %>%
    mutate(class = factor(class)) %>%
    mutate_at(c(2:ncol(.)), as.numeric)
  #convert data frame to data table and preserve rownames
  setDT(train_dat_temp, keep.rownames = TRUE, check.names=FALSE) 
  train_dat_temp_samplenames <- train_dat_temp$rn
  train_dat_temp <- train_dat_temp[,-1] #remove rn column
  #sanity check
  if(all(colnames(train_dat_temp)[-1] == new_train_feat) & #remove class column
     all(train_dat_temp_samplenames == rownames(train_features))){
    print("downselected training set good to go!")
  } else {
    print("please check to see if training samples and features match.")
  }
  #assign data and labels using one hot encoding 
  train_labels <- train_dat_temp$class
  new_train <- model.matrix(~.+0, data = train_dat_temp[,-c("class"),with=F])
  colnames(new_train) <- gsub("\\`","",colnames(new_train))
  #convert factor to numeric 
  train_labels <- as.numeric(train_labels)-1 #note that 0=infected (not protected) and 1 = protected
  #prepare matrix
  dtrain <- xgb.DMatrix(data = new_train, label = train_labels)
  #convert characters to factors
  fact_col <- colnames(train_dat_temp)[sapply(train_dat_temp,is.character)]
  for(k in fact_col) set(train_dat_temp, j=k, value = factor(train_dat_temp[[k]]))
  #make dataframe to link original colnames to syntactical colnames; allows you to always map back to original names
  colname_key_df <- data.frame(og_colname = colnames(train_dat_temp),
                               syntactic_colname = make.names(colnames(train_dat_temp), unique = TRUE))
  if(all(colnames(train_dat_temp) == colname_key_df$og_colname)){
    colnames(train_dat_temp) <- colname_key_df$syntactic_colname
  } else {
    print("names don't match")
  }
  #create tasks
  traintask <- mlr::makeClassifTask(data = as.data.frame(train_dat_temp), target = "class")
  #do one hot encoding`<br/> 
  traintask <- createDummyFeatures (obj = traintask)
  #search strategy
  ctrl <- makeTuneControlRandom(maxit = maxiterations)
  print(paste0("Running hyperparameter tuning on run number ", i, " of ", runs, " total runs."))
  print(paste0("maxiterations for hyperparameter tuning: ", maxiterations))
  tic(msg = paste0("hyperparameter tuning for run ", i))
  tune_ds_res <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params, control = ctrl, show.info = T)
  #set hyperparameters
  lrn_tune_ds <- setHyperPars(lrn, par.vals = tune_ds_res$x)
  best_hyper_pars_ds <- data.frame(iterations = maxiterations, lrn_tune_ds$par.vals)
  tuned_params_ds <- list(booster = best_hyper_pars_ds$booster,
                          objective = best_hyper_pars_ds$objective,
                          eta = best_hyper_pars_ds$eta,
                          gamma = best_hyper_pars_ds$gamma,
                          max_depth = best_hyper_pars_ds$max_depth,
                          min_child_weight = best_hyper_pars_ds$min_child_weight,
                          subsample = best_hyper_pars_ds$subsample,
                          colsample_bytree = best_hyper_pars_ds$colsample_bytree)
  toc()
  #retrain model with cv on best hyperparameters
  print(paste0("Training on best hyperparameters on run number ", i, " of ", runs, " total runs."))
  tic(msg = paste0("training on best hyperparameters for run ", i))
  xgb2 <- xgb.train(params = tuned_params_ds,
                    data = dtrain,
                    nrounds = 100,
                    watchlist = list(train=dtrain),
                    print_every_n = 10,
                    early_stopping_rounds = 20,
                    maximize = F ,
                    eval_metric = "error")
  #get variable importance
  mat2 <- xgb.importance(feature_names = xgb2$feature_names, model = xgb2)
  print(paste0("Top 3 downselected features of run number ", i, " of ", runs, " total runs are ",
               mat2$Feature[1], ", ",
               mat2$Feature[2], ", ",
               mat2$Feature[3], "."))
  #xgb.plot.importance (importance_matrix = mat2[1:20]) 
  features_ds <- as.data.frame(mat2) %>%
    rownames_to_column(var = "rank") %>%
    drop_na()
  xgb2_best_scores <- data.frame("features" = paste(sort(features_ds$Feature), collapse = '; '),
                                 "error" = xgb2$best_score) 
  toc()
  #test on validation set
  validation_dat_temp <- t(validation_features[,c("class", new_train_feat)]) %>%
    data.frame(check.names = FALSE) %>%
    t() %>%
    data.frame(check.names = FALSE) %>%
    mutate(class = factor(class)) %>%
    mutate_at(c(2:ncol(.)), as.numeric)
  #convert data frame to data table and preserve rownames
  setDT(validation_dat_temp, keep.rownames = TRUE, check.names=FALSE)
  validation_dat_temp_samplenames <- validation_dat_temp$rn
  validation_dat_temp <- validation_dat_temp[,-1] #remove rn column
  #sanity check
  if(all(colnames(validation_dat_temp)[-1] == new_train_feat) & #remove class column
     all(validation_dat_temp_samplenames == rownames(validation_features))){
    print(paste0("validation set good to go!"))
  } else {
    print("please check to see if validation samples and features match.")
  }
  #assign data and labels using one hot encoding
  validation_labels <- validation_dat$class
  new_validation <- model.matrix(~.+0,data = validation_dat[,-c("class"),with=F])
  colnames(new_validation) <- gsub("\\`","",colnames(new_validation))
  #convert factor to numeric 
  validation_labels <- as.numeric(validation_labels)-1 #note that 0=infected (not protected) and 1 = protected
  #prepare matrix
  dvalidation <- xgb.DMatrix(data = new_validation, label = validation_labels)
  ds_dvalidation <- xgb.DMatrix(data = as.matrix(validation_dat_temp[,-c("class")]), label = validation_labels, nthread = 8)
  xgbpred1 <- predict(xgb2, ds_dvalidation)
  xgbpred1 <- ifelse(xgbpred1 > 0.5,1,0)
  xgbpred1 <- factor(ifelse(xgbpred1 == 1, "protected", "not protected")) #remember that we are going with the og high-dose designation: infected=0,protected=1
  validation_lab_confusion_matrix <- factor(ifelse(validation_labels==1, "protected", "not protected"))
  #save confusion matrix for each into a list
  confusion_mat <- caret::confusionMatrix(xgbpred1, validation_lab_confusion_matrix)
  #run predictions for ROC
  response_labs <- factor(ifelse(as.character(validation_dat$class) == "infected", "not protected", "protected"))
  xgbpred2_prob <- predict(xgb2, ds_dvalidation)
  xgbpred2 <- ifelse(xgbpred2_prob > 0.5,1,0)
  xgbpred2 <- factor(ifelse(xgbpred2 == 1, "protected", "infected")) #remember that we are going with the og high-dose designation: infected=0, protected=1
  roc_data <- cbind(validation_features[, c("class", new_train_feat)],
                    "pred.prob" = xgbpred2_prob,
                    "predictor" = xgbpred2) %>%
    mutate(class1 = as.integer(class)-1) %>%
    mutate(predictor1 = as.integer(predictor)-1)
  i_wanna_roc <- pROC::roc(data = roc_data, response = "class1", predictor = "predictor1")
  validation_res <- data.frame("Features" = paste(sort(new_train_feat), collapse = '; '),
                               "AUC" = as.numeric(i_wanna_roc$auc),
                               unlist(t(confusion_mat$overall)))
  newlist <- list("downselected_xgb_results" = xgb2,
                  "downselected_xgb_lowest_train_error" = xgb2_best_scores,
                  "predictor_data" = roc_data,
                  "roc_results" = i_wanna_roc,
                  "confusion_matrix_results" = confusion_mat,
                  "validation_results" = validation_res)
  return(newlist)
}

xgb_train_on_selected_features <- function(train_features, features_to_test, maxiterations){
  #prepare training set
  train_dat <- train_features %>%
    t() %>%
    data.frame(check.names = FALSE) %>%
    t() %>%
    data.frame(check.names = FALSE) %>%
    mutate(class = factor(class)) %>%
    mutate_at(c(2:ncol(.)), as.numeric) 
  #convert data frame to data table and preserve rownames
  setDT(train_dat, keep.rownames = TRUE) 
  train_dat_samplenames <- train_dat$rn
  train_dat <- train_dat[,-1]
  
  #sanity check
  if(all(colnames(train_dat) == colnames(train_features)) &
     all(train_dat_samplenames == rownames(train_features))){
    print("training set good to go!")
  } else {
    print("please check to see if training samples and features match.")
  }
  print(paste0("Training set reduced to ", nrow(train_features), " samples and ", ncol(train_features), " features."))
 
  #convert characters to factors
  fact_col <- colnames(train_dat)[sapply(train_dat,is.character)]
  for(k in fact_col) set(train_dat, j=k, value = factor(train_dat))

    #make dataframe to link original colnames to syntactical colnames; allows you to always map back to original names
  colname_key_df <- data.frame(og_colname = colnames(train_dat),
                               syntactic_colname = make.names(colnames(train_dat), unique = TRUE))
  if(all(colnames(train_dat) == colname_key_df$og_colname)){
    colnames(train_dat) <- colname_key_df$syntactic_colname  #make colnames syntactic for downstream ML functions
    } else {
    print("names don't match")
  }
  #create tasks
  traintask <- mlr::makeClassifTask(data = as.data.frame(train_dat), target = "class")
  #do one hot encoding`<br/>
  traintask <- createDummyFeatures (obj = traintask)
  #assign data and labels using one hot encoding
  train_labels <- train_dat$class
  new_train <- model.matrix(~.+0,data = train_dat[,-c("class"),with=F])
  colnames(new_train) <- gsub("\\`","",colnames(new_train))
  #convert factor to numeric
  train_labels <- as.numeric(train_labels)-1 #note that 0=infected (not protected) and 1 = protected
  #prepare matrix
  dtrain <- xgb.DMatrix(data = new_train, label = train_labels)
  #match and replace synctactic colnames with original names
  colnames(dtrain) <- colname_key_df$og_colname[match(colnames(dtrain), colname_key_df$syntactic_colname)]
  #parameter tuning
  print(paste0("Max iterations for hyperparameter tuning: ", maxiterations))
  print("Running hyperparameter tuning.")
  tic(msg = "hyperparameter tuning...")
  #set parameter space
  #https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/
  #https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning
  #create learner
  lrn <- makeLearner("classif.xgboost",
                     objective="binary:logistic",
                     nrounds=1000,
                     early_stopping_rounds = 100,
                     eval_metric="error",
                     predict.type = "response")
  params <- makeParamSet( makeDiscreteParam("booster", values = c("gbtree")),
                          makeIntegerParam("gamma",lower = 0L,upper = 3L),
                          makeIntegerParam("max_depth",lower = 2L,upper = 5L),
                          makeNumericParam("eta",lower = 0.01,upper = 0.2),
                          makeNumericParam("min_child_weight",lower = 0L,upper = 4L),
                          makeNumericParam("subsample",lower = 0.75,upper = 0.9),
                          makeNumericParam("lambda",lower = 0, upper = 1),
                          makeNumericParam("alpha",lower = 0, upper = 1),
                          makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))
  
  #set resampling strategy
  rdesc <- makeResampleDesc(method = "CV",
                            predict = "test",
                            iters = 4,
                            stratify = T)
  
  #parameter tuning
  print("Begin training...")
  new_train_feat <- features_to_test
  train_dat_temp <- t(train_features[,c("class",new_train_feat)]) %>%
    data.frame(check.names = FALSE) %>%
    t() %>%
    data.frame(check.names = FALSE) %>%
    mutate(class = factor(class)) %>%
    mutate_at(c(2:ncol(.)), as.numeric)
  #convert data frame to data table and preserve rownames
  setDT(train_dat_temp, keep.rownames = TRUE, check.names=FALSE) 
  train_dat_temp_samplenames <- train_dat_temp$rn
  train_dat_temp <- train_dat_temp[,-1] #remove rn column
  #sanity check
  if(all(colnames(train_dat_temp)[-1] == new_train_feat) & #remove class column
     all(train_dat_temp_samplenames == rownames(train_features))){
    print("downselected training set good to go!")
  } else {
    print("please check to see if training samples and features match.")
  }
  #assign data and labels using one hot encoding 
  train_labels <- train_dat_temp$class
  new_train <- model.matrix(~.+0, data = train_dat_temp[,-c("class"),with=F])
  colnames(new_train) <- gsub("\\`","",colnames(new_train))
  #convert factor to numeric 
  train_labels <- as.numeric(train_labels)-1 #note that 0=infected (not protected) and 1 = protected
  #prepare matrix
  dtrain <- xgb.DMatrix(data = new_train, label = train_labels)
  #convert characters to factors
  fact_col <- colnames(train_dat_temp)[sapply(train_dat_temp,is.character)]
  for(k in fact_col) set(train_dat_temp, j=k, value = factor(train_dat_temp[[k]]))
  #make dataframe to link original colnames to syntactical colnames; allows you to always map back to original names
  colname_key_df <- data.frame(og_colname = colnames(train_dat_temp),
                               syntactic_colname = make.names(colnames(train_dat_temp), unique = TRUE))
  if(all(colnames(train_dat_temp) == colname_key_df$og_colname)){
    colnames(train_dat_temp) <- colname_key_df$syntactic_colname
  } else {
    print("names don't match")
  }
  #create tasks
  traintask <- mlr::makeClassifTask(data = as.data.frame(train_dat_temp), target = "class")
  #do one hot encoding`<br/> 
  traintask <- createDummyFeatures (obj = traintask)
  #search strategy
  ctrl <- makeTuneControlRandom(maxit = maxiterations)
  print("Running hyperparameter tuning...")
  print(paste0("maxiterations for hyperparameter tuning: ", maxiterations))
  tic(msg = "hyperparameter tuning...")
  tune_ds_res <- tuneParams(learner = lrn, task = traintask, resampling = rdesc, measures = acc, par.set = params, control = ctrl, show.info = T)
  #set hyperparameters
  lrn_tune_ds <- setHyperPars(lrn, par.vals = tune_ds_res$x)
  best_hyper_pars_ds <- data.frame(iterations = maxiterations, lrn_tune_ds$par.vals)
  tuned_params_ds <- list(booster = best_hyper_pars_ds$booster,
                          objective = best_hyper_pars_ds$objective,
                          eta = best_hyper_pars_ds$eta,
                          gamma = best_hyper_pars_ds$gamma,
                          max_depth = best_hyper_pars_ds$max_depth,
                          min_child_weight = best_hyper_pars_ds$min_child_weight,
                          subsample = best_hyper_pars_ds$subsample,
                          colsample_bytree = best_hyper_pars_ds$colsample_bytree)
  toc()
  #retrain model with cv on best hyperparameters
  print("Training on best hyperparameters.")
  tic(msg = "training on best hyperparameters")
  xgb <- xgb.train(params = tuned_params_ds,
                    data = dtrain,
                    nrounds = 100,
                    watchlist = list(train=dtrain),
                    print_every_n = 10,
                    early_stopping_rounds = 20,
                    maximize = F ,
                    eval_metric = "error")
  
  n_features <- length(xgb$feature_names)
  feat_importance <- xgb.importance(feature_names = xgb$feature_names, model = xgb)
  print(paste0("The ", n_features, " downselected features used in the model are ",
               paste(xgb$feature_names, collapse = "; ")))
  toc()
  newlist <- list("xgb_results" = xgb,
                  "feature_importance" = feat_importance)
  return(newlist)
}