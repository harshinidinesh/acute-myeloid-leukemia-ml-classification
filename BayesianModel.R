library(readxl)
library(rsample)

set.seed(123)

# Data processing
data <- read_excel("metadata_with_data_dictionary.xlsx")

# Create the split object (80% training, 20% testing)
data_split <- initial_split(data, prop = 0.8, strata = bag_label)
train_data <- training(data_split)
test_data  <- testing(data_split)

# Verify the proportions of each class
prop.table(table(train_data$bag_label))
prop.table(table(test_data$bag_label))

numeric_features <- c("age", "leucocytes_per_¬µl", "pb_myeloblast", "pb_promyelocyte", 
                      "pb_myelocyte", "pb_metamyelocyte", "pb_neutrophil_band", 
                      "pb_neutrophil_segmented", "pb_eosinophil", "pb_basophil", 
                      "pb_monocyte", "pb_lymph_typ", "pb_other")

for(col in numeric_features) {
  train_median <- median(train_data[[col]], na.rm = TRUE)
  train_data[[col]][is.na(train_data[[col]])] <- train_median
  test_data[[col]][is.na(test_data[[col]])]   <- train_median
}

# Prior Probabilities
priors <- table(train_data$bag_label) / nrow(train_data)
subtypes <- names(priors)

# Likelihood Parameters (Mean and SD) for each subtype
model_params <- list()

for(type in subtypes) {
  subset_df <- train_data[train_data$bag_label == type, ]
  
  # Store mean and sd for every numeric feature for this subtype
  params <- lapply(subset_df[, numeric_features], function(x) {
    list(mean = mean(x), sd = sd(x) + 0.0001) # Add epsilon to avoid zero SD
  })
  
  # Handle Categorical 'sex' (Probability distribution)
  params$sex <- table(factor(subset_df$sex_1f_2m, levels = c(1, 2))) / nrow(subset_df)
  
  model_params[[type]] <- params
}

predict_aml_bayesian <- function(patient_row, subtypes, priors, params) {
  log_posteriors <- numeric(length(subtypes))
  names(log_posteriors) <- subtypes
  
  for(type in subtypes) {
    
    log_p <- log(priors[type])
    
    for(feat in numeric_features) {
      val <- as.numeric(patient_row[feat])
      mean_val <- params[[type]][[feat]]$mean
      sd_val   <- params[[type]][[feat]]$sd
      log_p <- log_p + dnorm(val, mean = mean_val, sd = sd_val, log = TRUE)
    }
    
    sex_val <- as.character(patient_row["sex_1f_2m"])
    prob_sex <- params[[type]]$sex[sex_val]
    log_p <- log_p + log(max(prob_sex, 0.001))
    
    log_posteriors[type] <- log_p
  }
  
  probs <- exp(log_posteriors - max(log_posteriors))
  probs <- probs / sum(probs)
  
  return(probs)
}

test_predictions <- t(apply(test_data, 1, function(row) {
  predict_aml_bayesian(row, subtypes, priors, model_params)
}))

results <- as.data.frame(test_predictions)
results$predicted_subtype <- colnames(test_predictions)[apply(test_predictions, 1, which.max)]
results$actual_subtype <- test_data$bag_label

head(results[, c("actual_subtype", "predicted_subtype", "control", "NPM1", "PML_RARA")])

# Accuracy
mean(results$predicted_subtype == results$actual_subtype)

# Confusion matrix
conf_matrix <- table(Predicted = results$predicted_subtype, Actual = results$actual_subtype)
print(conf_matrix)

# Accuracy per class
diag(conf_matrix) / colSums(conf_matrix)
