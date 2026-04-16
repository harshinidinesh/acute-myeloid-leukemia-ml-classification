library(readxl)
library(rsample)
library(ggplot2)
library(stats)
library(infotheo)

set.seed(123)

# Data processing
data <- read_excel("metadata_with_data_dictionary.xlsx")
data_split <- initial_split(data, prop = 0.8, strata = bag_label)
train_data <- training(data_split)
test_data  <- testing(data_split)

# Features
numeric_features <- c("age", "leucocytes_per_¬µl", "pb_myeloblast", "pb_promyelocyte", 
                      "pb_myelocyte", "pb_metamyelocyte", "pb_neutrophil_band", 
                      "pb_neutrophil_segmented", "pb_eosinophil", "pb_basophil", 
                      "pb_monocyte", "pb_lymph_typ", "pb_other")

# Median imputation
for(col in numeric_features) {
  train_median <- median(train_data[[col]], na.rm = TRUE)
  train_data[[col]][is.na(train_data[[col]])] <- train_median
  test_data[[col]][is.na(test_data[[col]])]   <- train_median
}

# Training
# Compute prior probabilities
priors <- table(train_data$bag_label) / nrow(train_data)
subtypes <- names(priors)

# Compute likelihood params
model_params <- list()

for(type in subtypes) {
  subset_df <- train_data[train_data$bag_label == type, ]
  
  # Estimate Gaussian parameters (Mean and SD) for numeric features
  params <- lapply(subset_df[, numeric_features], function(x) {
    list(mean = mean(x), sd = sd(x) + 0.0001) # prevent divide by zero issues
  })
  
  # Estimate categorical distribution for 'sex'
  params$sex <- table(factor(subset_df$sex_1f_2m, levels = c(1, 2))) / nrow(subset_df)
  
  model_params[[type]] <- params
}

# Prediction
predict_aml_bayesian <- function(patient_row, subtypes, priors, params) {
  log_posteriors <- numeric(length(subtypes))
  names(log_posteriors) <- subtypes
  
  for(type in subtypes) {
    log_p <- log(priors[type])
    
    # Add log-likelihoods of numeric features
    for(feat in numeric_features) {
      val <- as.numeric(patient_row[feat])
      mean_val <- params[[type]][[feat]]$mean
      sd_val   <- params[[type]][[feat]]$sd
      log_p <- log_p + dnorm(val, mean = mean_val, sd = sd_val, log = TRUE)
    }
    
    # Add log-likelihood of categorical data
    sex_val <- as.character(patient_row["sex_1f_2m"])
    prob_sex <- params[[type]]$sex[sex_val]
    log_p <- log_p + log(max(prob_sex, 0.001))
    
    log_posteriors[type] <- log_p
  }
  
  # Convert back to probabilities
  probs <- exp(log_posteriors - max(log_posteriors))
  probs <- probs / sum(probs)
  
  return(probs)
}

# Evaluation
# Generate predictions on test set
test_predictions <- t(apply(test_data, 1, function(row) {
  predict_aml_bayesian(row, subtypes, priors, model_params)
}))

# Get results
results <- as.data.frame(test_predictions)
results$predicted_subtype <- colnames(test_predictions)[apply(test_predictions, 1, which.max)]
results$actual_subtype <- test_data$bag_label

# Overall accuracy
accuracy <- mean(results$predicted_subtype == results$actual_subtype)
cat(sprintf("Overall Accuracy: %.2f%%\n", accuracy * 100))

# Class-Level Accuracy
cat("\nAccuracy Per Class:\n")
class_acc <- diag(conf_matrix) / colSums(conf_matrix)
print(round(class_acc, 4))

acc_df <- data.frame(Subtype = names(class_acc), Accuracy = as.numeric(class_acc))

print(ggplot(acc_df, aes(x = reorder(Subtype, Accuracy), y = Accuracy, fill = Subtype)) +
        geom_col(width = 0.6, show.legend = FALSE) +
        geom_text(aes(label = paste0(round(Accuracy * 100, 1), "%")),
                  hjust = -0.15, size = 4) +
        scale_y_continuous(labels = function(x) paste0(round(x * 100), "%"),
                           limits = c(0, 1.15)) +
        scale_fill_manual(values = c("CBFB_MYH11"    = "#8a4fb5",
                                     "control"       = "#3266ad",
                                     "NPM1"          = "#5b9e7a",
                                     "PML_RARA"      = "#c96e3f",
                                     "RUNX1_RUNX1T1" = "#c94f4f")) +
        coord_flip() +
        labs(title = "Classification Accuracy by Subtype", x = NULL, y = "Accuracy") +
        theme_minimal(base_size = 13) +
        theme(panel.grid.major.y = element_blank(),
              plot.title = element_text(face = "bold")))

# Confusion Matrix & Visualization
cat("\nConfusion Matrix:\n")
conf_matrix <- table(Predicted = results$predicted_subtype, Actual = results$actual_subtype)
print(conf_matrix)

cm_df <- as.data.frame(conf_matrix)
names(cm_df) <- c("Predicted", "Actual", "Count")

print(ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Count)) +
        geom_tile(color = "white", linewidth = 0.8) +
        geom_text(aes(label = Count), size = 5, fontface = "bold",
                  color = ifelse(cm_df$Count > max(cm_df$Count) * 0.5, "white", "black")) +
        scale_fill_gradient(low = "#E6F1FB", high = "#185FA5") +
        labs(title = "Confusion Matrix: AML Subtype Prediction", 
             x = "Actual Subtype", y = "Predicted Subtype") +
        theme_minimal(base_size = 13) +
        theme(panel.grid = element_blank(),
              axis.text.x = element_text(angle = 30, hjust = 1),
              plot.title = element_text(face = "bold")))

# Mutual Information
cat("\nMutual Information:\n")

# Calculate MI scores
data_discrete <- discretize(train_data[, numeric_features])
mi_scores <- sapply(numeric_features, function(feat) {
  mutinformation(data_discrete[[feat]], train_data$bag_label)
})

mi_table <- data.frame(
  Feature = names(mi_scores),
  MI_Score = round(mi_scores, 4)
)
mi_table <- mi_table[order(mi_table$MI_Score, decreasing = TRUE), ]
rownames(mi_table) <- NULL

print(mi_table)
print(ggplot(mi_table, aes(x = reorder(Feature, MI_Score), y = MI_Score, fill = MI_Score)) +
        geom_col(width = 0.6, show.legend = FALSE) +
        geom_text(aes(label = round(MI_Score, 3)), hjust = -0.15, size = 3.5) +
        scale_fill_gradient(low = "#B5D4F4", high = "#185FA5") +
        scale_y_continuous(limits = c(0, max(mi_table$MI_Score) * 1.2)) +
        coord_flip() +
        labs(title = "Feature Importance (Mutual Information)",
             subtitle = "Higher values indicate stronger predictors of AML subtype",
             x = NULL, y = "Mutual Information (bits)") +
        theme_minimal(base_size = 13) +
        theme(panel.grid.major.y = element_blank(),
              plot.title = element_text(face = "bold")))