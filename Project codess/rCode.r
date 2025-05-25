install.packages("languageserver")
install.packages("tidyverse")    # Data manipulation (dplyr, tidyr, ggplot2, etc.)
install.packages("caret")        # Machine learning toolkit
install.packages("corrplot")     # Correlation visualization
install.packages("Rtsne")        # t-SNE implementation
install.packages("lubridate")    # Date/time handling
install.packages("keras")        # Deep learning (requires TensorFlow backend)
install.packages("reticulate")   # Python interface (for Keras integration)
 


# Libraries
library(tidyverse)
library(caret)
library(corrplot)
library(Rtsne)
library(keras)
library(lubridate)
library(dplyr)

#%% Pre-processing ------------------------------------------------------------
DataLoader <- function(filename, test_size = 0.2, random_state = NULL) {
  tryCatch({
    # Load data
    df <- read_csv(filename)
    
    # Split features and labels
    X <- df %>% select(-total_amount)
    y <- df$total_amount
    
    # Train-test split
    set.seed(random_state)
    train_index <- createDataPartition(y, p = 1 - test_size, list = FALSE)
    
    list(
      data_train = X[train_index, ],
      labels_train = y[train_index],
      data_test = X[-train_index, ],
      labels_test = y[-train_index],
      df = df
    )
  }, error = function(e) {
    message("Error loading data: ", e$message)
    NULL
  })
}

DataParseDates <- function(data_loader) {
  tryCatch({
    parse_safe <- function(dt_col) {
      # parse with quiet=TRUE to avoid warnings, convert bad values to NA
      ymd_hms(dt_col, quiet = TRUE)
    }
    
    # Parse datetimes in train and test sets
    data_loader$data_train <- data_loader$data_train %>%
      mutate(
        tpep_pickup_datetime = parse_safe(tpep_pickup_datetime),
        tpep_dropoff_datetime = parse_safe(tpep_dropoff_datetime)
      )
    
    data_loader$data_test <- data_loader$data_test %>%
      mutate(
        tpep_pickup_datetime = parse_safe(tpep_pickup_datetime),
        tpep_dropoff_datetime = parse_safe(tpep_dropoff_datetime)
      )
    
    message("Datetime columns parsed safely")
    data_loader
  }, error = function(e) {
    message("Datetime parsing error: ", e$message)
    data_loader
  })
}


DataPreprocessing <- function(data_loader, num_cat_features) {
  tryCatch({
    # Identify numerical and categorical features
    num_features <- names(data_loader$data_train)[1:(ncol(data_loader$data_train) - num_cat_features)]
    cat_features <- tail(names(data_loader$data_train), num_cat_features)
    
    # Normalize numerical features
    preproc_num <- preProcess(data_loader$data_train[num_features], method = c("center", "scale"))
    data_loader$data_train[num_features] <- predict(preproc_num, data_loader$data_train[num_features])
    data_loader$data_test[num_features] <- predict(preproc_num, data_loader$data_test[num_features])
    
    # Normalize categorical features
    preproc_cat <- preProcess(data_loader$data_train[cat_features], method = c("range"))
    data_loader$data_train[cat_features] <- predict(preproc_cat, data_loader$data_train[cat_features])
    data_loader$data_test[cat_features] <- predict(preproc_cat, data_loader$data_test[cat_features])
    
    message("Features normalized successfully")
    data_loader
  }, error = function(e) {
    message("Preprocessing error: ", e$message)
    data_loader
  })
}

DataCleaning <- function(data_loader) {
  tryCatch({
    # Remove duplicates
    data_loader$data_train <- distinct(data_loader$data_train)
    
    # Handle missing values
    data_loader$data_train <- na.omit(data_loader$data_train)
    data_loader$data_test <- na.omit(data_loader$data_test)
    
    # Remove outliers using z-score (example for one column)
    remove_outliers <- function(x, threshold = 3) {
      z <- scale(x)
      x[abs(z) > threshold] <- NA
      x
    }

    data_loader$data_train <- data_loader$data_train %>% 
      mutate(across(where(is.numeric), remove_outliers))

    
    message("Data cleaned successfully")
    data_loader
  }, error = function(e) {
    message("Cleaning error: ", e$message)
    data_loader
  })
}

# Usage example:
data_loader <- DataLoader("yellow_tripdata_2019-01/yellow_tripdata_2019-01.csv")
data_loader <- DataParseDates(data_loader)   # <<-- parse datetime columns safely
data_loader <- DataPreprocessing(data_loader, 2)
data_loader <- DataCleaning(data_loader)

#%% EDA ------------------------------------------------------------------------
perform_EDA <- function(data_loader) {
  df <- data_loader$df
  
  # Distribution plot
  print(
    ggplot(df, aes(x = total_amount)) +
      geom_histogram(bins = 30, fill = "blue", alpha = 0.7) +
      xlim(0, 80) +
      ggtitle("Fare Amount Distribution")
  )
  
  # Correlation heatmap
  numeric_df <- df %>% select(where(is.numeric))
  cor_matrix <- cor(numeric_df, use = "complete.obs")
  corrplot(cor_matrix, method = "color", type = "upper")
}

# perform_EDA(data_loader)

#%% Feature Engineering --------------------------------------------------------
feature_engineering <- function(data_loader) {
  df <- data_loader$data_train
  
  # Date features
  df <- df %>%
    mutate(
      tpep_pickup_datetime = ymd_hms(tpep_pickup_datetime),
      tpep_dropoff_datetime = ymd_hms(tpep_dropoff_datetime),
      trip_duration = as.numeric(difftime(tpep_dropoff_datetime, tpep_pickup_datetime, units = "mins")),
      day_of_week = wday(tpep_pickup_datetime),
      pickup_hour = hour(tpep_pickup_datetime),
      is_weekend = ifelse(day_of_week %in% c(6, 7), 1, 0)
    )
  
  # Other features
  df <- df %>%
    mutate(
      distance_per_passenger = trip_distance / passenger_count,
      fare_per_mile = fare_amount / trip_distance,
      fare_per_mile = ifelse(is.infinite(fare_per_mile), 0, fare_per_mile)
    )
  
  data_loader$data_train <- df
  data_loader
}

data_loader <- feature_engineering(data_loader)

perform_EDA(data_loader)


print(summary(data_loader$df))
head(data_loader$df)



library(ggplot2)

ggplot(data_loader$df, aes(x = total_amount)) +
  geom_histogram(bins = 30, fill = "blue", alpha = 0.7) +
  xlim(0, 80) +
  ggtitle("Fare Amount Distribution")


