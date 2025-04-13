NYC Taxi Trip Data Analysis & Prediction
📊 Project Overview
This project aims to analyze and model the New York City Taxi Trips dataset. The goal is to explore the data, perform hypothesis testing, engineer relevant features, and build predictive models that can accurately estimate taxi fares and classify fare ranges. The project has both regression and classification objectives.

🎯 Objectives
Predict taxi fare amounts based on trip-related data (regression).

Classify fares into categories (e.g., low, medium, high) to aid in better understanding fare distribution (classification).

Perform exploratory data analysis (EDA) to uncover patterns and relationships within the dataset.

Test hypotheses using statistical methods to validate assumptions.

Engineer and select features that improve model performance.

Preprocess the dataset, handling missing values, outliers, and data types effectively.

🧩 Project Phases
Data Preprocessing

Handle missing values and outliers

Convert and normalize datetime fields

Remove irrelevant or constant columns

Optimize data types for performance

Exploratory Data Analysis (EDA)

Visualize distributions and correlations

Explore spatial and temporal trends

Identify anomalies and seasonality effects

Hypothesis Testing

Compare fare prices across boroughs and zones

Analyze whether time-based features (e.g., day/night, weekday/weekend) affect fares

Feature Engineering

Create distance-related features (e.g., Haversine distance)

Extract datetime features (hour, day of week, month, etc.)

Encode categorical variables

Modeling

Regression models to predict fare amount

Classification models to categorize fares

Evaluate performance using appropriate metrics

Evaluation & Refinement

Tune hyperparameters

Compare different model types

Address data imbalance if needed

Validate results with test sets

🚧 Current Status
Data cleaning and EDA are complete, though the dataset was significantly reduced due to outlier and null value removal.

Some preprocessing issues (e.g., datetime formatting, constant columns) still require refinement.

New features have been engineered and will be tested in upcoming model iterations.

🔮 Next Steps
Refine data preprocessing steps to reduce information loss

Finalize feature selection for modeling

Begin training and evaluating regression and classification models

Optimize and document model results

🛠️ Tools & Technologies
Python (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, SciPy)

Jupyter Notebook

Statistical hypothesis testing

Machine learning algorithms
