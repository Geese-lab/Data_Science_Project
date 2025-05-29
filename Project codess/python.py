#%% Libraries
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier, BaggingRegressor, BaggingClassifier
from sklearn.manifold import TSNE 
from sklearn.metrics import ConfusionMatrixDisplay, r2_score, confusion_matrix, make_scorer, mean_squared_error, mean_absolute_error, classification_report, silhouette_score, accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations_with_replacement
from scipy.stats import f_oneway, kruskal, ttest_ind, pearsonr, spearmanr, stats
import pickle
from sklearn.cluster import KMeans, DBSCAN
import warnings
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
warnings.filterwarnings("ignore")




#%% Pre-processing
class DataLoader:
    def __init__(self, filename, test_size=0.2, random_state=None):
        """
        Initializes the DataLoader with the filename of the dataset,
        the proportion of data to include in the test split,
        and the random state for reproducibility.
        """
        self.filename = filename
        self.test_size = test_size
        self.random_state = random_state
        self.data_train = None
        self.labels_train = None
        self.data_test = None
        self.labels_test = None
        self.df = None

        # Load data
        #self._add_target()
        self._load_data()

    """
    def _add_target(self):
        
        Adds the target classification labels
        """


    def _load_data(self):
        """
        Loads the dataset from the specified filename,
        splits it into training and testing sets using train_test_split(),
        and assigns the data and labels to the appropriate attributes.
        """
        try:
            # Load the dataset
            self.df = pd.read_csv(self.filename)

            # Split the data into features and labels
            X = self.df.drop(columns=['total_amount'])
            y = self.df['total_amount']



            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                                random_state=self.random_state)

            # Assign the data and labels to attributes
            self.data_train = X_train
            self.labels_train = y_train
            self.data_test = X_test
            self.labels_test = y_test

            print("Data loaded successfully.")
        except FileNotFoundError:
            print("File not found. Please check the file path.")

    def get_full_data(self):
        """Returns the full dataset as a DataFrame."""
        return self.df

class DataPreprocessing:
    """
    Class responsible for preprocessing the loaded dataset. Need to pass the data with first columns as numerical and the last columns as categorical, then indicate how many categorical features are present in the dataset

    Methods:
        _normalize_features(): Normalizes all features using standard scaling for numerical and min-max scalling for categorical.
    """

    def __init__(self, data_loader, number_categorical_features):
        """
        Initializes the DataPreprocessing class with a DataLoader object.
        """
        self.data_loader = data_loader
        self.number_categorical_features = number_categorical_features

        # Preprocess data
        self._normalize_features()

    def _normalize_features(self):
        """
        Normalizes features using using standard scaling for numerical and min-max scalling for categorical.
        """
        try:
            # Check if data_train and data_test are not None
            if self.data_loader.data_train is None or self.data_loader.data_test is None:
                raise ValueError("Data has not been loaded yet.")
            # Check if labels_train and labels_test are not None
            if self.data_loader.labels_train is None or self.data_loader.labels_test is None:
                raise ValueError("Labels have not been loaded yet.")

            # Identify real numerical features (excluding the last two columns we know are categorical)
            real_numerical_features = self.data_loader.data_train.columns[:-self.number_categorical_features]

            # Normalize real numerical features using StandardScaler
            scaler = StandardScaler()
            self.data_loader.data_train[real_numerical_features] = scaler.fit_transform(
                self.data_loader.data_train[real_numerical_features])
            self.data_loader.data_test[real_numerical_features] = scaler.transform(
                self.data_loader.data_test[real_numerical_features])

            # Identify encoded features
            encoded_features = self.data_loader.data_train.columns[-self.number_categorical_features:]

            # Normalize encoded features using MinMaxScaler
            scaler = MinMaxScaler()
            self.data_loader.data_train[encoded_features] = scaler.fit_transform(
                self.data_loader.data_train[encoded_features])
            self.data_loader.data_test[encoded_features] = scaler.transform(
                self.data_loader.data_test[encoded_features])

            print("Features normalized successfully.")

        except ValueError as ve:
            print("Error:", ve)

class DataCleaning:
    """
    Class for cleaning operations.

    Methods:
        remove_duplicates(): Remove duplicate rows from the dataset.
        handle_missing_values(strategy='mean'): Handle missing values using the specified strategy.
        remove_outliers(threshold=3): Remove outliers from the dataset
    """

    def __init__(self, data_loader):
        """
        Initializes the DataPreprocessing class with a DataLoader object.
        """
        self.data_loader = data_loader

    def remove_duplicates(self):
        """
        Remove duplicate rows from the train dataset.
        """
        try:
            # Check if data and labels are not None
            if self.data_loader.data_train is None:
                raise ValueError("Data has not been loaded yet.")
            if self.data_loader.labels_train is None:
                raise ValueError("Labels have not been loaded yet.")

            # Remove duplicate rows from training data (do not apply to test data)
            self.data_loader.data_train.drop_duplicates(inplace=True)
            self.data_loader.labels_train = self.data_loader.labels_train[self.data_loader.data_train.index]

            print("Duplicate rows removed from training data.")

        except ValueError as ve:
            print("Error:", ve)

    def handle_missing_values(self, strategy='drop'):
        """
        Handle missing values using the specified strategy.

        Parameters:
            strategy (str): The strategy to handle missing values ('mean', 'median', 'most_frequent', or a constant value).
        """
        try:
            # Check if data is not None
            if self.data_loader.data_train is None or self.data_loader.data_test is None:
                raise ValueError("Data has not been loaded yet.")

            # Check if there are missing values
            if self.data_loader.data_train.isnull().sum().sum() == 0 and self.data_loader.data_test.isnull().sum().sum() == 0:
                print("No missing values found in the data.")
                return

            # Handle missing values based on the specified strategy
            if strategy == 'mean':
                self.data_loader.data_train.fillna(self.data_loader.data_train.mean(), inplace=True)
                self.data_loader.data_test.fillna(self.data_loader.data_test.mean(), inplace=True)
            elif strategy == 'median':
                self.data_loader.data_train.fillna(self.data_loader.data_train.median(), inplace=True)
                self.data_loader.data_test.fillna(self.data_loader.data_test.median(), inplace=True)
            elif strategy == 'most_frequent':
                self.data_loader.data_train.fillna(self.data_loader.data_train.mode().iloc[0], inplace=True)
                self.data_loader.data_test.fillna(self.data_loader.data_test.mode().iloc[0], inplace=True)
            elif strategy == 'fill_nan':
                self.data_loader.data_train.fillna(strategy, inplace=True)
                self.data_loader.data_test.fillna(strategy, inplace=True)
            elif strategy == 'drop':
                self.data_loader.data_train = self.data_loader.data_train.dropna(axis=0)
                self.data_loader.labels_train = self.data_loader.labels_train[self.data_loader.data_train.index]
                self.data_loader.data_test = self.data_loader.data_test.dropna(axis=0)
                self.data_loader.labels_test = self.data_loader.labels_test[self.data_loader.data_test.index]

            else:
                raise ValueError("Invalid strategy.")
            print("Missing values handled using strategy:", strategy)

        except ValueError as ve:
            print("Error:", ve)

    def _detect_outliers(self, threshold=4):
        """
        Detect outliers in numerical features using z-score method.

        Parameters:
            threshold (float): The threshold value for determining outliers.

        Returns:
            outliers (DataFrame): DataFrame containing the outliers.
        """
        try:
            # Check if test data is not None
            if self.data_loader.data_train is None:
                raise ValueError("Data has not been loaded yet.")

            # Identify numerical features
            numerical_features = self.data_loader.data_train.select_dtypes(include=['number'])

            # Calculate z-scores for numerical features
            z_scores = (numerical_features - numerical_features.mean()) / numerical_features.std()

            # Find outliers based on threshold
            outliers = self.data_loader.data_train[(z_scores.abs() > threshold).any(axis=1)]

            return outliers

        except ValueError as ve:
            print("Error:", ve)

    def remove_outliers(self, threshold=2):
        """
        Remove outliers from the dataset using z-score method.

        Parameters:
            threshold (float): The threshold value for determining outliers.
        """
        try:
            # Check if data_loader.data is not None
            if self.data_loader.data_train is None:
                raise ValueError("Data has not been loaded yet.")

            # Detect outliers
            outliers = self._detect_outliers(threshold)

            # Remove outliers from the dataset
            self.data_loader.data_train = self.data_loader.data_train.drop(outliers.index)
            self.data_loader.labels_train = self.data_loader.labels_train[self.data_loader.data_train.index]

            print("Outliers removed from the dataset.")

        except ValueError as ve:
            print("Error:", ve)

    def remove_unnamed_columns(self):
        """
        Remove columns that start with 'Unnamed' from training and test datasets.
        """
        try:
            if self.data_loader.data_train is None or self.data_loader.data_test is None:
                raise ValueError("Data has not been loaded yet.")

            self.data_loader.data_train = self.data_loader.data_train.loc[:, ~self.data_loader.data_train.columns.str.startswith('Unnamed')]
            self.data_loader.data_test = self.data_loader.data_test.loc[:, ~self.data_loader.data_test.columns.str.startswith('Unnamed')]

            print("Unnamed columns removed from datasets.")

        except ValueError as ve:
            print("Error:", ve)

    def remove_unnecessary_columns(self, columns_to_remove):
        """
        Remove specified columns from train and test data.
        """
        try:
            for col in columns_to_remove:
                if col in self.data_loader.data_train.columns:
                    self.data_loader.data_train.drop(columns=col, inplace=True)
                if col in self.data_loader.data_test.columns:
                    self.data_loader.data_test.drop(columns=col, inplace=True)
            print(f"Removed columns: {columns_to_remove}")
        except Exception as e:
            print("Error while removing unnecessary columns:", e)




data_loader = DataLoader("yellow_tripdata_2019-01\yellow_tripdata_2019-01.csv")
data_preprocessing = DataPreprocessing(data_loader, 2)
# Access the data and labels attributes
print("\n\nBefore data preprocessing")
print("Training data shape:", data_loader.data_train.shape)
print("Training labels shape:", data_loader.labels_train.shape)
print("Testing data shape:", data_loader.data_test.shape)
print("Testing labels shape:", data_loader.labels_test.shape)

data_cleaner = DataCleaning(data_loader)
print("Rows remaining after [step]:", len(data_loader.data_train))
data_cleaner.remove_duplicates()
print("Rows remaining after [step]:", len(data_loader.data_train))
data_cleaner.handle_missing_values()
print("Rows remaining after [step]:", len(data_loader.data_train))
data_cleaner.remove_outliers()
print("Rows remaining after [step]:", len(data_loader.data_train))
data_cleaner.remove_unnamed_columns()
print("Rows remaining after [step]:", len(data_loader.data_train))
data_cleaner.remove_unnecessary_columns(["Unnamed: 17"])
print("Rows remaining after [step]:", len(data_loader.data_train))


print("\n\nAfter data preprocessing")
print("Training data shape:", data_loader.data_train.shape)
print("Training labels shape:", data_loader.labels_train.shape)
print("Testing data shape:", data_loader.data_test.shape)
print("Testing labels shape:", data_loader.labels_test.shape)


# Serialize data_loader object to save a copy of the cleaned data
with open('data_loader.pkl', 'wb') as f:
    pickle.dump(data_loader, f)

# Deserialize data_loader object to restore the cleaned data
with open('data_loader.pkl', 'rb') as f:
    data_loader_loaded = pickle.load(f)
print("\n\nDeserialized data")
print("Training data shape:", data_loader_loaded.data_train.shape)
print("Training labels shape:", data_loader_loaded.labels_train.shape)
print("Testing data shape:", data_loader_loaded.data_test.shape)
print("Testing labels shape:", data_loader_loaded.labels_test.shape)


# EDA 
class EDA:
    """
    A class responsible for exploratory data analysis (EDA).

    Attributes:
        data_loader (DataLoader): An object of the DataLoader class containing the dataset.

    Methods:
        perform_eda(): Performs exploratory data analysis.
        plot_distributions(): Plots distributions of the features.
        plot_correlation_heatmap(): Plots a correlation heatmap between features and labels.
        plot_feature_importance(): Computes and visualizes feature importance using permutation importance.
    """

    def __init__(self, data_loader):
        """
        Initializes the EDA class with a DataLoader object.
        """
        self.data_loader = data_loader

    def perform_eda(self):
        """
        Performs exploratory data analysis.
        """
        print("Exploratory Data Analysis (EDA) Report:")
        print("--------------------------------------")

        # Summary statistics
        print("\nSummary Statistics for train data:")
        print(self.data_loader.data_train.describe())
        print("\nSummary Statistics for test data:")
        print(self.data_loader.data_test.describe())

        # Distribution analysis
        self.plot_distributions()
        # Correlation analysis
        self.plot_correlation_heatmap()
        # Outliers by using IQR
        print(self.detect_outliers_iqr())
        
        #Dimensionality Reduction
        self.apply_pca() # Linear
        #self.apply_tsne() # Non-Linear

        # Grid-based visualizations
        print("\nPlotting distributions in grid layout...")
        self.plot_boxplots_grid()
        self.plot_violinplots_grid()
        self.plot_histograms_grid()
        self.plot_ecdf_grid()


    def plot_distributions(self):
        """
        Plots distributions of the features.
        """
        df = self.data_loader.get_full_data()
        filtered_df = df[df['total_amount'] <= 80]
        plt.figure(figsize=(10, 4))
        sns.histplot(filtered_df['total_amount'], bins='auto', kde=True)
        plt.xlim(0, 80)
        plt.title("Fare Amount Distribution")
        plt.xlabel("Total Amount")
        plt.ylabel("Frequency")
        plt.show()

        print(df['total_amount'].describe())
        print(df['total_amount'].value_counts().head(10))  # see common values



    def plot_correlation_heatmap(self):
        """
        Plots a correlation heatmap between features and labels.
        """
        df = self.data_loader.get_full_data()
        df_numeric = df.select_dtypes(include=['number'])
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            df_numeric.corr(), 
            annot=True, 
            cmap='coolwarm', 
            fmt='.2f'
            # annot_kws={"size": 12}
        )
        plt.title("Feature Correlation", fontsize=14)
        # plt.xticks(fontsize=6)
        # plt.yticks(fontsize=6)
        
        plt.show()

    def detect_outliers_iqr(self):
        """
        Detects outliers using the Interquartile Range (IQR) method.

        :param df: DataFrame containing the dataset.
        :param features: List of numerical columns to check for outliers.
        :return: Dictionary where keys are column names and values are indices of detected outliers.
        """
        outliers = {}

        df = self.data_loader.get_full_data()

        df_numeric = df.select_dtypes(include=['number'])

        for feature in df_numeric.columns:
            Q1 = df_numeric[feature].quantile(0.25)
            Q3 = df_numeric[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_indices = df_numeric[(df_numeric[feature] < lower_bound) | 
                                         (df_numeric[feature] > upper_bound)].index.tolist()
            outliers[feature] = outlier_indices

        return outliers
    
    def apply_pca(self):
        """
        Applies PCA (Linear Dimensionality Reduction) with NaN handling and better visualization.
        """
        df = self.data_loader.get_full_data().select_dtypes(include=[np.number])

        # Handle NaN values (choose one)
        df.fillna(df.mean(), inplace=True)  # Option 1: Replace NaN with mean
        # df.dropna(inplace=True)  # Option 2: Drop rows with NaN

        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(df_scaled)

        # Convert PCA result to DataFrame
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

        # Filter to remove extreme outliers from visualization (not from data)
        filtered = pca_df[(pca_df["PC1"] <= 100) & (pca_df["PC2"] >= -100)]

        # Plot filtered data
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=filtered["PC1"], y=filtered["PC2"])
        plt.title("PCA")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.show()

    def apply_tsne(self):
        """
        Applies t-SNE (Nonlinear Dimensionality Reduction).
        """
        df = self.data_loader.get_full_data().select_dtypes(include=[np.number])

        # Handle NaN values
        df.fillna(df.mean(), inplace=True)

        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)

        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(df_scaled)

        plt.figure(figsize=(8,6))
        sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1])
        plt.title("t-SNE: Nonlinear Dimensionality Reduction")
        plt.show()

    # ===== NEW VISUALIZATION METHODS =====
    
    def plot_boxplots_grid(self, show_outliers=False):
        """
        Plots boxplots for all numerical features in grid layout.
        """
        df = self.data_loader.get_full_data()
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows), constrained_layout=True)
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            # Calculate IQR bounds if not showing outliers
            if not show_outliers:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                plot_data = df[col][(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            else:
                plot_data = df[col]
            
            # Create plot
            sns.boxplot(y=plot_data, ax=axes[i], color='skyblue', width=0.5, showfliers=show_outliers)
            
            # Configure plot appearance
            axes[i].set_title(f'Boxplot of {col}', fontsize=12, pad=15)
            axes[i].set_ylabel('', labelpad=10)
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
            
            # Set limits if not showing outliers
            if not show_outliers and len(plot_data) > 0:
                axes[i].set_ylim(plot_data.min() * 0.95, plot_data.max() * 1.05)
        
        # Hide empty subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
            
        plt.show()

    def plot_violinplots_grid(self, show_outliers=False):
        """
        Plots violin plots for all numerical features in grid layout.
        """
        df = self.data_loader.get_full_data()
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows), constrained_layout=True)
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            # Filter outliers if needed
            if not show_outliers:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                plot_data = df[col][(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            else:
                plot_data = df[col]
            
            sns.violinplot(y=plot_data, ax=axes[i], color='lightgreen', width=0.8)
            
            # Configure plot appearance
            axes[i].set_title(f'Violin Plot of {col}', fontsize=12, pad=15)
            axes[i].set_ylabel('', labelpad=10)
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
            
            if not show_outliers and len(plot_data) > 0:
                axes[i].set_ylim(plot_data.min() * 0.95, plot_data.max() * 1.05)
        
        # Hide empty subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
            
        plt.show()

    def plot_histograms_grid(self, show_outliers=False):
        """
        Plots histograms for all numerical features in grid layout.
        """
        df = self.data_loader.get_full_data()
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows), constrained_layout=True)
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            # Filter outliers if needed
            if not show_outliers:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                plot_data = df[col][(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            else:
                plot_data = df[col]
            
            sns.histplot(plot_data, ax=axes[i], kde=True, color='salmon', bins=30)
            
            # Configure plot appearance
            axes[i].set_title(f'Distribution of {col}', fontsize=12, pad=15)
            axes[i].set_xlabel('', labelpad=10)
            axes[i].grid(linestyle='--', alpha=0.7)
            
            if not show_outliers and len(plot_data) > 0:
                axes[i].set_xlim(plot_data.min() * 0.95, plot_data.max() * 1.05)
        
        # Hide empty subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
            
        plt.show()

    def plot_ecdf_grid(self):
        """
        Plots ECDF for all numerical features in grid layout.
        """
        df = self.data_loader.get_full_data()
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows), constrained_layout=True)
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            sns.ecdfplot(df[col], ax=axes[i], color='purple')
            
            # Configure plot appearance
            axes[i].set_title(f'ECDF of {col}', fontsize=12, pad=15)
            axes[i].set_xlabel('', labelpad=10)
            axes[i].grid(linestyle='--', alpha=0.7)
        
        # Hide empty subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
            
        plt.show()


eda = EDA(data_loader)
eda.perform_eda()














class HypothesisTesting:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def check_label_type(self):
        """
        Checks if labels are categorical or continuous.
        """
        unique_values = self.data_loader.labels_train.nunique()
        print(f"Unique labels: {unique_values}")

        if unique_values > 20:  # Assume continuous if there are many unique values
            return "continuous"
        else:
            return "categorical"

    def group_rare_labels(self, min_samples=5):
        """
        Groups rare labels into an 'Other' category.
        """
        label_counts = self.data_loader.labels_train.value_counts()
        self.data_loader.labels_train = self.data_loader.labels_train.apply(
            lambda x: x if label_counts[x] >= min_samples else "Other"
        )

    def _perform_anova_test(self, feature):
        """
        Perform an analysis of variance (ANOVA) test for the given feature and multiclass target variable.
        """
        # Step 1: Load the full dataset and select numeric columns
        df = self.data_loader.get_full_data().select_dtypes(include=[np.number])

        # Check if the given feature exists in the DataFrame
        if feature not in df.columns:
            print(f"Error: The feature '{feature}' is not present in the dataset.")
            return None, None

        # Handle missing values in the feature (you can also choose to drop rows or fill with a value)
        df.dropna(subset=[feature], inplace=True)

        # Step 2: Group data by 'RatecodeID' and extract the feature values for each group
        groups = [df[df['RatecodeID'] == ratecode][feature] for ratecode in df['RatecodeID'].unique()]

        # Step 3: Check if there is more than one non-empty group
        groups = [group for group in groups if len(group) > 0]

        if len(groups) < 2:
            print(f"Error: Not enough groups for ANOVA test for {feature}. Skipping the test.")
            return None, None

        # Step 4: Perform the ANOVA test
        f_statistic, p_value = stats.f_oneway(*groups)

        # Step 5: Interpret the result
        print(f"ANOVA results for {feature}: F-statistic = {f_statistic}, p-value = {p_value}")
        if p_value < 0.05:
            print("There is a significant difference between the groups.")
            return p_value, True
        else:
            print("There is no significant difference between the groups.")
            return p_value, False


    def _perform_kruskal_test(self, feature):
        """
        Perform a Kruskal-Wallis test for the given feature and multiclass target variable.
        """
        # Step 1: Load the full dataset and select numeric columns
        df = self.data_loader.get_full_data().select_dtypes(include=[np.number])

        # Check if the given feature exists in the DataFrame
        if feature not in df.columns:
            print(f"Error: The feature '{feature}' is not present in the dataset.")
            return None, None

        # Handle missing values in the feature (you can also choose to drop rows or fill with a value)
        df.dropna(subset=[feature], inplace=True)

        # Step 2: Group data by 'RatecodeID' and extract the feature values for each group
        groups = [df[df['RatecodeID'] == ratecode][feature] for ratecode in df['RatecodeID'].unique()]

        # Step 3: Filter out any empty groups
        non_empty_groups = [group for group in groups if len(group) > 0]

        if len(non_empty_groups) < 2:
            print(f"Error: Need at least two non-empty groups to perform the Kruskal-Wallis test for {feature}.")
            return None, None

        # Step 4: Perform the Kruskal-Wallis test
        h_statistic, p_value = stats.kruskal(*non_empty_groups)

        # Step 5: Interpret the result
        print(f"Kruskal-Wallis results for {feature}: H-statistic = {h_statistic}, p-value = {p_value}")
        if p_value < 0.05:
            print("There is a significant difference between the groups.")
            return p_value, True
        else:
            print("There is no significant difference between the groups.")
            return p_value, False

    def _perform_correlation(self, feature):
        """
        Uses Pearson or Spearman correlation for continuous labels.
        """
        print(f"\nFeature: {feature}")

        x = self.data_loader.data_train[feature].dropna()
        y = self.data_loader.labels_train.loc[x.index]

        if len(x) < 3:  # Correlation needs at least 3 points
            print("Skipping correlation: Not enough data points.")
            return np.nan, False

        pearson_corr, pearson_p = pearsonr(x, y)
        spearman_corr, spearman_p = spearmanr(x, y)

        print(f"Pearson: {pearson_corr:.3f} (p={pearson_p:.3f}), Spearman: {spearman_corr:.3f} (p={spearman_p:.3f})")
        return pearson_p, pearson_p < 0.05

    def run_tests(self):
        """
        Runs the appropriate test for each feature.
        """
        label_type = self.check_label_type()

        if label_type == "categorical":
            print("Labels are categorical. Running ANOVA and Kruskal-Wallis tests.")
            self.group_rare_labels()

            for feature in self.data_loader.data_train.columns:
                p_value, significant = self._perform_anova_test(feature)
                print(f"Feature: {feature}, ANOVA p-value: {p_value}, Significant: {significant}")

                p_value, significant = self._perform_kruskal_test(feature)
                print(f"Feature: {feature}, Kruskal p-value: {p_value}, Significant: {significant}")

        else:
            print("Labels are continuous. Running Pearson and Spearman correlation tests.")
            numeric_features = self.data_loader.data_train.select_dtypes(include=[np.number]).columns
            for feature in numeric_features:
                p_value, significant = self._perform_correlation(feature)
                print(f"Feature: {feature}, Correlation p-value: {p_value}, Significant: {significant}")
ht = HypothesisTesting(data_loader)
ht.run_tests()















class FeatureEngineering():
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.X_train = self.data_loader.data_train.copy()
        self.y_train = self.data_loader.labels_train.copy()
        self.X_test = self.data_loader.data_test.copy()
        self.y_test = self.data_loader.labels_test.copy()
        self.feature_engineering()

    def feature_engineering(self):
        combined = self.X_train.copy()
        combined['total_amount'] = self.y_train
        combined.dropna(subset=['total_amount'], inplace=True)

        self.X_train = combined.drop(columns=['total_amount'])
        self.y_train = combined['total_amount']
        self.X_train['total_amount'] = self.y_train

        self.X_train['total_amount'] = pd.to_numeric(self.X_train['total_amount'], errors='coerce')
        self.X_train['passenger_count'] = pd.to_numeric(self.X_train['passenger_count'], errors='coerce')

        print(self.X_train.columns)
        
        """Feature engineering: Produce at least 10 new features."""
        # Example of feature engineering (you can customize this based on your dataset)
        self.X_train['tpep_pickup_datetime'] = pd.to_datetime(self.X_train['tpep_pickup_datetime'], dayfirst=True)
        self.X_train['tpep_dropoff_datetime'] = pd.to_datetime(self.X_train['tpep_dropoff_datetime'], dayfirst=True)

        # 1. Trip duration (in minutes)
        self.X_train['trip duration'] = (self.X_train['tpep_dropoff_datetime'] - self.X_train['tpep_pickup_datetime']).dt.total_seconds() / 60

        # 2. Day of the week (0: Monday, 6: Sunday)
        self.X_train['day_of_week'] = self.X_train['tpep_pickup_datetime'].dt.dayofweek

        # 3. Hour of the day (0 to 23)
        self.X_train['pickup_hour'] = self.X_train['tpep_pickup_datetime'].dt.hour

        # 4. Weekend flag (1 if weekend, 0 if weekday)
        self.X_train['is_weekend'] = self.X_train['day_of_week'].isin([5, 6]).astype(int)

        # 5. Distance per passenger
        self.X_train['distance_per_passenger'] = self.X_train['trip_distance'] / self.X_train['passenger_count']

        # 6. Fare per mile
        self.X_train['fare_per_mile'] = self.X_train['fare_amount'] / self.X_train['trip_distance']
        self.X_train['fare_per_mile'].replace([np.inf, -np.inf], 0, inplace=True)  # Handle division by zero

        # 7. Total charges per passenger
        self.X_train['total_per_passenger'] = self.X_train['total_amount'] / self.X_train['passenger_count']
        self.X_train['total_per_passenger'].replace([np.inf, -np.inf], 0, inplace=True)
        self.X_train['total_per_passenger'].fillna(0, inplace=True)

        
        # 8. Time of day category
        def time_of_day(hour):
            if 5 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 21:
                return 'Evening'
            else:
                return 'Night'
        
        self.X_train['time_of_day'] = self.X_train['pickup_hour'].apply(time_of_day)

        # 9. Tip flag (1 if tip > 0, 0 otherwise)
        self.X_train['tip_flag'] = (self.X_train['tip_amount'] > 0).astype(int)

        # 10. Average fare per trip (based on passenger count)
        avg_fare = self.X_train.groupby('passenger_count')['fare_amount'].mean().reset_index()
        avg_fare.columns = ['passenger_count', 'avg_fare_per_trip']
        self.X_train = pd.merge(self.X_train, avg_fare, on='passenger_count', how='left')

        # Geographic features
        self.X_train['is_airport'] = ((self.X_train['PULocationID'].isin([1, 132]))) | ((self.X_train['DOLocationID'].isin([1, 132]))).astype(int)

feature_engineering = FeatureEngineering(data_loader)
feature_engineering.feature_engineering()
print(feature_engineering.X_train.head())
















class DataLoader:
    def __init__(self, filename, test_size=0.2, random_state=None, task='regression'):
        self.filename = filename
        self.test_size = test_size
        self.random_state = random_state
        self.task = task
        self.data_train = None
        self.labels_train = None
        self.data_test = None
        self.labels_test = None
        self.df = None
        self.scaler = None
        self._load_data()

    def _add_target(self):
        bins = [-np.inf, 10, 30, 60, np.inf]
        labels = [1, 2, 3, 4]
        self.df['fare_class'] = pd.cut(self.df['fare_amount'], bins=bins, labels=labels)

    def _preprocess_features(self, X):
        # Convert datetime
        X['tpep_pickup_datetime'] = pd.to_datetime(X['tpep_pickup_datetime'])
        X['tpep_dropoff_datetime'] = pd.to_datetime(X['tpep_dropoff_datetime'])
        X['pickup_hour'] = X['tpep_pickup_datetime'].dt.hour
        X['pickup_day'] = X['tpep_pickup_datetime'].dt.dayofweek
        X['trip_duration'] = (X['tpep_dropoff_datetime'] - X['tpep_pickup_datetime']).dt.total_seconds() / 60
        X = X.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

        # Encode categorical
        X['store_and_fwd_flag'] = X['store_and_fwd_flag'].map({'Y': 1, 'N': 0})
        le = LabelEncoder()
        for col in ['PULocationID', 'DOLocationID']:
            X[col] = le.fit_transform(X[col])
        X = pd.get_dummies(X, columns=['RatecodeID', 'payment_type'])

        # Scale features
        numerical = ['passenger_count', 'trip_distance', 'pickup_hour', 'pickup_day', 'trip_duration']
        if self.scaler:
            X[numerical] = self.scaler.transform(X[numerical])
        else:
            self.scaler = StandardScaler().fit(X[numerical])
            X[numerical] = self.scaler.transform(X[numerical])
            # Ensure all columns are float64
        X = X.astype(np.float64)
        return X
        


    def _load_data(self):
        try:
            self.df = pd.read_csv(self.filename)
            if self.task == 'classification':
                self._add_target()

            # Drop columns based on task
            drop_cols = ['total_amount', 'extra', 'mta_tax', 'tip_amount', 
                          'tolls_amount', 'improvement_surcharge', 'congestion_surcharge']
            target = 'fare_amount' if self.task == 'regression' else 'fare_class'
            X = self.df.drop(columns=drop_cols + [target])
            y = self.df[target]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

            # Preprocess
            self.data_train = self._preprocess_features(X_train)
            self.data_test = self._preprocess_features(X_test)
            self.labels_train = y_train.values
            self.labels_test = y_test.values

            print("Data loaded successfully.")
        except FileNotFoundError:
            print("File not found.")
class KNNRegressor:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train.to_numpy().astype(np.float64)
        self.y_train = y_train.astype(np.float64)

    def _euclidean_distance(self, x):
        return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

    def predict(self, X_test):
        X_test = X_test.to_numpy().astype(np.float64)        
        preds = []
        for x in X_test:
            dists = self._euclidean_distance(x)
            k_indices = np.argpartition(dists, self.k)[:self.k]
            preds.append(np.mean(self.y_train[k_indices]))
        return np.array(preds)
    
    def evaluate(self, X_test, y_test, plot=True):
        """Evaluate model performance with optional plots"""
        y_pred = self.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Mean Absolute Error: ${mae:.2f}")
        print(f"R-squared Score: {r2:.3f}")
        
        if plot:
            plt.figure(figsize=(12, 5))
            
            # Actual vs Predicted plot
            plt.subplot(1, 2, 1)
            sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
            plt.xlabel('Actual Fare Amount ($)')
            plt.ylabel('Predicted Fare Amount ($)')
            plt.title('Actual vs Predicted Fares')
            
            # Error distribution plot
            plt.subplot(1, 2, 2)
            errors = y_test - y_pred
            sns.histplot(errors, kde=True, bins=30)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.xlabel('Prediction Error ($)')
            plt.title('Error Distribution')
            
            plt.tight_layout()
            plt.show()
            
        return {'mae': mae, 'r2': r2}
    
class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train.to_numpy().astype(np.float64)
        self.y_train = y_train

    def _euclidean_distance(self, x):
        return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

    def predict(self, X_test):
        X_test = X_test.to_numpy().astype(np.float64)
        preds = []
        for x in X_test:
            dists = self._euclidean_distance(x)
            k_indices = np.argpartition(dists, self.k)[:self.k]
            k_labels = self.y_train[k_indices]
            preds.append(Counter(k_labels).most_common(1)[0][0])
        return np.array(preds)
    
    def evaluate(self, X_test, y_test, plot=True):
        """Evaluate model performance with optional plots"""
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        
        print(f"Accuracy: {accuracy*100:.1f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=[
            "Class 1 (<$10)", "Class 2 ($10-$30)", 
            "Class 3 ($30-$60)", "Class 4 (>$60)"
        ]))
        
        if plot:
            plt.figure(figsize=(12, 5))
            
            # Confusion matrix
            plt.subplot(1, 2, 1)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Class 1', 'Class 2', 'Class 3', 'Class 4'],
                        yticklabels=['Class 1', 'Class 2', 'Class 3', 'Class 4'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            
            # Class distribution comparison
            plt.subplot(1, 2, 2)
            class_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4']
            actual_counts = [sum(y_test == i) for i in range(1,5)]
            pred_counts = [sum(y_pred == i) for i in range(1,5)]
            
            x = np.arange(len(class_names))
            width = 0.35
            
            plt.bar(x - width/2, actual_counts, width, label='Actual')
            plt.bar(x + width/2, pred_counts, width, label='Predicted')
            
            plt.xlabel('Fare Classes')
            plt.ylabel('Count')
            plt.title('Actual vs Predicted Class Distribution')
            plt.xticks(x, class_names)
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
        return {'accuracy': accuracy}

print("\n=== Regression Task ===")
loader_reg = DataLoader("yellow_tripdata_2019-01\yellow_tripdata_2019-01.csv", task='regression', test_size=0.1)

# Train and predict
knn_reg = KNNRegressor(k=3)
knn_reg.fit(loader_reg.data_train, loader_reg.labels_train)
pred_reg = knn_reg.predict(loader_reg.data_test.iloc[:50])  # Predict on first 50 test samples

reg_results = knn_reg.evaluate(loader_reg.data_test.iloc[:50], loader_reg.labels_test[:50])

# Print predictions vs actual values
print("\nSample Predictions (Regression):")
for i in range(5):  # Show first 5 predictions
    print(f"Predicted: ${pred_reg[i]:.2f} | Actual: ${loader_reg.labels_test[i]:.2f}")

# Calculate Mean Absolute Error (MAE)
mae = np.mean(np.abs(pred_reg - loader_reg.labels_test[:50]))
print(f"\nMean Absolute Error (MAE): ${mae:.2f}")

# --------------------------------------------------
# Classification Task
# --------------------------------------------------
print("\n=== Classification Task ===")
loader_clf = DataLoader("yellow_tripdata_2019-01\yellow_tripdata_2019-01.csv", task='classification', test_size=0.1)

# Train and predict
knn_clf = KNNClassifier(k=3)
knn_clf.fit(loader_clf.data_train, loader_clf.labels_train)
pred_clf = knn_clf.predict(loader_clf.data_test.iloc[:50])  # Predict on first 50 test samples

# Print predictions vs actual classes
print("\nSample Predictions (Classification):")
class_names = {
    1: "Short trip (<$10)",
    2: "Medium trip ($10-$30)",
    3: "Long trip ($30-$60)",
    4: "Premium fare (>$60)"
}
for i in range(5):  # Show first 5 predictions
    print(f"Predicted: {class_names[pred_clf[i]]} | Actual: {class_names[loader_clf.labels_test[i]]}")

clf_results = knn_clf.evaluate(loader_clf.data_test.iloc[:50], loader_clf.labels_test[:50])


# Calculate Accuracy
accuracy = np.mean(pred_clf == loader_clf.labels_test[:50])
print(f"\nAccuracy: {accuracy*100:.1f}%")












#%%
class Supervised_Learning():
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.X_train = self.data_loader.data_train.copy()
        self.y_train = self.data_loader.labels_train.copy()
        self.X_test = self.data_loader.data_test.copy()
        self.y_test = self.data_loader.labels_test.copy()
        self.feature_engineering()
        self.regression_model = None
        self.classification_model = None
        self._create_classification_targets()

    def _create_classification_targets(self):
        """Create binned fare classes for classification task"""
        bins = [-np.inf, 10, 30, 60, np.inf]
        labels = ['Class1', 'Class2', 'Class3', 'Class4']
        self.y_train_clf = pd.cut(self.y_train, bins=bins, labels=labels)
        self.y_test_clf = pd.cut(self.y_test, bins=bins, labels=labels)

    def feature_engineering(self):
        """Enhanced feature engineering with data leakage prevention"""
        # Process both training and test data
        for df in [self.X_train, self.X_test]:
            # Temporal features
            df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
            df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
            
            df['trip_duration'] = (df['tpep_dropoff_datetime'] - 
                                  df['tpep_pickup_datetime']).dt.total_seconds() / 60
            df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
            df['pickup_day'] = df['tpep_pickup_datetime'].dt.day_name()
            
            # Interaction features
            df['distance_per_passenger'] = df['trip_distance'] / (df['passenger_count'] + 1e-6)
            df['speed'] = df['trip_distance'] / (df['trip_duration'] / 60 + 1e-6)
            
            # Time-based features
            df['is_peak_hour'] = df['pickup_hour'].between(7, 9).astype(int)
            df['is_night'] = df['pickup_hour'].between(22, 4).astype(int)
            
            # Geographic features
            df['is_airport'] = ((df['PULocationID'].isin([1, 132]))) | ((df['DOLocationID'].isin([1, 132]))).astype(int)
            
            # Handle missing values properly
            numeric_cols = df.select_dtypes(include=np.number).columns
            cat_cols = df.select_dtypes(exclude=np.number).columns
            
            df[numeric_cols] = df[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)
            df[cat_cols] = df[cat_cols].fillna('Unknown')

    def prepare_data(self, task_type='regression'):
        """Prepare data for modeling"""
        features_to_drop = ['tpep_pickup_datetime', 'tpep_dropoff_datetime']
        X = self.X_train.drop(columns=features_to_drop, errors='ignore')
        X = pd.get_dummies(X)
        
        if task_type == 'regression':
            y = self.y_train
        else:
            y = self.y_train_clf
        
        return X, y

    def train_models(self):
        """Train both regression and classification models"""
        # Regression models
        reg_models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=1, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=1, random_state=42)
        }

        # Classification models
        clf_models = {
            "RandomForestClf": RandomForestClassifier(n_estimators=1, random_state=42),
            "GradientBoostingClf": GradientBoostingClassifier(n_estimators=1, random_state=42)
        }

        # Regression evaluation
        X_reg, y_reg = self.prepare_data('regression')
        reg_results = self._evaluate_models(reg_models, X_reg, y_reg, task_type='regression')

        # Classification evaluation
        X_clf, y_clf = self.prepare_data('classification')
        clf_results = self._evaluate_models(clf_models, X_clf, y_clf, task_type='classification')

        return {'regression': reg_results, 'classification': clf_results}

    def _evaluate_models(self, models, X, y, task_type='regression'):
        """Generic evaluation function"""
        results = {}
        scoring = 'neg_root_mean_squared_error' if task_type == 'regression' else 'accuracy'
        
        for name, model in models.items():
            scores = cross_val_score(model, X, y, scoring=scoring, cv=5)
            results[name] = {
                'mean_score': -np.mean(scores) if task_type == 'regression' else np.mean(scores),
                'std_score': np.std(scores)
            }
        return results

    def final_evaluation(self, model_type='regression', sample_size=20000):
        """Optimized final evaluation with test set sampling"""
        # Common setup
        if model_type == 'regression':
            model = GradientBoostingRegressor(n_estimators=1, random_state=42)
            X, y = self.prepare_data('regression')
            y_true = self.y_test
        else:
            model = GradientBoostingClassifier(n_estimators=1, random_state=42)
            X, y = self.prepare_data('classification')
            y_true = self.y_test_clf

        # Prepare full test set
        X_test_full = self.X_test.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'], errors='ignore')
        X_test_full = pd.get_dummies(X_test_full).reindex(columns=X.columns, fill_value=0)
        X_test_full.fillna(0, inplace=True)

        # Strategic sampling
        if len(X_test_full) > sample_size:
            if model_type == 'classification':
                # Stratified sampling to preserve class distribution
                X_test, _, y_true, _ = train_test_split(
                    X_test_full, y_true, 
                    train_size=sample_size, 
                    stratify=y_true, 
                    random_state=42
                )
            else:
                # Random sampling with noise filtering
                idx = X_test_full.sample(sample_size, random_state=42).index
                X_test = X_test_full.loc[idx]
                y_true = y_true.loc[idx]
        else:
            X_test = X_test_full

        # Train on full data but evaluate on sample
        model.fit(X, y)
        preds = model.predict(X_test)
        
        # Generate evaluation plots
        plt.figure(figsize=(15, 5))
        if model_type == 'regression':
            # Actual vs Predicted values
            plt.subplot(1, 3, 1)
            plt.scatter(y_true, preds, alpha=0.3)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title('Actual vs. Predicted')
            
            # Residuals plot
            residuals = y_true - preds
            plt.subplot(1, 3, 2)
            plt.scatter(preds, residuals, alpha=0.3)
            plt.axhline(0, color='r', linestyle='--')
            plt.xlabel('Predicted')
            plt.ylabel('Residuals')
            plt.title('Residuals vs. Predicted')
            
            # Residual distribution
            plt.subplot(1, 3, 3)
            sns.histplot(residuals, kde=True)
            plt.xlabel('Residuals')
            plt.title('Residual Distribution')
        else:
            # Confusion matrix
            plt.subplot(1, 2, 1)
            cm = confusion_matrix(y_true, preds, labels=model.classes_)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            disp.plot(ax=plt.gca())
            plt.title('Confusion Matrix')
            
            # Feature importances
            if hasattr(model, 'feature_importances_'):
                plt.subplot(1, 2, 2)
                importances = model.feature_importances_
                top_indices = np.argsort(importances)[-10:]
                plt.barh(range(len(top_indices)), importances[top_indices], align='center')
                plt.yticks(range(len(top_indices)), [X.columns[i] for i in top_indices])
                plt.xlabel('Importance')
                plt.title('Top 10 Feature Importances')
        
        plt.tight_layout()
        plt.show()

        # Return metrics based on sampled test set
        if model_type == 'regression':
            return {
                'MAE': mean_absolute_error(y_true, preds),
                'RMSE': np.sqrt(mean_squared_error(y_true, preds))
            }
        else:
            return {
                'Accuracy': accuracy_score(y_true, preds),
                'F1-Score': f1_score(y_true, preds, average='weighted')
            }

supervised_learning = Supervised_Learning(data_loader)
results = supervised_learning.train_models()

print("Regression Results:")
print(pd.DataFrame(results['regression']).T.sort_values("mean_score"))

print("\nClassification Results:")
print(pd.DataFrame(results['classification']).T.sort_values("mean_score"))

print("\nFinal Regression Evaluation:")
print(supervised_learning.final_evaluation('regression',  sample_size=500))

print("\nFinal Classification Evaluation:")
print(supervised_learning.final_evaluation('classification',  sample_size=200))

























class DataLoader:
    def __init__(self, file_path, test_size=0.1, random_state=42):
        # Load and preprocess data
        df = pd.read_csv(file_path)
        
        # Basic preprocessing (adjust based on your actual data)
        df = df[['trip_distance', 'total_amount', 'passenger_count', 
                'PULocationID', 'DOLocationID']].dropna()
        df['fare_amount'] = df['total_amount']  # Assuming fare_amount is our target
        
        # Split features/target
        X = df.drop('fare_amount', axis=1)
        y = df['fare_amount']
        
        # Train-test split
        self.data_train, self.data_test, self.labels_train, self.labels_test = \
            train_test_split(X, y, test_size=test_size, random_state=random_state)
            
    def process_for_classification(self):
        """Convert regression labels to classification bins"""
        bins = [-np.inf, 10, 30, 60, np.inf]
        labels = [1, 2, 3, 4]
        self.labels_train = pd.cut(self.labels_train, bins=bins, labels=labels)
        self.labels_test = pd.cut(self.labels_test, bins=bins, labels=labels)

class BaggingRegressionModel:
    """Predicts continuous fare amounts"""
    def __init__(self, n_estimators=50, max_samples=0.8, max_depth=10, random_state=0):
        base = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
        self.model = BaggingRegressor(
            estimator=base,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, loader):
        X = loader.data_train.select_dtypes(include=[np.number]).fillna(0).to_numpy()
        y = loader.labels_train.to_numpy()
        self.model.fit(X, y)

    def predict(self, loader):
        X_test = loader.data_test.select_dtypes(include=[np.number]).fillna(0).to_numpy()
        return self.model.predict(X_test)

    def evaluate(self, loader):
        preds = self.predict(loader)
        y_true = loader.labels_test.to_numpy()
        return {
            "MAE": mean_absolute_error(y_true, preds),
            "RMSE": np.sqrt(mean_squared_error(y_true, preds))
        }
    
def plot_regression_evaluation(model, loader, model_name):
    """Generates residual and actual vs predicted plots for regression"""
    y_pred = model.predict(loader)
    y_true = loader.labels_test.to_numpy()

    plt.figure(figsize=(12, 5))
    
    # Residual plot
    plt.subplot(1, 2, 1)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs Predicted ({model_name})')
    
    # Actual vs Predicted plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.3)
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted ({model_name})')
    
    plt.tight_layout()
    plt.show()

class BoostingRegressionModel:
    """Predicts continuous fare amounts"""
    def __init__(self, n_estimators=50, learning_rate=0.05, max_depth=5, random_state=0):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )

    def fit(self, loader):
        X = loader.data_train.select_dtypes(include=[np.number]).fillna(0).to_numpy()
        y = loader.labels_train.to_numpy()
        self.model.fit(X, y)

    def predict(self, loader):
        X_test = loader.data_test.select_dtypes(include=[np.number]).fillna(0).to_numpy()
        return self.model.predict(X_test)

    def evaluate(self, loader):
        preds = self.predict(loader)
        y_true = loader.labels_test.to_numpy()
        return {
            "MAE": mean_absolute_error(y_true, preds),
            "RMSE": np.sqrt(mean_squared_error(y_true, preds))
        }
    
def plot_regression_evaluation(model, loader, model_name):
    """Generates residual and actual vs predicted plots for regression"""
    y_pred = model.predict(loader)
    y_true = loader.labels_test.to_numpy()

    plt.figure(figsize=(12, 5))
    
    # Residual plot
    plt.subplot(1, 2, 1)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals vs Predicted ({model_name})')
    
    # Actual vs Predicted plot
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.3)
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted ({model_name})')
    
    plt.tight_layout()
    plt.show()

class BaggingClassificationModel:
    """Classifies fares into 4 categories"""
    def __init__(self, n_estimators=50, max_samples=0.8, max_depth=10, random_state=0):
        base = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        self.model = BaggingClassifier(
            estimator=base,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, loader):
        X = loader.data_train.select_dtypes(include=[np.number]).fillna(0).to_numpy()
        y = loader.labels_train.to_numpy()
        self.model.fit(X, y)

    def predict(self, loader):
        X_test = loader.data_test.select_dtypes(include=[np.number]).fillna(0).to_numpy()
        return self.model.predict(X_test)

    def evaluate(self, loader):
        preds = self.predict(loader)
        y_true = loader.labels_test.to_numpy()
        return {
            "Accuracy": accuracy_score(y_true, preds),
            "F1-Score": f1_score(y_true, preds, average='weighted')
        }


class BoostingClassificationModel:
    """Classifies fares into 4 categories"""
    def __init__(self, n_estimators=50, learning_rate=0.05, max_depth=5, random_state=0):
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )

    def fit(self, loader):
        X = loader.data_train.select_dtypes(include=[np.number]).fillna(0).to_numpy()
        y = loader.labels_train.to_numpy()
        self.model.fit(X, y)

    def predict(self, loader):
        X_test = loader.data_test.select_dtypes(include=[np.number]).fillna(0).to_numpy()
        return self.model.predict(X_test)

    def evaluate(self, loader):
        preds = self.predict(loader)
        y_true = loader.labels_test.to_numpy()
        return {
            "Accuracy": accuracy_score(y_true, preds),
            "F1-Score": f1_score(y_true, preds, average='weighted')
        }
    
def plot_classification_evaluation(model, loader, model_name):
    """Generates confusion matrix for classification"""
    y_pred = model.predict(loader)
    y_true = loader.labels_test.to_numpy()
    
    labels = [1, 2, 3, 4]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix ({model_name})')
    plt.xticks(rotation=45)
    plt.show()


if __name__ == "__main__":
    # Load data once
    loader = DataLoader("yellow_tripdata_2019-01\yellow_tripdata_2019-01.csv")

    print("\n=== REGRESSION TASK ===")
    # Bagging Regression
    bag_reg = BaggingRegressionModel(n_estimators=50)
    bag_reg.fit(loader)
    print("Bagging Regression:", bag_reg.evaluate(loader))  # Existing code
    plot_regression_evaluation(bag_reg, loader, "Bagging Regression")  # New plot
    
    # Boosting Regression
    boost_reg = BoostingRegressionModel(n_estimators=50)
    boost_reg.fit(loader)
    print("Boosting Regression:", boost_reg.evaluate(loader))  # Existing code
    plot_regression_evaluation(boost_reg, loader, "Boosting Regression")  # New plot
    
    print("\n=== CLASSIFICATION TASK ===")
    # Convert labels to classes
    loader.process_for_classification()
    
    # Bagging Classification
    bag_clf = BaggingClassificationModel(n_estimators=50)
    bag_clf.fit(loader)
    print("Bagging Classification:", bag_clf.evaluate(loader))  # Existing code
    plot_classification_evaluation(bag_clf, loader, "Bagging Classification")  # New plot
    
    # Boosting Classification
    boost_clf = BoostingClassificationModel(n_estimators=50)
    boost_clf.fit(loader)
    print("Boosting Classification:", boost_clf.evaluate(loader))  # Existing code
    plot_classification_evaluation(boost_clf, loader, "Boosting Classification")  # New plotBoosting Classification")






























class TaxiFarePredictor:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.regression_model = None
        self.classification_model = None
        self.scaler = None
        self.class_bins = [-np.inf, 10, 30, 60, np.inf]
        self.class_labels = [0, 1, 2, 3]  # Corresponding to your 4 classes
        
    def preprocess_data(self):
        """Preprocess the data for both regression and classification tasks"""
        df = self.data_loader.df.copy()
        
        # Convert datetime to useful features
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
        
        # Extract time features
        df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
        df['pickup_day'] = df['tpep_pickup_datetime'].dt.dayofweek
        df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
        
        # Convert categorical variables
        df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({'N': 0, 'Y': 1})
        
        # Select features
        features = ['passenger_count', 'trip_distance', 'RatecodeID', 'PULocationID', 
                'DOLocationID', 'payment_type', 'pickup_hour', 'pickup_day', 
                'trip_duration', 'extra', 'mta_tax', 'tolls_amount', 
                'improvement_surcharge', 'congestion_surcharge']
        
        # Target variables
        fare_amount = df['fare_amount'].values
        total_amount = df['total_amount'].values
        
        # Create classification target - convert to numeric explicitly
        fare_class = pd.cut(df['fare_amount'], bins=self.class_bins, labels=self.class_labels)
        fare_class = fare_class.astype('int64')  # Convert from category to int
        
        # Verify
        print("Class distribution:")
        print(pd.Series(fare_class).value_counts())

        # Split data
        X_train, X_test, y_reg_train, y_reg_test = train_test_split(
            df[features], fare_amount, test_size=0.2, random_state=42)
        
        _, _, y_cls_train, y_cls_test = train_test_split(
            df[features], fare_class, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return (X_train_scaled, X_test_scaled, 
                y_reg_train, y_reg_test,
                y_cls_train, y_cls_test)
    
    def build_regression_model(self, input_shape):
        """Build a neural network for fare amount prediction"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)  # Linear activation for regression
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        return model
    
    def build_classification_model(self, input_shape, num_classes=4):
        """Build a neural network for fare classification"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_shape,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        return model
    
    def train_models(self):
        """Train both regression and classification models"""
        (X_train, X_test, 
         y_reg_train, y_reg_test,
         y_cls_train, y_cls_test) = self.preprocess_data()
        
        # Regression model
        print("Training regression model...")
        self.regression_model = self.build_regression_model(X_train.shape[1])
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history_reg = self.regression_model.fit(
            X_train, y_reg_train,
            validation_data=(X_test, y_reg_test),
            epochs=50,
            batch_size=1024,
            callbacks=[early_stop],
            verbose=1)
        
        # Classification model
        print("\nTraining classification model...")
        self.classification_model = self.build_classification_model(X_train.shape[1])
        
        history_cls = self.classification_model.fit(
            X_train, y_cls_train,
            validation_data=(X_test, y_cls_test),
            epochs=50,
            batch_size=1024,
            callbacks=[early_stop],
            verbose=1)
        
        return history_reg, history_cls
    
    def evaluate_models(self, X_test, y_reg_test, y_cls_test):
        """Evaluate both models on test data"""
        # Regression evaluation
        reg_pred = self.regression_model.predict(X_test)
        reg_mse = mean_squared_error(y_reg_test, reg_pred)
        reg_rmse = np.sqrt(reg_mse)
        print(f"\nRegression Results - RMSE: {reg_rmse:.2f}")
        
        # Classification evaluation
        cls_pred = np.argmax(self.classification_model.predict(X_test), axis=1)
        cls_acc = accuracy_score(y_cls_test, cls_pred)
        print(f"\nClassification Results - Accuracy: {cls_acc:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_cls_test, cls_pred, 
                                  target_names=['< $10', '$10-$30', '$30-$60', '> $60']))
        
        return reg_rmse, cls_acc

# Create and train the predictor
predictor = TaxiFarePredictor(data_loader)

reg_history, cls_history = predictor.train_models()

# Get the preprocessed test data
(_, X_test, _, y_reg_test, _, y_cls_test) = predictor.preprocess_data()

# Evaluate the models
predictor.evaluate_models(X_test, y_reg_test, y_cls_test)

predictor.regression_model.save("taxi_fare_regression_model.h5")
predictor.classification_model.save("taxi_fare_classifier.h5")
















class Clusterer:
    def __init__(self, data_loader, sample_size=50000, random_state=42):
        self.data_loader = data_loader
        self.sample_size = sample_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.df = None
        self.features = None
        self.scaled_features = None

    def preprocess_data(self):
        """Prepare data for clustering"""
        df = self.data_loader.get_full_data().copy()
        
        # Feature engineering
        df['trip_duration'] = (pd.to_datetime(df['tpep_dropoff_datetime']) - 
                              pd.to_datetime(df['tpep_pickup_datetime'])).dt.total_seconds() / 60
        df['hour_of_day'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.hour
        df['is_weekend'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.weekday >= 5
        
        # Select relevant features for clustering
        features = [
            'trip_distance', 
            'trip_duration',
            'fare_amount',
            'passenger_count',
            'PULocationID',
            'DOLocationID',
            'hour_of_day'
        ]
        
        # Sample data
        self.df = df.sample(min(self.sample_size, len(df)), random_state=self.random_state)
        self.features = self.df[features]
        
        # Scale features
        self.scaled_features = self.scaler.fit_transform(self.features)
        return self.scaled_features

    def apply_kmeans(self, n_clusters_range=range(2, 8)):
        """Apply K-Means with varying cluster numbers"""
        results = {}
        silhouette_scores = []
        
        for n in n_clusters_range:
            kmeans = KMeans(n_clusters=n, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(self.scaled_features)
            
            silhouette = silhouette_score(self.scaled_features, cluster_labels)
            silhouette_scores.append(silhouette)
            
            results[n] = {
                'model': kmeans,
                'labels': cluster_labels,
                'silhouette': silhouette,
                'inertia': kmeans.inertia_
            }
            
            print(f"K-Means with {n} clusters - Silhouette: {silhouette:.3f}")
        
        # Plot results
        self._plot_cluster_metrics(
            n_clusters_range, 
            [results[n]['silhouette'] for n in n_clusters_range],
            'Silhouette Score',
            'K-Means Clustering Quality'
        )
        
        self._plot_cluster_metrics(
            n_clusters_range, 
            [results[n]['inertia'] for n in n_clusters_range],
            'Inertia',
            'K-Means Inertia'
        )
        
        return results

    def apply_dbscan(self, eps_range=[0.5, 1.0, 1.5], min_samples_range=[5, 10, 20]):
        """Apply DBSCAN with different parameters"""
        results = {}
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels = dbscan.fit_predict(self.scaled_features)
                
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                
                if n_clusters > 1:
                    silhouette = silhouette_score(self.scaled_features, cluster_labels)
                else:
                    silhouette = -1
                
                results[(eps, min_samples)] = {
                    'model': dbscan,
                    'labels': cluster_labels,
                    'silhouette': silhouette,
                    'n_clusters': n_clusters,
                    'noise_points': sum(cluster_labels == -1)
                }
                
                print(f"DBSCAN (eps={eps}, min_samples={min_samples}) - "
                      f"Clusters: {n_clusters}, Silhouette: {silhouette:.3f}, "
                      f"Noise: {results[(eps, min_samples)]['noise_points']}")
        
        return results

    def _plot_cluster_metrics(self, x_values, y_values, y_label, title):
        """Helper function to plot cluster metrics"""
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, 'bo-')
        plt.xlabel('Number of clusters')
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        plt.show()

    def visualize_clusters(self, cluster_labels, algorithm_name):
        """Visualize clusters using PCA"""
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(self.scaled_features)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                            c=cluster_labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title(f'{algorithm_name} Clustering (PCA-reduced)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.show()
        
        return cluster_labels

    def analyze_clusters(self, cluster_labels):
        """Analyze cluster characteristics"""
        temp_df = self.df.copy()
        temp_df['cluster'] = cluster_labels
        
        numeric_df = temp_df.select_dtypes(include=[np.number])
        cluster_stats = numeric_df.groupby('cluster').mean()
        cluster_counts = numeric_df['cluster'].value_counts().rename('count')
        
        result_df = cluster_stats.join(cluster_counts)
        
        print("\nCluster Characteristics:")
        print(result_df[['trip_distance', 'trip_duration', 'fare_amount', 
                       'passenger_count', 'hour_of_day', 'count']])
        
        return result_df

# Initialize components
clusterer = Clusterer(data_loader)
clusterer.preprocess_data()

# K-Means analysis
kmeans_results = clusterer.apply_kmeans(n_clusters_range=range(2, 6))
best_k = max(kmeans_results, key=lambda k: kmeans_results[k]['silhouette'])
labels = kmeans_results[best_k]['labels']

# Visualize and analyze
clusterer.visualize_clusters(labels, "K-Means")
cluster_stats = clusterer.analyze_clusters(labels)

# DBSCAN analysis (similar pattern)
dbscan_results = clusterer.apply_dbscan()
best_params = max(dbscan_results, key=lambda k: dbscan_results[k]['silhouette'])
labels_dbscan = dbscan_results[best_params]['labels']

clusterer.visualize_clusters(labels_dbscan, "DBSCAN")
clusterer.analyze_clusters(labels_dbscan)

# %%
