# %% [markdown]
# # Data Preprocessing Module
# ========================
# 
# This module contains data loading and preprocessing routines for the research project.
# The autonomous agent will populate this file with appropriate preprocessing steps
# based on the dataset characteristics discovered during EDA.
# 
# Author: Autonomous Research Agent
# Date: Generated automatically

# %%
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# %% [markdown]
# ## Data Preprocessing Class Definition
# 
# This class handles all data preprocessing operations including:
# - Data loading from various formats
# - Missing value handling
# - Feature encoding and scaling
# - Train/test splitting

# %%
class DataPreprocessor:
    """
    Data preprocessing class that handles complete data preparation pipeline.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the preprocessor with data path.
        
        Args:
            data_path (str): Path to the dataset
        """
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.label_encoders = {}
        self.imputer = None
        self.feature_names = []
        self.target_column = None
        
    def load_data(self, target_column=None):
        """
        Load data from the specified path with automatic format detection.
        
        Args:
            target_column (str): Name of the target column for supervised learning
        """
        if not self.data_path:
            print("No data path specified")
            return False
        
        try:
            # Detect file format and load accordingly
            if self.data_path.endswith('.csv'):
                self.data = pd.read_csv(self.data_path)
            elif self.data_path.endswith('.json'):
                self.data = pd.read_json(self.data_path)
            elif self.data_path.endswith('.xlsx') or self.data_path.endswith('.xls'):
                self.data = pd.read_excel(self.data_path)
            elif self.data_path.endswith('.parquet'):
                self.data = pd.read_parquet(self.data_path)
            else:
                # Try CSV as default
                self.data = pd.read_csv(self.data_path)
            
            self.target_column = target_column
            print(f"✅ Data loaded successfully!")
            print(f"Shape: {self.data.shape}")
            print(f"Columns: {list(self.data.columns)}")
            
            if target_column and target_column in self.data.columns:
                print(f"Target column '{target_column}' found")
                print(f"Target distribution:\n{self.data[target_column].value_counts()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False

# %% [markdown]
# ## Data Structure Exploration
# 
# Comprehensive analysis of the dataset structure and characteristics

# %%
    def explore_data_structure(self):
        """
        Explore basic data structure and characteristics.
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        print("=" * 50)
        print("DATA STRUCTURE ANALYSIS")
        print("=" * 50)
        
        # Basic info
        print(f"\n📊 Dataset Shape: {self.data.shape}")
        print(f"📊 Memory Usage: {self.data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Data types
        print(f"\n📋 Data Types:")
        dtype_counts = self.data.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")
        
        # Missing values
        missing_data = self.data.isnull().sum()
        missing_percent = (missing_data / len(self.data)) * 100
        
        print(f"\n🔍 Missing Values:")
        if missing_data.sum() == 0:
            print("  No missing values found!")
        else:
            missing_df = pd.DataFrame({
                'Missing Count': missing_data[missing_data > 0],
                'Missing %': missing_percent[missing_data > 0]
            }).sort_values('Missing %', ascending=False)
            print(missing_df)
        
        # Numerical columns summary
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print(f"\n📈 Numerical Columns ({len(numerical_cols)}):")
            print(self.data[numerical_cols].describe())
        
        # Categorical columns summary
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            print(f"\n📝 Categorical Columns ({len(categorical_cols)}):")
            for col in categorical_cols:
                unique_count = self.data[col].nunique()
                print(f"  {col}: {unique_count} unique values")
                if unique_count <= 10:
                    print(f"    Values: {list(self.data[col].unique())}")

# %% [markdown]
# ## Missing Value Handling
# 
# Intelligent missing value imputation based on data types and patterns

# %%
    def handle_missing_values(self, strategy='auto'):
        """
        Handle missing values in the dataset using intelligent strategies.
        
        Args:
            strategy (str): Imputation strategy ('auto', 'mean', 'median', 'mode', 'drop')
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        missing_data = self.data.isnull().sum()
        if missing_data.sum() == 0:
            print("✅ No missing values to handle!")
            return
        
        print("🔧 Handling missing values...")
        
        # Separate numerical and categorical columns
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        
        if strategy == 'auto':
            # Intelligent strategy selection
            for col in numerical_cols:
                if self.data[col].isnull().sum() > 0:
                    # Use median for numerical columns (robust to outliers)
                    self.data[col].fillna(self.data[col].median(), inplace=True)
                    print(f"  Filled {col} with median value")
            
            for col in categorical_cols:
                if self.data[col].isnull().sum() > 0:
                    # Use mode for categorical columns
                    mode_value = self.data[col].mode()[0] if not self.data[col].mode().empty else 'Unknown'
                    self.data[col].fillna(mode_value, inplace=True)
                    print(f"  Filled {col} with mode value: {mode_value}")
        
        elif strategy == 'drop':
            # Drop rows with missing values
            original_shape = self.data.shape
            self.data.dropna(inplace=True)
            print(f"  Dropped rows: {original_shape[0] - self.data.shape[0]}")
        
        else:
            # Use sklearn imputer
            if len(numerical_cols) > 0:
                imputer_num = SimpleImputer(strategy=strategy if strategy in ['mean', 'median'] else 'median')
                self.data[numerical_cols] = imputer_num.fit_transform(self.data[numerical_cols])
            
            if len(categorical_cols) > 0:
                imputer_cat = SimpleImputer(strategy='most_frequent')
                self.data[categorical_cols] = imputer_cat.fit_transform(self.data[categorical_cols])
        
        print(f"✅ Missing value handling complete. Remaining missing values: {self.data.isnull().sum().sum()}")

# %% [markdown]
# ## Categorical Variable Encoding
# 
# Convert categorical variables to numerical format for machine learning

# %%
    def encode_categorical_variables(self, encoding_strategy='auto'):
        """
        Encode categorical variables for machine learning.
        
        Args:
            encoding_strategy (str): 'auto', 'label', 'onehot'
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        
        # Remove target column from encoding if it exists
        if self.target_column and self.target_column in categorical_cols:
            categorical_cols = categorical_cols.drop(self.target_column)
        
        if len(categorical_cols) == 0:
            print("✅ No categorical variables to encode!")
            return
        
        print("🔧 Encoding categorical variables...")
        
        for col in categorical_cols:
            unique_count = self.data[col].nunique()
            
            if encoding_strategy == 'auto':
                # Use label encoding for high cardinality, one-hot for low cardinality
                if unique_count > 10:
                    # Label encoding for high cardinality
                    le = LabelEncoder()
                    self.data[col] = le.fit_transform(self.data[col].astype(str))
                    self.label_encoders[col] = le
                    print(f"  Label encoded {col} ({unique_count} categories)")
                else:
                    # One-hot encoding for low cardinality
                    dummies = pd.get_dummies(self.data[col], prefix=col, drop_first=True)
                    self.data = pd.concat([self.data.drop(col, axis=1), dummies], axis=1)
                    print(f"  One-hot encoded {col} ({unique_count} categories)")
            
            elif encoding_strategy == 'label':
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
                print(f"  Label encoded {col}")
            
            elif encoding_strategy == 'onehot':
                dummies = pd.get_dummies(self.data[col], prefix=col, drop_first=True)
                self.data = pd.concat([self.data.drop(col, axis=1), dummies], axis=1)
                print(f"  One-hot encoded {col}")
        
        print("✅ Categorical encoding complete!")

# %% [markdown]
# ## Feature Scaling and Normalization
# 
# Scale numerical features for optimal model performance

# %%
    def scale_numerical_features(self, scaling_method='standard'):
        """
        Scale numerical features for better model performance.
        
        Args:
            scaling_method (str): 'standard', 'minmax', 'robust'
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        
        # Remove target column from scaling if it exists and is numerical
        if self.target_column and self.target_column in numerical_cols:
            numerical_cols = numerical_cols.drop(self.target_column)
        
        if len(numerical_cols) == 0:
            print("✅ No numerical features to scale!")
            return
        
        print(f"🔧 Scaling numerical features using {scaling_method} scaling...")
        
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
        
        # Fit and transform the numerical columns
        self.data[numerical_cols] = self.scaler.fit_transform(self.data[numerical_cols])
        
        print(f"✅ Scaled {len(numerical_cols)} numerical features")
        print(f"  Features scaled: {list(numerical_cols)}")

# %% [markdown]
# ## Train-Test Split Creation
# 
# Create training and testing datasets for model evaluation

# %%
    def create_train_test_split(self, test_size=0.2, random_state=42, stratify=True):
        """
        Create train/test splits for model evaluation.
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            stratify (bool): Whether to stratify split for classification
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return False
        
        if not self.target_column or self.target_column not in self.data.columns:
            print("Target column not specified or not found in data.")
            return False
        
        print("🔧 Creating train-test split...")
        
        # Separate features and target
        X = self.data.drop(self.target_column, axis=1)
        y = self.data[self.target_column]
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Determine if we should stratify
        stratify_param = None
        if stratify and y.dtype == 'object' or y.nunique() < 20:
            stratify_param = y
            print("  Using stratified split for classification")
        
        # Create the split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        
        print(f"✅ Train-test split created:")
        print(f"  Training set: {self.X_train.shape[0]} samples")
        print(f"  Test set: {self.X_test.shape[0]} samples")
        print(f"  Features: {self.X_train.shape[1]}")
        
        if stratify_param is not None:
            print(f"  Training target distribution:\n{self.y_train.value_counts()}")
            print(f"  Test target distribution:\n{self.y_test.value_counts()}")
        
        return True

# %% [markdown]
# ## Data Saving and Export
# 
# Save processed data for model training and future use

# %%
    def save_processed_data(self, output_dir="processed_data"):
        """
        Save processed data for model training.
        
        Args:
            output_dir (str): Directory to save processed data
        """
        if self.data is None:
            print("No data to save. Please load and process data first.")
            return False
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving processed data to {output_dir}...")
        
        # Save full processed dataset
        self.data.to_csv(output_path / "processed_data.csv", index=False)
        print(f"  Saved: processed_data.csv")
        
        # Save train-test splits if they exist
        if self.X_train is not None:
            self.X_train.to_csv(output_path / "X_train.csv", index=False)
            self.X_test.to_csv(output_path / "X_test.csv", index=False)
            self.y_train.to_csv(output_path / "y_train.csv", index=False)
            self.y_test.to_csv(output_path / "y_test.csv", index=False)
            print(f"  Saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
        
        # Save preprocessing metadata
        metadata = {
            'feature_names': self.feature_names,
            'target_column': self.target_column,
            'data_shape': list(self.data.shape),
            'preprocessing_steps': {
                'missing_values_handled': True,
                'categorical_encoded': len(self.label_encoders) > 0,
                'features_scaled': self.scaler is not None,
                'train_test_split': self.X_train is not None
            }
        }
        
        import json
        with open(output_path / "preprocessing_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved: preprocessing_metadata.json")
        
        print("✅ All processed data saved successfully!")
        return True

# %% [markdown]
# ## Complete Preprocessing Pipeline
# 
# Execute the full preprocessing workflow

# %%
def run_preprocessing_pipeline(data_path, target_column=None, output_dir="processed_data"):
    """
    Run the complete preprocessing pipeline.
    
    Args:
        data_path (str): Path to the raw dataset
        target_column (str): Name of the target column
        output_dir (str): Directory to save processed data
    """
    print("🚀 Starting Data Preprocessing Pipeline")
    print("=" * 50)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(data_path)
    
    # Step 1: Load data
    if not preprocessor.load_data(target_column):
        print("❌ Failed to load data. Exiting pipeline.")
        return None
    
    # Step 2: Explore data structure
    preprocessor.explore_data_structure()
    
    # Step 3: Handle missing values
    preprocessor.handle_missing_values(strategy='auto')
    
    # Step 4: Encode categorical variables
    preprocessor.encode_categorical_variables(encoding_strategy='auto')
    
    # Step 5: Scale numerical features
    preprocessor.scale_numerical_features(scaling_method='standard')
    
    # Step 6: Create train-test split
    if target_column:
        preprocessor.create_train_test_split(test_size=0.2, random_state=42)
    
    # Step 7: Save processed data
    preprocessor.save_processed_data(output_dir)
    
    print("\n🎉 Preprocessing pipeline completed successfully!")
    return preprocessor

# %% [markdown]
# ## Main Execution
# 
# Execute preprocessing when run as standalone script

# %%
def main():
    """
    Main function to run preprocessing pipeline.
    """
    print("Data Preprocessing Module - Jupyter Compatible Format")
    print("Agent will populate this with dataset-specific preprocessing steps.")
    
    # Example usage (agent will customize this)
    # preprocessor = run_preprocessing_pipeline(
    #     data_path="data/sample_dataset.csv",
    #     target_column="target",
    #     output_dir="processed_data"
    # )
    
if __name__ == "__main__":
    main() 