#!/usr/bin/env python3
"""
Create sample dataset for testing the Autonomous Research Agent workflow.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
import os

def create_classification_dataset():
    """Create a sample classification dataset."""
    print("Creating sample classification dataset...")
    
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Add some categorical features
    df['category_A'] = np.random.choice(['Type1', 'Type2', 'Type3'], size=len(df))
    df['category_B'] = np.random.choice(['Group_X', 'Group_Y'], size=len(df))
    
    # Add some missing values randomly
    missing_indices = np.random.choice(df.index, size=50, replace=False)
    df.loc[missing_indices, 'feature_3'] = np.nan
    
    # Save to CSV
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/sample_classification.csv', index=False)
    print(f"✅ Classification dataset saved: {df.shape}")
    print(f"   Features: {df.shape[1]-1}, Samples: {df.shape[0]}")
    print(f"   Target classes: {df['target'].value_counts().to_dict()}")
    
    return df

def create_regression_dataset():
    """Create a sample regression dataset."""
    print("Creating sample regression dataset...")
    
    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=800,
        n_features=8,
        n_informative=6,
        noise=0.1,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'numeric_feature_{i+1}' for i in range(X.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target_value'] = y
    
    # Add some categorical features
    df['region'] = np.random.choice(['North', 'South', 'East', 'West'], size=len(df))
    df['size_category'] = np.random.choice(['Small', 'Medium', 'Large'], size=len(df))
    
    # Add some missing values
    missing_indices = np.random.choice(df.index, size=30, replace=False)
    df.loc[missing_indices, 'numeric_feature_2'] = np.nan
    
    # Save to CSV
    df.to_csv('data/raw/sample_regression.csv', index=False)
    print(f"✅ Regression dataset saved: {df.shape}")
    print(f"   Features: {df.shape[1]-1}, Samples: {df.shape[0]}")
    print(f"   Target range: {df['target_value'].min():.2f} to {df['target_value'].max():.2f}")
    
    return df

def main():
    """Create both sample datasets."""
    print("🚀 Creating Sample Datasets for Testing")
    print("=" * 50)
    
    # Create datasets
    classification_df = create_classification_dataset()
    regression_df = create_regression_dataset()
    
    print("\n📊 Sample datasets created successfully!")
    print("📁 Location: data/raw/")
    print("   - sample_classification.csv (classification problem)")
    print("   - sample_regression.csv (regression problem)")
    
    print("\n🧪 Test the agent with:")
    print("   python demo_agent.py --data_path 'data/raw/sample_classification.csv' --target_column 'target'")
    print("   python demo_agent.py --data_path 'data/raw/sample_regression.csv' --target_column 'target_value'")

if __name__ == "__main__":
    main() 