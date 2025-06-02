# %% [markdown]
# # Exploratory Data Analysis Module
# ===============================
# 
# This module contains exploratory data analysis scripts for understanding
# the dataset characteristics, distributions, and patterns.
# 
# Author: Autonomous Research Agent
# Date: Generated automatically

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# %% [markdown]
# ## EDA Class Definition
# 
# Comprehensive exploratory data analysis with automatic visualization generation
# and programmatic image analysis for autonomous research workflows

# %%
class ExploratoryDataAnalyzer:
    """
    Comprehensive Exploratory Data Analysis class with full implementation
    for autonomous research workflows.
    """
    
    def __init__(self, data_path=None, output_dir="../data_analysis/visualizations"):
        """
        Initialize the EDA analyzer.
        
        Args:
            data_path (str): Path to the dataset
            output_dir (str): Directory to save visualizations
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = None
        self.numerical_cols = []
        self.categorical_cols = []
        self.target_column = None
        self.eda_results = {}
        
    def load_and_inspect_data(self, target_column=None):
        """
        Load data and perform initial inspection with comprehensive analysis.
        
        Args:
            target_column (str): Name of the target column for supervised learning
        """
        if not self.data_path:
            print("No data path specified")
            return False
        
        try:
            # Load data with format detection
            if self.data_path.endswith('.csv'):
                self.data = pd.read_csv(self.data_path)
            elif self.data_path.endswith('.json'):
                self.data = pd.read_json(self.data_path)
            elif self.data_path.endswith('.xlsx') or self.data_path.endswith('.xls'):
                self.data = pd.read_excel(self.data_path)
            else:
                self.data = pd.read_csv(self.data_path)
            
            self.target_column = target_column
            
            # Identify column types
            self.numerical_cols = list(self.data.select_dtypes(include=[np.number]).columns)
            self.categorical_cols = list(self.data.select_dtypes(include=['object', 'category']).columns)
            
            print("✅ Data loaded successfully!")
            print(f"📊 Dataset Shape: {self.data.shape}")
            print(f"📈 Numerical columns: {len(self.numerical_cols)}")
            print(f"📝 Categorical columns: {len(self.categorical_cols)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False

# %% [markdown]
# ## Summary Statistics Generation
# 
# Comprehensive statistical analysis of the dataset

# %%
    def generate_summary_statistics(self):
        """
        Generate comprehensive summary statistics for all variables.
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        print("📊 GENERATING SUMMARY STATISTICS")
        print("=" * 50)
        
        # Basic dataset info
        print(f"\n🔍 Dataset Overview:")
        print(f"  Shape: {self.data.shape}")
        print(f"  Memory usage: {self.data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        print(f"  Duplicate rows: {self.data.duplicated().sum()}")
        
        # Numerical statistics
        if self.numerical_cols:
            print(f"\n📈 Numerical Variables Summary:")
            numerical_stats = self.data[self.numerical_cols].describe()
            print(numerical_stats)
            
            # Additional statistics
            print(f"\n📊 Additional Numerical Statistics:")
            for col in self.numerical_cols:
                skewness = stats.skew(self.data[col].dropna())
                kurtosis = stats.kurtosis(self.data[col].dropna())
                print(f"  {col}:")
                print(f"    Skewness: {skewness:.3f}")
                print(f"    Kurtosis: {kurtosis:.3f}")
                print(f"    Zeros: {(self.data[col] == 0).sum()}")
        
        # Categorical statistics
        if self.categorical_cols:
            print(f"\n📝 Categorical Variables Summary:")
            for col in self.categorical_cols:
                unique_count = self.data[col].nunique()
                most_frequent = self.data[col].mode()[0] if not self.data[col].mode().empty else 'N/A'
                print(f"  {col}:")
                print(f"    Unique values: {unique_count}")
                print(f"    Most frequent: {most_frequent}")
                if unique_count <= 10:
                    print(f"    Value counts:\n{self.data[col].value_counts()}")
        
        # Target variable analysis
        if self.target_column and self.target_column in self.data.columns:
            print(f"\n🎯 Target Variable Analysis ({self.target_column}):")
            if self.target_column in self.numerical_cols:
                print(f"  Type: Numerical")
                print(f"  Mean: {self.data[self.target_column].mean():.3f}")
                print(f"  Std: {self.data[self.target_column].std():.3f}")
            else:
                print(f"  Type: Categorical")
                print(f"  Class distribution:\n{self.data[self.target_column].value_counts()}")
                print(f"  Class proportions:\n{self.data[self.target_column].value_counts(normalize=True)}")

# %% [markdown]
# ## Missing Values Analysis
# 
# Comprehensive analysis of missing data patterns

# %%
    def analyze_missing_values(self):
        """
        Analyze patterns in missing values with visualization.
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        missing_data = self.data.isnull().sum()
        missing_percent = (missing_data / len(self.data)) * 100
        
        print("🔍 MISSING VALUES ANALYSIS")
        print("=" * 50)
        
        if missing_data.sum() == 0:
            print("✅ No missing values found in the dataset!")
            return
        
        # Create missing values summary
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing %': missing_percent
        }).sort_values('Missing %', ascending=False)
        
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        print(f"\n📊 Missing Values Summary:")
        print(missing_df)
        
        # Visualize missing values
        if len(missing_df) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Missing values bar plot
            missing_df['Missing %'].plot(kind='bar', ax=axes[0], color='coral')
            axes[0].set_title('Missing Values by Column (%)')
            axes[0].set_ylabel('Missing Percentage')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Missing values heatmap
            if self.data.shape[1] <= 20:  # Only for manageable number of columns
                sns.heatmap(self.data.isnull(), cbar=True, ax=axes[1], cmap='viridis')
                axes[1].set_title('Missing Values Heatmap')
            else:
                axes[1].text(0.5, 0.5, f'Too many columns ({self.data.shape[1]}) for heatmap', 
                           ha='center', va='center', transform=axes[1].transAxes)
                axes[1].set_title('Missing Values Pattern')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'missing_values_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"💾 Saved: missing_values_analysis.png")

# %% [markdown]
# ## Distribution Analysis and Visualization
# 
# Comprehensive analysis of variable distributions

# %%
    def plot_distributions(self):
        """
        Create comprehensive distribution plots for numerical variables.
        """
        if self.data is None or not self.numerical_cols:
            print("No numerical data to plot.")
            return
        
        print("📈 CREATING DISTRIBUTION PLOTS")
        print("=" * 50)
        
        # Calculate subplot dimensions
        n_cols = min(3, len(self.numerical_cols))
        n_rows = (len(self.numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(self.numerical_cols):
            if i < len(axes):
                # Create histogram with KDE
                self.data[col].hist(bins=30, alpha=0.7, ax=axes[i], density=True)
                self.data[col].plot.kde(ax=axes[i], color='red', linewidth=2)
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_ylabel('Density')
                
                # Add statistics text
                mean_val = self.data[col].mean()
                std_val = self.data[col].std()
                axes[i].axvline(mean_val, color='green', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
                axes[i].legend()
        
        # Hide empty subplots
        for i in range(len(self.numerical_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: numerical_distributions.png")
        
        # Box plots for outlier detection
        if len(self.numerical_cols) > 1:
            fig, ax = plt.subplots(figsize=(12, 6))
            self.data[self.numerical_cols].boxplot(ax=ax)
            ax.set_title('Box Plots for Outlier Detection')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'boxplots_outliers.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"💾 Saved: boxplots_outliers.png")

# %% [markdown]
# ## Categorical Variables Analysis
# 
# Comprehensive analysis and visualization of categorical variables

# %%
    def plot_categorical_variables(self):
        """
        Create visualizations for categorical variables.
        """
        if self.data is None or not self.categorical_cols:
            print("No categorical data to plot.")
            return
        
        print("📝 CREATING CATEGORICAL PLOTS")
        print("=" * 50)
        
        for col in self.categorical_cols:
            unique_count = self.data[col].nunique()
            
            if unique_count <= 20:  # Only plot if manageable number of categories
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))
                
                # Bar plot
                value_counts = self.data[col].value_counts()
                value_counts.plot(kind='bar', ax=axes[0], color='skyblue')
                axes[0].set_title(f'{col} - Value Counts')
                axes[0].set_ylabel('Count')
                axes[0].tick_params(axis='x', rotation=45)
                
                # Pie chart (if not too many categories)
                if unique_count <= 10:
                    value_counts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
                    axes[1].set_title(f'{col} - Distribution')
                    axes[1].set_ylabel('')
                else:
                    axes[1].text(0.5, 0.5, f'Too many categories\n({unique_count}) for pie chart', 
                               ha='center', va='center', transform=axes[1].transAxes)
                    axes[1].set_title(f'{col} - Categories: {unique_count}')
                
                plt.tight_layout()
                plt.savefig(self.output_dir / f'categorical_{col.lower().replace(" ", "_")}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
                print(f"💾 Saved: categorical_{col.lower().replace(' ', '_')}.png")
            else:
                print(f"⚠️  Skipping {col} - too many categories ({unique_count})")

# %% [markdown]
# ## Correlation Analysis
# 
# Comprehensive correlation analysis with advanced visualizations

# %%
    def analyze_correlations(self):
        """
        Analyze and visualize correlations between variables.
        """
        if self.data is None or len(self.numerical_cols) < 2:
            print("Insufficient numerical data for correlation analysis.")
            return
        
        print("🔗 CORRELATION ANALYSIS")
        print("=" * 50)
        
        # Calculate correlation matrix
        correlation_matrix = self.data[self.numerical_cols].corr()
        
        # Create correlation heatmap
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Full correlation heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, ax=axes[0])
        axes[0].set_title('Correlation Matrix Heatmap')
        
        # High correlation pairs
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Threshold for high correlation
                    high_corr_pairs.append((correlation_matrix.columns[i], 
                                          correlation_matrix.columns[j], corr_val))
        
        # Plot high correlations
        if high_corr_pairs:
            high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Var1', 'Var2', 'Correlation'])
            high_corr_df['Abs_Correlation'] = high_corr_df['Correlation'].abs()
            high_corr_df = high_corr_df.sort_values('Abs_Correlation', ascending=True)
            
            high_corr_df.plot(x='Var1', y='Correlation', kind='barh', ax=axes[1], 
                             color=['red' if x < 0 else 'blue' for x in high_corr_df['Correlation']])
            axes[1].set_title('High Correlation Pairs (|r| > 0.5)')
            axes[1].set_xlabel('Correlation Coefficient')
            
            print(f"\n🔍 High Correlation Pairs Found:")
            for var1, var2, corr in high_corr_pairs:
                print(f"  {var1} - {var2}: {corr:.3f}")
        else:
            axes[1].text(0.5, 0.5, 'No high correlations\n(|r| > 0.5) found', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('High Correlation Pairs')
            print("  No high correlation pairs found")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: correlation_analysis.png")

# %% [markdown]
# ## Outlier Detection and Analysis
# 
# Comprehensive outlier detection using multiple methods

# %%
    def detect_outliers(self):
        """
        Detect and visualize outliers in the data using multiple methods.
        """
        if self.data is None or not self.numerical_cols:
            print("No numerical data for outlier detection.")
            return
        
        print("🎯 OUTLIER DETECTION")
        print("=" * 50)
        
        outlier_summary = {}
        
        for col in self.numerical_cols:
            # IQR method
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
            
            # Z-score method
            z_scores = np.abs(stats.zscore(self.data[col].dropna()))
            z_outliers = self.data[z_scores > 3]
            
            outlier_summary[col] = {
                'IQR_outliers': len(iqr_outliers),
                'Z_score_outliers': len(z_outliers),
                'IQR_percentage': (len(iqr_outliers) / len(self.data)) * 100,
                'Z_score_percentage': (len(z_outliers) / len(self.data)) * 100
            }
            
            print(f"\n📊 {col}:")
            print(f"  IQR outliers: {len(iqr_outliers)} ({outlier_summary[col]['IQR_percentage']:.2f}%)")
            print(f"  Z-score outliers: {len(z_outliers)} ({outlier_summary[col]['Z_score_percentage']:.2f}%)")
        
        # Create outlier visualization
        if len(self.numerical_cols) <= 6:  # Manageable number for visualization
            fig, axes = plt.subplots(2, min(3, len(self.numerical_cols)), 
                                   figsize=(5*min(3, len(self.numerical_cols)), 10))
            if len(self.numerical_cols) == 1:
                axes = axes.reshape(-1, 1)
            elif len(self.numerical_cols) <= 3:
                axes = axes.reshape(2, -1)
            
            for i, col in enumerate(self.numerical_cols[:6]):
                row = i // 3
                col_idx = i % 3
                
                if len(self.numerical_cols) <= 3:
                    ax_box = axes[0, col_idx] if len(self.numerical_cols) > 1 else axes[0]
                    ax_hist = axes[1, col_idx] if len(self.numerical_cols) > 1 else axes[1]
                else:
                    ax_box = axes[row, col_idx]
                    ax_hist = axes[row + 1, col_idx] if row == 0 else None
                
                # Box plot
                self.data[col].plot.box(ax=ax_box)
                ax_box.set_title(f'{col} - Box Plot')
                
                # Histogram with outlier boundaries
                if ax_hist is not None:
                    self.data[col].hist(bins=30, alpha=0.7, ax=ax_hist)
                    Q1 = self.data[col].quantile(0.25)
                    Q3 = self.data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    ax_hist.axvline(Q1 - 1.5 * IQR, color='red', linestyle='--', alpha=0.7, label='IQR bounds')
                    ax_hist.axvline(Q3 + 1.5 * IQR, color='red', linestyle='--', alpha=0.7)
                    ax_hist.set_title(f'{col} - Distribution with Outlier Bounds')
                    ax_hist.legend()
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'outlier_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"💾 Saved: outlier_analysis.png")

# %% [markdown]
# ## Target Variable Analysis
# 
# Comprehensive analysis of the target variable and its relationships

# %%
    def create_target_analysis(self):
        """
        Analyze the target variable and its relationships with features.
        """
        if self.data is None or not self.target_column or self.target_column not in self.data.columns:
            print("No target variable specified or found.")
            return
        
        print(f"🎯 TARGET VARIABLE ANALYSIS: {self.target_column}")
        print("=" * 50)
        
        target_data = self.data[self.target_column]
        
        # Target variable distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Distribution plot
        if target_data.dtype in ['object', 'category'] or target_data.nunique() < 20:
            # Categorical target
            target_counts = target_data.value_counts()
            target_counts.plot(kind='bar', ax=axes[0, 0], color='lightblue')
            axes[0, 0].set_title(f'{self.target_column} Distribution')
            axes[0, 0].set_ylabel('Count')
            
            # Pie chart
            target_counts.plot(kind='pie', ax=axes[0, 1], autopct='%1.1f%%')
            axes[0, 1].set_title(f'{self.target_column} Proportions')
            axes[0, 1].set_ylabel('')
            
            print(f"📊 Target Distribution:")
            print(target_counts)
            print(f"\n📊 Target Proportions:")
            print(target_counts / target_counts.sum())
            
        else:
            # Numerical target
            target_data.hist(bins=30, ax=axes[0, 0], alpha=0.7)
            axes[0, 0].set_title(f'{self.target_column} Distribution')
            axes[0, 0].set_ylabel('Frequency')
            
            target_data.plot.box(ax=axes[0, 1])
            axes[0, 1].set_title(f'{self.target_column} Box Plot')
            
            print(f"📊 Target Statistics:")
            print(target_data.describe())
        
        # Target vs numerical features
        if self.numerical_cols and len(self.numerical_cols) > 0:
            # Select top correlated features
            feature_cols = [col for col in self.numerical_cols if col != self.target_column]
            if feature_cols:
                if target_data.dtype in ['object', 'category'] or target_data.nunique() < 20:
                    # Box plot for categorical target
                    if len(feature_cols) > 0:
                        feature_to_plot = feature_cols[0]  # Plot first numerical feature
                        sns.boxplot(data=self.data, x=self.target_column, y=feature_to_plot, ax=axes[1, 0])
                        axes[1, 0].set_title(f'{feature_to_plot} by {self.target_column}')
                        axes[1, 0].tick_params(axis='x', rotation=45)
                else:
                    # Scatter plot for numerical target
                    if len(feature_cols) > 0:
                        feature_to_plot = feature_cols[0]
                        self.data.plot.scatter(x=feature_to_plot, y=self.target_column, ax=axes[1, 0], alpha=0.6)
                        axes[1, 0].set_title(f'{self.target_column} vs {feature_to_plot}')
        
        # Correlation with target (for numerical features)
        if self.numerical_cols and self.target_column in self.numerical_cols:
            feature_cols = [col for col in self.numerical_cols if col != self.target_column]
            if feature_cols:
                correlations = self.data[feature_cols + [self.target_column]].corr()[self.target_column].drop(self.target_column)
                correlations.abs().sort_values(ascending=True).plot(kind='barh', ax=axes[1, 1])
                axes[1, 1].set_title(f'Feature Correlations with {self.target_column}')
                axes[1, 1].set_xlabel('Absolute Correlation')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'target_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: target_analysis.png")

# %% [markdown]
# ## Feature Importance and Relationships
# 
# Analysis of feature importance and inter-feature relationships

# %%
    def generate_feature_importance_plots(self):
        """
        Generate plots showing feature importance and relationships.
        """
        if self.data is None:
            print("No data loaded.")
            return
        
        print("🔍 FEATURE RELATIONSHIP ANALYSIS")
        print("=" * 50)
        
        # Pairwise relationships for numerical features (if not too many)
        if len(self.numerical_cols) <= 5 and len(self.numerical_cols) > 1:
            print("Creating pairplot for numerical features...")
            
            # Create pairplot
            if self.target_column and self.target_column in self.data.columns:
                # Color by target if categorical
                if (self.data[self.target_column].dtype in ['object', 'category'] or 
                    self.data[self.target_column].nunique() < 10):
                    pairplot = sns.pairplot(self.data[self.numerical_cols + [self.target_column]], 
                                          hue=self.target_column, diag_kind='hist')
                else:
                    pairplot = sns.pairplot(self.data[self.numerical_cols], diag_kind='hist')
            else:
                pairplot = sns.pairplot(self.data[self.numerical_cols], diag_kind='hist')
            
            pairplot.fig.suptitle('Pairwise Feature Relationships', y=1.02)
            plt.savefig(self.output_dir / 'feature_pairplot.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"💾 Saved: feature_pairplot.png")
        
        # Feature variance analysis
        if self.numerical_cols:
            feature_variance = self.data[self.numerical_cols].var().sort_values(ascending=True)
            
            plt.figure(figsize=(10, 6))
            feature_variance.plot(kind='barh', color='lightgreen')
            plt.title('Feature Variance Analysis')
            plt.xlabel('Variance')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'feature_variance.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"💾 Saved: feature_variance.png")
            
            print(f"\n📊 Feature Variance Summary:")
            print(feature_variance)

# %% [markdown]
# ## Comprehensive EDA Report Generation
# 
# Execute the complete EDA workflow and generate summary report

# %%
    def generate_eda_report(self):
        """
        Generate a comprehensive EDA report with all visualizations and analysis.
        """
        print("🚀 STARTING COMPREHENSIVE EDA ANALYSIS")
        print("=" * 60)
        
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        # Execute all analysis steps
        print("\n1️⃣ Generating summary statistics...")
        self.generate_summary_statistics()
        
        print("\n2️⃣ Analyzing missing values...")
        self.analyze_missing_values()
        
        print("\n3️⃣ Creating distribution plots...")
        self.plot_distributions()
        
        print("\n4️⃣ Analyzing categorical variables...")
        self.plot_categorical_variables()
        
        print("\n5️⃣ Performing correlation analysis...")
        self.analyze_correlations()
        
        print("\n6️⃣ Detecting outliers...")
        self.detect_outliers()
        
        print("\n7️⃣ Analyzing target variable...")
        self.create_target_analysis()
        
        print("\n8️⃣ Generating feature importance plots...")
        self.generate_feature_importance_plots()
        
        # Create summary visualization
        self._create_summary_dashboard()
        
        print(f"\n✅ EDA ANALYSIS COMPLETE!")
        print(f"📁 All visualizations saved to: {self.output_dir}")
        print(f"📊 Total visualizations generated: {len(list(self.output_dir.glob('*.png')))}")

# %% [markdown]
# ## Summary Dashboard Creation
# 
# Create a comprehensive summary dashboard

# %%
    def _create_summary_dashboard(self):
        """
        Create a comprehensive summary dashboard.
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('EDA Summary Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Dataset overview
        ax1 = axes[0, 0]
        overview_text = f"""Dataset Overview
        
Shape: {self.data.shape}
Numerical cols: {len(self.numerical_cols)}
Categorical cols: {len(self.categorical_cols)}
Missing values: {self.data.isnull().sum().sum()}
Duplicates: {self.data.duplicated().sum()}
Memory: {self.data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB"""
        
        ax1.text(0.1, 0.9, overview_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        ax1.set_title('Dataset Overview')
        ax1.axis('off')
        
        # 2. Data types distribution
        ax2 = axes[0, 1]
        dtype_counts = self.data.dtypes.value_counts()
        dtype_counts.plot(kind='pie', ax=ax2, autopct='%1.1f%%')
        ax2.set_title('Data Types Distribution')
        ax2.set_ylabel('')
        
        # 3. Missing values summary
        ax3 = axes[0, 2]
        missing_data = self.data.isnull().sum()
        if missing_data.sum() > 0:
            missing_data[missing_data > 0].plot(kind='bar', ax=ax3, color='coral')
            ax3.set_title('Missing Values by Column')
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=14)
            ax3.set_title('Missing Values Status')
            ax3.axis('off')
        
        # 4. Numerical features summary
        ax4 = axes[1, 0]
        if self.numerical_cols:
            self.data[self.numerical_cols[:5]].boxplot(ax=ax4)  # Show first 5 numerical columns
            ax4.set_title('Numerical Features Distribution')
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'No Numerical Features', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Numerical Features')
            ax4.axis('off')
        
        # 5. Target variable summary
        ax5 = axes[1, 1]
        if self.target_column and self.target_column in self.data.columns:
            if (self.data[self.target_column].dtype in ['object', 'category'] or 
                self.data[self.target_column].nunique() < 20):
                self.data[self.target_column].value_counts().plot(kind='bar', ax=ax5, color='lightblue')
                ax5.set_title(f'Target: {self.target_column}')
                ax5.tick_params(axis='x', rotation=45)
            else:
                self.data[self.target_column].hist(bins=20, ax=ax5, alpha=0.7)
                ax5.set_title(f'Target: {self.target_column}')
        else:
            ax5.text(0.5, 0.5, 'No Target Variable', ha='center', va='center',
                    transform=ax5.transAxes, fontsize=14)
            ax5.set_title('Target Variable')
            ax5.axis('off')
        
        # 6. Correlation summary
        ax6 = axes[1, 2]
        if len(self.numerical_cols) > 1:
            corr_matrix = self.data[self.numerical_cols].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5, ax=ax6, cbar_kws={"shrink": .8})
            ax6.set_title('Feature Correlations')
        else:
            ax6.text(0.5, 0.5, 'Insufficient Features\nfor Correlation', ha='center', va='center',
                    transform=ax6.transAxes, fontsize=14)
            ax6.set_title('Correlation Analysis')
            ax6.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'eda_summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: eda_summary_dashboard.png")

# %% [markdown]
# ## Complete EDA Pipeline Function
# 
# Execute the complete EDA workflow

# %%
def run_eda_pipeline(data_path, target_column=None, output_dir="../data_analysis/visualizations"):
    """
    Run the complete EDA pipeline.
    
    Args:
        data_path (str): Path to the dataset
        target_column (str): Name of the target column
        output_dir (str): Directory to save visualizations
    """
    print("🚀 Starting Comprehensive EDA Pipeline")
    print("=" * 50)
    
    # Initialize EDA analyzer
    analyzer = ExploratoryDataAnalyzer(data_path, output_dir)
    
    # Load and analyze data
    if not analyzer.load_and_inspect_data(target_column):
        print("❌ Failed to load data. Exiting EDA pipeline.")
        return None
    
    # Generate comprehensive EDA report
    analyzer.generate_eda_report()
    
    print("\n🎉 EDA pipeline completed successfully!")
    return analyzer

# %% [markdown]
# ## Main Execution
# 
# Execute EDA when run as standalone script

# %%
def main():
    """
    Main function to run EDA pipeline.
    """
    print("Exploratory Data Analysis Module - Jupyter Compatible Format")
    print("Agent will populate this with dataset-specific EDA steps.")
    
    # Example usage (agent will customize this)
    # analyzer = run_eda_pipeline(
    #     data_path="data/sample_dataset.csv",
    #     target_column="target",
    #     output_dir="data_analysis/visualizations"
    # )
    
if __name__ == "__main__":
    main() 