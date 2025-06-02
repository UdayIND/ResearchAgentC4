# %% [markdown]
# # Model Pipeline Module
# ====================
# 
# This module contains the machine learning model pipeline including
# model selection, training, evaluation, and result analysis.
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
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           classification_report, confusion_matrix, roc_auc_score, roc_curve)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import joblib

# Deep Learning imports (optional)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# %% [markdown]
# ## Model Pipeline Class Definition
# 
# Comprehensive machine learning pipeline with full implementation
# for autonomous research workflows

# %%
class ModelPipeline:
    """
    Comprehensive Machine Learning pipeline class with full implementation
    for autonomous research workflows.
    """
    
    def __init__(self, problem_type="classification", output_dir="../data_analysis/visualizations"):
        """
        Initialize the model pipeline.
        
        Args:
            problem_type (str): Type of ML problem (classification, regression, etc.)
            output_dir (str): Directory to save results and visualizations
        """
        self.problem_type = problem_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = []
        
        # Model containers
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
        # Results tracking
        self.training_history = {}
        
    def load_processed_data(self, data_path=None, X_train=None, X_test=None, y_train=None, y_test=None):
        """
        Load preprocessed data for model training.
        
        Args:
            data_path (str): Path to processed data directory
            X_train, X_test, y_train, y_test: Direct data arrays
        """
        if data_path:
            # Load from files
            data_dir = Path(data_path)
            try:
                self.X_train = pd.read_csv(data_dir / "X_train.csv")
                self.X_test = pd.read_csv(data_dir / "X_test.csv")
                self.y_train = pd.read_csv(data_dir / "y_train.csv").iloc[:, 0]
                self.y_test = pd.read_csv(data_dir / "y_test.csv").iloc[:, 0]
                self.feature_names = list(self.X_train.columns)
                
                print(f"✅ Data loaded from {data_path}")
                print(f"📊 Training set: {self.X_train.shape}")
                print(f"📊 Test set: {self.X_test.shape}")
                print(f"📊 Features: {len(self.feature_names)}")
                return True
                
            except Exception as e:
                print(f"❌ Error loading data: {str(e)}")
                return False
        else:
            # Use provided arrays
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            if hasattr(X_train, 'columns'):
                self.feature_names = list(X_train.columns)
            else:
                self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
            
            print(f"✅ Data loaded directly")
            print(f"📊 Training set: {self.X_train.shape}")
            print(f"📊 Test set: {self.X_test.shape}")
            return True

# %% [markdown]
# ## Model Setup and Configuration
# 
# Setup baseline and advanced models based on problem type

# %%
    def setup_baseline_models(self):
        """
        Set up baseline models for comparison based on problem type.
        """
        print(f"🔧 Setting up baseline models for {self.problem_type}")
        
        if self.problem_type == "classification":
            self.models = {
                'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Decision_Tree': DecisionTreeClassifier(random_state=42),
                'SVM': SVC(random_state=42, probability=True),
                'KNN': KNeighborsClassifier(n_neighbors=5),
                'Naive_Bayes': GaussianNB(),
                'Gradient_Boosting': GradientBoostingClassifier(random_state=42)
            }
        elif self.problem_type == "regression":
            self.models = {
                'Linear_Regression': LinearRegression(),
                'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Decision_Tree': DecisionTreeRegressor(random_state=42),
                'SVM': SVR(),
                'KNN': KNeighborsRegressor(n_neighbors=5),
                'Gradient_Boosting': GradientBoostingRegressor(random_state=42)
            }
        
        print(f"✅ {len(self.models)} baseline models configured")
        for model_name in self.models.keys():
            print(f"  - {model_name}")
    
    def setup_advanced_models(self):
        """
        Set up advanced models (neural networks, transformers, etc.).
        """
        print("🚀 Setting up advanced models...")
        
        if TORCH_AVAILABLE:
            print("  PyTorch available - can add neural networks")
            # Could add custom neural network models here
        
        if TRANSFORMERS_AVAILABLE:
            print("  Transformers available - can add transformer models")
            # Could add transformer-based models here
        
        print("✅ Advanced model setup complete")

# %% [markdown]
# ## Model Training and Evaluation
# 
# Comprehensive model training with cross-validation and evaluation

# %%
    def train_model(self, model_name, model, X_train, y_train):
        """
        Train a specific model with comprehensive logging.
        
        Args:
            model_name (str): Name identifier for the model
            model: The model object to train
            X_train: Training features
            y_train: Training targets
        """
        print(f"🏋️ Training {model_name}...")
        
        start_time = datetime.now()
        
        try:
            # Train the model
            model.fit(X_train, y_train)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            print(f"  ✅ Training completed in {training_time:.2f} seconds")
            
            # Store training info
            self.training_history[model_name] = {
                'training_time': training_time,
                'training_samples': len(X_train),
                'features': X_train.shape[1] if hasattr(X_train, 'shape') else len(X_train[0])
            }
            
            return model
            
        except Exception as e:
            print(f"  ❌ Training failed: {str(e)}")
            return None
    
    def evaluate_model(self, model_name, model, X_test, y_test):
        """
        Evaluate a trained model and store comprehensive results.
        
        Args:
            model_name (str): Name identifier for the model
            model: The trained model object
            X_test: Test features
            y_test: Test targets
        """
        print(f"📊 Evaluating {model_name}...")
        
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics based on problem type
            if self.problem_type == "classification":
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # ROC AUC for binary classification
                try:
                    if hasattr(model, 'predict_proba') and len(np.unique(y_test)) == 2:
                        y_proba = model.predict_proba(X_test)[:, 1]
                        roc_auc = roc_auc_score(y_test, y_proba)
                    else:
                        roc_auc = None
                except:
                    roc_auc = None
                
                self.results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'predictions': y_pred,
                    'model': model
                }
                
                print(f"  📈 Accuracy: {accuracy:.4f}")
                print(f"  📈 Precision: {precision:.4f}")
                print(f"  📈 Recall: {recall:.4f}")
                print(f"  📈 F1-Score: {f1:.4f}")
                if roc_auc:
                    print(f"  📈 ROC AUC: {roc_auc:.4f}")
                
            elif self.problem_type == "regression":
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                self.results[model_name] = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2_score': r2,
                    'predictions': y_pred,
                    'model': model
                }
                
                print(f"  📈 MSE: {mse:.4f}")
                print(f"  📈 RMSE: {rmse:.4f}")
                print(f"  📈 MAE: {mae:.4f}")
                print(f"  📈 R² Score: {r2:.4f}")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Evaluation failed: {str(e)}")
            return False
    
    def perform_cross_validation(self, model, X, y, cv=5):
        """
        Perform cross-validation for robust model evaluation.
        
        Args:
            model: The model to evaluate
            X: Features
            y: Targets
            cv (int): Number of cross-validation folds
        """
        print(f"🔄 Performing {cv}-fold cross-validation...")
        
        try:
            if self.problem_type == "classification":
                scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
                metric_name = "Accuracy"
            else:
                scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
                metric_name = "R² Score"
            
            print(f"  📊 {metric_name} scores: {scores}")
            print(f"  📊 Mean {metric_name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            
            return scores
            
        except Exception as e:
            print(f"  ❌ Cross-validation failed: {str(e)}")
            return None

# %% [markdown]
# ## Hyperparameter Tuning
# 
# Automated hyperparameter optimization for best performance

# %%
    def hyperparameter_tuning(self, model_name, model, param_grid, X_train, y_train):
        """
        Perform hyperparameter tuning using grid search.
        
        Args:
            model_name (str): Name of the model
            model: The model object
            param_grid (dict): Parameter grid for tuning
            X_train: Training features
            y_train: Training targets
        """
        print(f"🎛️ Tuning hyperparameters for {model_name}...")
        
        try:
            if self.problem_type == "classification":
                scoring = 'accuracy'
            else:
                scoring = 'r2'
            
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring=scoring, 
                n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            print(f"  ✅ Best parameters: {grid_search.best_params_}")
            print(f"  ✅ Best score: {grid_search.best_score_:.4f}")
            
            # Update the model with best parameters
            self.models[model_name] = grid_search.best_estimator_
            
            return grid_search.best_estimator_
            
        except Exception as e:
            print(f"  ❌ Hyperparameter tuning failed: {str(e)}")
            return model

# %% [markdown]
# ## Visualization and Analysis
# 
# Comprehensive visualization of model performance and analysis

# %%
    def create_confusion_matrix_plot(self, y_true, y_pred, model_name):
        """
        Create and save confusion matrix visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name (str): Name of the model for the plot title
        """
        if self.problem_type != "classification":
            return
        
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'confusion_matrix_{model_name.lower()}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: confusion_matrix_{model_name.lower()}.png")
    
    def create_feature_importance_plot(self, model, feature_names, model_name):
        """
        Create feature importance visualization for tree-based models.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            model_name (str): Name of the model
        """
        if not hasattr(model, 'feature_importances_'):
            return
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importance - {model_name}')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(self.output_dir / f'feature_importance_{model_name.lower()}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: feature_importance_{model_name.lower()}.png")
    
    def create_learning_curves(self, model, X, y, model_name):
        """
        Create learning curves to analyze model performance vs training size.
        
        Args:
            model: The model to analyze
            X: Features
            y: Targets
            model_name (str): Name of the model
        """
        print(f"📈 Creating learning curves for {model_name}...")
        
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y, cv=5, n_jobs=-1, 
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='accuracy' if self.problem_type == 'classification' else 'r2'
            )
            
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training score')
            plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation score')
            plt.fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                           np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1)
            plt.fill_between(train_sizes, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                           np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.1)
            
            plt.xlabel('Training Set Size')
            plt.ylabel('Accuracy' if self.problem_type == 'classification' else 'R² Score')
            plt.title(f'Learning Curves - {model_name}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.output_dir / f'learning_curves_{model_name.lower()}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"💾 Saved: learning_curves_{model_name.lower()}.png")
            
        except Exception as e:
            print(f"  ❌ Learning curves failed: {str(e)}")

# %% [markdown]
# ## Model Comparison and Selection
# 
# Compare all models and select the best performer

# %%
    def compare_models(self):
        """
        Compare all trained models and identify the best performer.
        """
        if not self.results:
            print("❌ No model results to compare")
            return
        
        print("🏆 COMPARING MODEL PERFORMANCE")
        print("=" * 50)
        
        # Create comparison DataFrame
        if self.problem_type == "classification":
            comparison_data = []
            for model_name, results in self.results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': results['accuracy'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1-Score': results['f1_score'],
                    'ROC AUC': results.get('roc_auc', 'N/A')
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            print(comparison_df.round(4))
            
            # Find best model based on F1-score
            best_idx = comparison_df['F1-Score'].idxmax()
            self.best_model_name = comparison_df.loc[best_idx, 'Model']
            self.best_model = self.results[self.best_model_name]['model']
            
            print(f"\n🥇 Best Model: {self.best_model_name}")
            print(f"   F1-Score: {comparison_df.loc[best_idx, 'F1-Score']:.4f}")
            
        elif self.problem_type == "regression":
            comparison_data = []
            for model_name, results in self.results.items():
                comparison_data.append({
                    'Model': model_name,
                    'MSE': results['mse'],
                    'RMSE': results['rmse'],
                    'MAE': results['mae'],
                    'R² Score': results['r2_score']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            print(comparison_df.round(4))
            
            # Find best model based on R² score
            best_idx = comparison_df['R² Score'].idxmax()
            self.best_model_name = comparison_df.loc[best_idx, 'Model']
            self.best_model = self.results[self.best_model_name]['model']
            
            print(f"\n🥇 Best Model: {self.best_model_name}")
            print(f"   R² Score: {comparison_df.loc[best_idx, 'R² Score']:.4f}")
        
        # Create comparison visualization
        self._create_model_comparison_plot(comparison_df)
        
        return comparison_df
    
    def _create_model_comparison_plot(self, comparison_df):
        """Create visualization comparing all models."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        if self.problem_type == "classification":
            # Plot accuracy and F1-score
            x_pos = np.arange(len(comparison_df))
            
            axes[0].bar(x_pos, comparison_df['Accuracy'], alpha=0.7, label='Accuracy')
            axes[0].bar(x_pos, comparison_df['F1-Score'], alpha=0.7, label='F1-Score')
            axes[0].set_xlabel('Models')
            axes[0].set_ylabel('Score')
            axes[0].set_title('Model Performance Comparison')
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels(comparison_df['Model'], rotation=45)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot precision and recall
            axes[1].bar(x_pos, comparison_df['Precision'], alpha=0.7, label='Precision')
            axes[1].bar(x_pos, comparison_df['Recall'], alpha=0.7, label='Recall')
            axes[1].set_xlabel('Models')
            axes[1].set_ylabel('Score')
            axes[1].set_title('Precision vs Recall')
            axes[1].set_xticks(x_pos)
            axes[1].set_xticklabels(comparison_df['Model'], rotation=45)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
        else:  # regression
            x_pos = np.arange(len(comparison_df))
            
            axes[0].bar(x_pos, comparison_df['R² Score'], alpha=0.7, color='green')
            axes[0].set_xlabel('Models')
            axes[0].set_ylabel('R² Score')
            axes[0].set_title('Model R² Score Comparison')
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels(comparison_df['Model'], rotation=45)
            axes[0].grid(True, alpha=0.3)
            
            axes[1].bar(x_pos, comparison_df['RMSE'], alpha=0.7, color='red')
            axes[1].set_xlabel('Models')
            axes[1].set_ylabel('RMSE')
            axes[1].set_title('Model RMSE Comparison (Lower is Better)')
            axes[1].set_xticks(x_pos)
            axes[1].set_xticklabels(comparison_df['Model'], rotation=45)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: model_comparison.png")

# %% [markdown]
# ## Results Saving and Persistence
# 
# Save model results and trained models for future use

# %%
    def save_results(self, filename="model_results.json"):
        """
        Save all model results to a JSON file.
        
        Args:
            filename (str): Name of the results file
        """
        print(f"💾 Saving results to {filename}...")
        
        # Prepare results for JSON serialization
        json_results = {}
        for model_name, results in self.results.items():
            json_results[model_name] = {}
            for key, value in results.items():
                if key == 'model':
                    continue  # Skip model object
                elif key == 'predictions':
                    json_results[model_name][key] = value.tolist() if hasattr(value, 'tolist') else list(value)
                else:
                    json_results[model_name][key] = float(value) if value is not None else None
        
        # Add metadata
        json_results['_metadata'] = {
            'problem_type': self.problem_type,
            'best_model': self.best_model_name,
            'training_timestamp': datetime.now().isoformat(),
            'feature_count': len(self.feature_names),
            'training_samples': len(self.X_train) if self.X_train is not None else 0,
            'test_samples': len(self.X_test) if self.X_test is not None else 0
        }
        
        # Save to file
        with open(self.output_dir / filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✅ Results saved to {self.output_dir / filename}")
    
    def save_best_model(self, filename="best_model.joblib"):
        """
        Save the best performing model.
        
        Args:
            filename (str): Name of the model file
        """
        if self.best_model is None:
            print("❌ No best model identified. Run compare_models() first.")
            return
        
        print(f"💾 Saving best model ({self.best_model_name}) to {filename}...")
        
        # Save model
        joblib.dump(self.best_model, self.output_dir / filename)
        
        # Save model metadata
        metadata = {
            'model_name': self.best_model_name,
            'problem_type': self.problem_type,
            'feature_names': self.feature_names,
            'training_timestamp': datetime.now().isoformat(),
            'model_type': type(self.best_model).__name__
        }
        
        with open(self.output_dir / f"{filename.split('.')[0]}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Best model saved to {self.output_dir / filename}")

# %% [markdown]
# ## Complete Model Pipeline Execution
# 
# Execute the full modeling workflow

# %%
    def generate_model_report(self):
        """
        Generate a comprehensive model performance report.
        """
        print("🚀 STARTING COMPREHENSIVE MODEL PIPELINE")
        print("=" * 60)
        
        if self.X_train is None:
            print("❌ No training data loaded. Please load data first.")
            return
        
        # Step 1: Setup models
        print("\n1️⃣ Setting up baseline models...")
        self.setup_baseline_models()
        
        print("\n2️⃣ Setting up advanced models...")
        self.setup_advanced_models()
        
        # Step 3: Train and evaluate all models
        print("\n3️⃣ Training and evaluating models...")
        for model_name, model in self.models.items():
            print(f"\n--- Processing {model_name} ---")
            
            # Train model
            trained_model = self.train_model(model_name, model, self.X_train, self.y_train)
            
            if trained_model is not None:
                # Evaluate model
                self.evaluate_model(model_name, trained_model, self.X_test, self.y_test)
                
                # Create visualizations
                if self.problem_type == "classification":
                    y_pred = self.results[model_name]['predictions']
                    self.create_confusion_matrix_plot(self.y_test, y_pred, model_name)
                
                # Feature importance (if available)
                self.create_feature_importance_plot(trained_model, self.feature_names, model_name)
                
                # Learning curves
                self.create_learning_curves(trained_model, self.X_train, self.y_train, model_name)
                
                # Cross-validation
                self.perform_cross_validation(trained_model, self.X_train, self.y_train)
        
        # Step 4: Compare models
        print("\n4️⃣ Comparing model performance...")
        comparison_df = self.compare_models()
        
        # Step 5: Save results
        print("\n5️⃣ Saving results...")
        self.save_results()
        self.save_best_model()
        
        print(f"\n✅ MODEL PIPELINE COMPLETE!")
        print(f"📁 All results saved to: {self.output_dir}")
        print(f"🏆 Best model: {self.best_model_name}")
        
        return comparison_df

# %% [markdown]
# ## Complete Model Pipeline Function
# 
# Execute the complete modeling workflow

# %%
def run_model_pipeline(data_path=None, X_train=None, X_test=None, y_train=None, y_test=None, 
                      problem_type="classification", output_dir="../data_analysis/visualizations"):
    """
    Run the complete model pipeline.
    
    Args:
        data_path (str): Path to processed data directory
        X_train, X_test, y_train, y_test: Direct data arrays
        problem_type (str): 'classification' or 'regression'
        output_dir (str): Directory to save results
    """
    print("🚀 Starting Complete Model Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = ModelPipeline(problem_type, output_dir)
    
    # Load data
    if not pipeline.load_processed_data(data_path, X_train, X_test, y_train, y_test):
        print("❌ Failed to load data. Exiting pipeline.")
        return None
    
    # Generate comprehensive report
    results = pipeline.generate_model_report()
    
    print("\n🎉 Model pipeline completed successfully!")
    return pipeline, results

# %% [markdown]
# ## Main Execution
# 
# Execute modeling when run as standalone script

# %%
def main():
    """
    Main function to run the model pipeline.
    """
    print("Model Pipeline Module - Jupyter Compatible Format")
    print("Agent will populate this with dataset-specific modeling steps.")
    
    # Example usage (agent will customize this)
    # pipeline, results = run_model_pipeline(
    #     data_path="processed_data",
    #     problem_type="classification",
    #     output_dir="data_analysis/visualizations"
    # )
    
if __name__ == "__main__":
    main() 