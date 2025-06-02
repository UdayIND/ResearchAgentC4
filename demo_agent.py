#!/usr/bin/env python3
"""
Autonomous Research Agent Demo
=============================

This script demonstrates how to use the autonomous research agent
with example data and shows the complete workflow including:

1. Literature review
2. Model discovery  
3. Data analysis with visualization
4. Jupyter-compatible notebook generation
5. Programmatic visualization analysis

Usage:
    python demo_agent.py

The demo will create sample data and run the complete research workflow.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from agent_core import AutonomousResearchAgent

def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    print("📊 Creating sample dataset for demonstration...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create a sample classification dataset
    n_samples = 1000
    
    # Generate features
    feature_1 = np.random.normal(0, 1, n_samples)
    feature_2 = np.random.exponential(2, n_samples)
    feature_3 = np.random.uniform(-1, 1, n_samples)
    
    # Create some correlation
    feature_4 = feature_1 * 0.7 + np.random.normal(0, 0.3, n_samples)
    
    # Generate categorical features
    category_A = np.random.choice(['Type_1', 'Type_2', 'Type_3'], n_samples, p=[0.5, 0.3, 0.2])
    category_B = np.random.choice(['Group_X', 'Group_Y'], n_samples)
    
    # Generate target variable with some relationship to features
    target_prob = 1 / (1 + np.exp(-(feature_1 + feature_2 * 0.3 - feature_3 * 0.5)))
    target = np.random.binomial(1, target_prob, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'numerical_feature_1': feature_1,
        'numerical_feature_2': feature_2, 
        'numerical_feature_3': feature_3,
        'correlated_feature': feature_4,
        'categorical_A': category_A,
        'categorical_B': category_B,
        'target': target
    })
    
    # Add some missing values for realism
    missing_indices = np.random.choice(n_samples, size=50, replace=False)
    data.loc[missing_indices, 'numerical_feature_3'] = np.nan
    
    # Save the dataset
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    dataset_path = data_dir / 'sample_classification_dataset.csv'
    data.to_csv(dataset_path, index=False)
    
    print(f"✅ Sample dataset created: {dataset_path}")
    print(f"   - Shape: {data.shape}")
    print(f"   - Features: {list(data.columns)}")
    print(f"   - Target distribution: {data['target'].value_counts().to_dict()}")
    
    return str(dataset_path)

def run_demo():
    """Run the complete autonomous research agent demonstration."""
    print("🤖 Autonomous Research Agent Demo")
    print("=" * 50)
    
    # Create sample dataset
    dataset_path = create_sample_dataset()
    
    # Define research topic
    research_topic = "machine learning classification"
    
    print(f"\n🔬 Research Topic: {research_topic}")
    print(f"📁 Dataset: {dataset_path}")
    
    # Initialize the agent
    print("\n🚀 Initializing Autonomous Research Agent...")
    agent = AutonomousResearchAgent(project_root=".")
    
    # Run the complete research workflow
    print("\n📋 Starting Research Workflow...")
    print("This will demonstrate:")
    print("  1. Literature review and paper discovery")
    print("  2. Model and repository discovery")
    print("  3. Comprehensive data analysis")
    print("  4. Visualization generation and programmatic analysis")
    print("  5. Jupyter-compatible notebook creation")
    
    # Execute the workflow
    results = agent.run_research_workflow(research_topic, dataset_path)
    
    # Display results
    print("\n" + "=" * 50)
    if results['success']:
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total execution time: {results['workflow_time']:.2f} seconds")
        print(f"📚 Literature papers found: {results['literature_review']['papers_found']}")
        print(f"🤖 HuggingFace models discovered: {len(results['model_discovery']['hf_models'])}")
        print(f"📊 Visualizations generated: {len(results['data_analysis']['visualizations_generated'])}")
        print(f"📓 Jupyter notebooks created: {len(results['notebooks_generated'])}")
        
        print("\n📂 Generated Files:")
        print("  📄 research_logs/development_log.md - Complete workflow log")
        print("  📄 summaries/papers_summary.md - Literature review results")
        print("  📄 summaries/model_discoveries.md - Model discovery results")
        print("  📄 data_analysis/dataset_overview.md - Data analysis summary")
        print("  📄 agent_workspace/exploratory_data_analysis.py - Jupyter-compatible notebook")
        print("  🖼️  data_analysis/visualizations/ - Generated visualization images")
        
        print("\n🎯 Key Features Demonstrated:")
        print("  ✅ Autonomous literature review")
        print("  ✅ Model and repository discovery")
        print("  ✅ Comprehensive EDA with visualizations")
        print("  ✅ Programmatic visualization analysis")
        print("  ✅ Jupyter-compatible notebook format (# %% cells)")
        print("  ✅ Complete documentation and logging")
        
        print("\n📋 Next Steps:")
        print("  1. Review the generated Jupyter-compatible notebook:")
        print("     agent_workspace/exploratory_data_analysis.py")
        print("  2. Copy cells to Kaggle notebook for reproduction")
        print("  3. Examine generated visualizations in data_analysis/visualizations/")
        print("  4. Review research logs for complete workflow documentation")
        
    else:
        print("❌ DEMO FAILED")
        print(f"Error: {results['error']}")
        print(f"Time before failure: {results['workflow_time']:.2f} seconds")
    
    print("\n" + "=" * 50)

def show_notebook_example():
    """Show an example of the generated Jupyter-compatible notebook format."""
    print("\n📓 Example Jupyter-Compatible Notebook Format:")
    print("-" * 40)
    
    example_notebook = '''# %% [markdown]
# # Exploratory Data Analysis
# 
# This notebook demonstrates the autonomous research agent's approach to data analysis
# with explicit visualization handling and programmatic image analysis.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('data/sample_classification_dataset.csv')
print(f"Dataset shape: {data.shape}")

# %% [markdown]
# **Execution Status**: ✅ Success
# **Execution Time**: 0.15 seconds
# 
# **Output**:
# ```
# Dataset shape: (1000, 7)
# ```

# %%
# Generate data overview visualization
plt.figure(figsize=(12, 8))
data.hist(bins=20, alpha=0.7)
plt.tight_layout()
plt.savefig('data_analysis/visualizations/data_distribution.png', dpi=300)
plt.close()
print("Visualization saved as 'data_distribution.png'")

# %% [markdown]
# **Visualization Generated**: `data_distribution.png`
# 
# The agent has generated and saved a visualization as `data_distribution.png`.

# %% [markdown]
# **Agent Action Required**:
# - The agent must now explicitly load and analyze the generated visualization image file programmatically.
# - After analyzing the image, the agent must document its interpretations of the visual patterns observed.

# %%
# Programmatic analysis of visualization: data_distribution.png
from PIL import Image
import numpy as np
import os

try:
    if os.path.exists('data_analysis/visualizations/data_distribution.png'):
        img = Image.open('data_analysis/visualizations/data_distribution.png')
        img_array = np.array(img)
        
        # Basic image analysis
        height, width = img_array.shape[:2]
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        
        print(f"Image Analysis:")
        print(f"- Dimensions: {width} x {height} pixels")
        print(f"- Average brightness: {brightness:.2f}")
        print(f"- Contrast (std dev): {contrast:.2f}")
        print("- Analysis: High contrast detected - clear, well-defined visualization")
        
except Exception as e:
    print(f"Error analyzing visualization: {e}")

# %% [markdown]
# **Agent's Visual Analysis Summary**:
# 
# Based on the programmatic analysis of `data_distribution.png`:
# 
# - The visualization displays clear data distributions across all features
# - High contrast values indicate clear, interpretable visualizations
# - The analysis confirms the visualization quality is suitable for research documentation
# 
# **Implications and Next Steps**:
# - The agent will use these insights to inform subsequent modeling decisions
# - Clear distributions suggest appropriate preprocessing strategies
'''
    
    print(example_notebook)
    print("-" * 40)
    print("This format allows easy copying to Kaggle notebooks (.ipynb) for reproduction!")

if __name__ == "__main__":
    # Show the notebook format example first
    show_notebook_example()
    
    # Run the complete demo
    run_demo()
    
    print("\n🎉 Demo completed! Check the generated files to see the autonomous research agent in action.") 