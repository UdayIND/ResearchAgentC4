# Autonomous Research Agent (OpenAPI ADK)

## Overview

This project implements a **fully autonomous research agent** that can independently perform end-to-end AI research tasks – from literature review to model development to writing scientific papers – with minimal user input. The agent leverages modern LLM frameworks and specialized tools to accept a project focus and local dataset, then plan and execute a complete research pipeline.

## 🎯 Key Innovation: Jupyter-Compatible Notebook Format

The agent generates all analysis code in **Jupyter-compatible notebook syntax** using `# %%` and `# %% [markdown]` cells, specifically designed for:

### ✅ **Easy Reproducibility on Kaggle**
- Direct copy-paste into Kaggle notebooks (.ipynb)
- Precise reproduction and verification of results
- Seamless transition from development to publication

### ✅ **Clear and Structured Documentation**
- Markdown cells document each step's purpose and rationale
- Transparent workflow for understanding and review
- Professional research documentation standards

### ✅ **Enhanced AI-Assisted Analysis**
- Cursor IDE's AI better understands structured plaintext format
- Improved relevance and accuracy of AI assistance
- Better code completion and suggestions

## 🔬 Autonomous Visualization Analysis

The agent **explicitly handles visualization interpretation** since Cursor IDE cannot directly render images for AI analysis:

### 1. **Generate and Save Visualizations**
```python
# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
data.hist()
plt.savefig('data_analysis/visualizations/data_distribution.png')
```

### 2. **Document Visualization Generation**
```python
# %% [markdown]
# **Visualization Generated**: "data_distribution.png"
#
# The agent generated a histogram visualization and saved it as `data_distribution.png`.
```

### 3. **Programmatic Image Analysis**
```python
# %%
from PIL import Image
import numpy as np

# Load and analyze the visualization programmatically
img = Image.open('data_analysis/visualizations/data_distribution.png')
img_array = np.array(img)
brightness = np.mean(img_array)
contrast = np.std(img_array)
print(f"Analysis: High contrast ({contrast:.2f}) indicates clear visualization")
```

### 4. **Document Visual Insights**
```python
# %% [markdown]
# **Agent's Visual Analysis Summary**:
# - The visualization displays clear data distribution patterns
# - High contrast values indicate interpretable visualizations
# - Skewness suggests data normalization may be beneficial
```

## Features

- **🤖 Automated Research Workflow**: Literature search, EDA, model development, and result analysis
- **🔧 Tool-Assisted Knowledge Gathering**: Brave Search, Hugging Face Hub, GitHub API, PDF parsing
- **⏱️ Iterative Planning & Execution**: 4-hour runtime with periodic reassessment
- **💻 Code Generation & Execution**: Real-time Python code writing and execution with output analysis
- **📊 Autonomous Visualization Analysis**: Programmatic image analysis and interpretation
- **📝 Continuous Logging**: Detailed progress tracking and decision documentation
- **📄 LaTeX Paper Generation**: Publication-ready scientific papers
- **📓 Jupyter Compatibility**: All code in `# %%` format for easy Kaggle reproduction

## Project Structure

```
project_root/
├── agent_core.py                  # Main autonomous agent implementation
├── demo_agent.py                  # Complete demonstration script
├── requirements.txt               # Python dependencies
├── config/
│   └── agent_config.yaml          # Agent configuration settings
├── agent_workspace/               # Generated Jupyter-compatible notebooks
│   ├── preprocessing.py           # Data preprocessing (# %% format)
│   ├── eda.py                     # Exploratory data analysis (# %% format)
│   ├── model_pipeline.py          # Model training pipeline (# %% format)
│   └── exploratory_data_analysis.py  # Complete EDA notebook
├── research_logs/
│   ├── development_log.md         # Chronological progress log
│   └── final_summary.md           # Research summary
├── summaries/
│   ├── papers_summary.md          # Literature review with citations
│   └── model_discoveries.md       # Model and repository findings
├── data_analysis/
│   ├── dataset_overview.md        # Dataset analysis summary
│   └── visualizations/           # Generated visualization images
├── final_paper/
│   └── main.tex                   # LaTeX research paper
├── tools/                         # Tool implementations
│   ├── web_search.py              # Brave Search integration
│   ├── huggingface_search.py      # HuggingFace Hub search
│   ├── github_search.py           # GitHub repository search
│   ├── pdf_parser.py              # PDF text extraction
│   ├── file_operations.py         # Safe file operations
│   └── python_executor.py         # Code execution engine
└── README.md                      # This file
```

## Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repository-url>
cd ResearchAgent-C4

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Demo
```bash
# Run the complete demonstration
python demo_agent.py
```

### 3. Custom Research
```bash
# Run with your own topic and dataset
python agent_core.py --topic "your_research_topic" --data_path "path/to/your/data.csv"
```

## Example Usage

The demo creates a sample dataset and runs the complete workflow:

```python
from agent_core import AutonomousResearchAgent

# Initialize the agent
agent = AutonomousResearchAgent(project_root=".")

# Run complete research workflow
results = agent.run_research_workflow(
    topic="machine learning classification",
    dataset_path="data/sample_dataset.csv"
)

# Results include:
# - Literature review findings
# - Model discoveries from HuggingFace/GitHub
# - Comprehensive data analysis
# - Jupyter-compatible notebooks
# - Programmatically analyzed visualizations
```

## Generated Outputs

### 📓 Jupyter-Compatible Notebooks
All analysis code uses the `# %%` cell format:
```python
# %% [markdown]
# # Data Loading and Initial Analysis
# This section loads the dataset and performs initial inspection.

# %%
import pandas as pd
import numpy as np
data = pd.read_csv('dataset.csv')
print(f"Dataset shape: {data.shape}")

# %% [markdown]
# **Execution Status**: ✅ Success
# **Output**: Dataset shape: (1000, 7)
```

### 🖼️ Visualization Analysis Workflow
```python
# %%
# Generate visualization
plt.figure(figsize=(12, 8))
data.hist(bins=20)
plt.savefig('data_analysis/visualizations/distribution.png')

# %% [markdown]
# **Visualization Generated**: `distribution.png`

# %%
# Programmatic analysis
from PIL import Image
img = Image.open('data_analysis/visualizations/distribution.png')
analysis = analyze_image_properties(img)
print(f"Contrast: {analysis['contrast']:.2f}")

# %% [markdown]
# **Visual Analysis**: High contrast indicates clear, interpretable distributions
```

## Dependencies

Core requirements (see `requirements.txt` for complete list):
- **Data Science**: pandas, numpy, matplotlib, seaborn, scikit-learn
- **Image Analysis**: Pillow, opencv-python (for visualization interpretation)
- **Web APIs**: requests, beautifulsoup4
- **PDF Processing**: PyMuPDF, pdfminer.six
- **Deep Learning**: torch, transformers (optional)
- **Jupyter Compatibility**: ipython

## Configuration

Customize the agent behavior in `config/agent_config.yaml`:
```yaml
agent:
  max_runtime_hours: 4
  notebook_format: "jupyter_compatible"

visualization:
  analysis_required: true
  default_format: "png"
  save_directory: "data_analysis/visualizations"
```

## Agent Capabilities

The agent can handle various research domains:
- **Computer Vision**: Image classification, object detection
- **Natural Language Processing**: Text classification, sentiment analysis  
- **Medical AI**: Diagnostic models, biomarker discovery
- **Time Series Analysis**: Forecasting, anomaly detection
- **General ML**: Classification, regression, clustering

## Safety and Limitations

- ✅ Sandboxed execution environment with controlled file access
- ✅ 4-hour runtime limit to prevent infinite loops
- ✅ All web searches and API calls logged for transparency
- ✅ Generated code is modular and reviewable
- ✅ Programmatic visualization analysis (no external image viewing required)

## Why This Format?

This implementation specifically addresses the need for:

1. **Kaggle Reproducibility**: Direct copy-paste to .ipynb notebooks
2. **Transparent Documentation**: Every step clearly explained in markdown
3. **AI-Enhanced Development**: Better Cursor IDE integration
4. **Autonomous Visualization**: No human intervention needed for image interpretation
5. **Professional Standards**: Research-grade documentation and code quality

## License

MIT License - See LICENSE file for details 