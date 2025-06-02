# Data Directory

This directory is where you should place your datasets for the Autonomous Research Agent to analyze.

## Directory Structure

```
data/
├── raw/          # Place your original datasets here
├── processed/    # Processed data will be saved here automatically
└── README.md     # This file
```

## Supported File Formats

The agent supports the following data formats:
- **CSV files** (`.csv`) - Most common format
- **Excel files** (`.xlsx`, `.xls`) - Spreadsheet format
- **JSON files** (`.json`) - Structured data format
- **Parquet files** (`.parquet`) - Columnar storage format

## How to Use

1. **Place your dataset** in the `data/raw/` folder
2. **Run the agent** with your dataset path:
   ```python
   python demo_agent.py --data_path "data/raw/your_dataset.csv" --target_column "your_target_column"
   ```
3. **Check results** in the generated folders:
   - `data_analysis/` - EDA visualizations and analysis
   - `agent_workspace/` - Generated Jupyter notebooks
   - `final_paper/` - Research paper output

## Example Datasets

For testing purposes, you can use publicly available datasets:
- Iris dataset (classification)
- Boston Housing (regression)
- Titanic dataset (classification)
- Any CSV with numerical and categorical features

## Dataset Requirements

- **Minimum**: 100 rows recommended
- **Features**: Mix of numerical and categorical features works best
- **Target column**: Specify the column you want to predict (for supervised learning)
- **Clean headers**: Column names should be descriptive and without special characters

## Notes

- The agent will automatically detect data types and handle missing values
- Large datasets (>1GB) may take longer to process
- Ensure your dataset fits in memory for optimal performance 