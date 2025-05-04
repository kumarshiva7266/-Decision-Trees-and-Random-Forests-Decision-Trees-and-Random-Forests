# Decision Trees and Random Forests Analyzer

A modern GUI application for analyzing and visualizing Decision Trees and Random Forests models.

## Features

- Load and process CSV datasets
- Train Decision Tree and Random Forest models
- Visualize decision trees and feature importance
- Adjust model parameters (max depth, min samples split)
- View model performance metrics
- Interactive visualizations

## Installation

1. Make sure you have Python 3.7+ installed
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Using the application:
   - Click "Load Dataset" to import your CSV file
   - Select the model type (Decision Tree or Random Forest)
   - Adjust model parameters as needed
   - Click "Train Model" to train and visualize the results

## Dataset Format

Your CSV file should have:
- Features in all columns except the last one
- Target variable in the last column
- No missing values
- Numerical data (categorical variables should be encoded)

## Example

1. Load a dataset (e.g., iris.csv)
2. Select "Decision Tree" as the model type
3. Set max depth to 5
4. Click "Train Model"
5. View the tree visualization and feature importance plot

## Requirements

- Python 3.7+
- PyQt5
- scikit-learn
- pandas
- numpy
- matplotlib
- graphviz
- seaborn 