# Forecasting-Project
This project aims to forecast product sales using time series analysis and machine learning techniques, including DeepAR and PatchTST models.
## Data Description

- `calendar.csv`: Contains information about the dates on which the products are sold.
- `sales_train_validation.csv`: Historical daily unit sales data per product and store.
- `sell_prices.csv`: Contains information about the price of the products sold per store and date.
- `sample_submission.csv`: Sample submission file format.

## Installation Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   pip install -r requirements.txt
   python scripts/preprocess.py
   python scripts/train_deepar.py
   python scripts/train_patchtst.py
   python scripts/forecast.py
# File Descriptions:
preprocess.py: Script for data loading and preprocessing.
train_deepar.py: Script for training the DeepAR model.
train_patchtst.py: Script for training the PatchTST model.
forecast.py: Script for generating forecasts and creating the submission files.
EDA.ipynb: Jupyter notebook for exploratory data analysis.
ModelTraining.ipynb: Jupyter notebook for model training and evaluation.
submission_deepar.csv: The generated submission file using DeepAR.
submission_patchtst.csv: The generated submission file using PatchTST.
# Assumptions
The data files are placed in the data/ directory.
The preprocessing steps are minimal to keep the focus on the modeling process.
The DeepAR and PatchTST models are used for demonstrating deep learning techniques for time series forecasting.
