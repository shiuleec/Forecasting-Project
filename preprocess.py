#Data Loading and Preprocessing
import pandas as pd

# Load datasets
calendar = pd.read_csv('/data/calendar.csv')
sales_train_validation = pd.read_csv('/data/sales_train_validation.csv')
sell_prices = pd.read_csv('/data/sell_prices.csv')
sample_submission = pd.read_csv('/data/sample_submission.csv')

# Display the structure of each loaded dataset
print(calendar.head())
print(sales_train_validation.head())
print(sell_prices.head())
print(sample_submission.head())


