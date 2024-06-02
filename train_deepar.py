import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator

# Prepare the data for DeepAR
def prepare_data_deepar(df, start_date, freq="D"):
    timeseries = []
    for _, row in df.iterrows():
        ts = row[sales_cols].values
        timeseries.append({"start": start_date, "target": ts})
    return ListDataset(timeseries, freq=freq)

start_date = calendar['date'].iloc[0]
sales_cols = [col for col in sales_train_validation.columns if 'd_' in col]
train_ds = prepare_data_deepar(sales_train_validation, start_date)

# Load preprocessed datasets
calendar = pd.read_csv('data/preprocessed_calendar.csv')
sales_train_validation = pd.read_csv('data/preprocessed_sales_train_validation.csv')

# Prepare the data for DeepAR
def prepare_data_deepar(df, start_date, freq="D"):
    timeseries = []
    for _, row in df.iterrows():
        ts = row[sales_cols].values
        timeseries.append({"start": start_date, "target": ts})
    return ListDataset(timeseries, freq=freq)

start_date = calendar['date'].iloc[0]
sales_cols = [col for col in sales_train_validation.columns if 'd_' in col]
train_ds = prepare_data_deepar(sales_train_validation, start_date)

# Define and train DeepAR model
estimator = DeepAREstimator(freq="D", prediction_length=28, trainer=Trainer(epochs=10))
predictor = estimator.train(train_ds)

# Save the trained model
predictor.serialize("deepar_model")