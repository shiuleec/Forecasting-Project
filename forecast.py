import pandas as pd
import torch

# Load preprocessed datasets and sample submission
sales_train_validation = pd.read_csv('data/preprocessed_sales_train_validation.csv')
sample_submission = pd.read_csv('data/sample_submission.csv')

# Load the trained PatchTST model
class PatchTST(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=64):
        super(PatchTST, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

input_size = 1913
output_size = 28
patchtst_model = PatchTST(input_size, output_size)
patchtst_model.load_state_dict(torch.load('patchtst_model.pth'))

# Function to create submission file for PatchTST
def create_submission_patchtst(model, data, sample_sub):
    submission = sample_sub.copy()
    model.eval()
    with torch.no_grad():
        for idx, row in data.iterrows():
            series = torch.tensor(row[-input_size:].values, dtype=torch.float32).unsqueeze(0)
            forecast = model(series).squeeze().numpy()
            item_id = row['id']
            submission.loc[submission['id'] == item_id, 'F1':'F28'] = forecast
    return submission

# Generate forecasts for PatchTST
submission_patchtst = create_submission_patchtst(patchtst_model, sales_train_validation, sample_submission)

# Save the submission files
submission_patchtst.to_csv('submission_patchtst.csv', index=False)

print("Submission file created successfully!")

