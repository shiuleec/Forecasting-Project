import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Load preprocessed datasets
sales_train_validation = pd.read_csv('data/preprocessed_sales_train_validation.csv')

# Prepare the data for PatchTST
class TimeSeriesDataset(Dataset):
    def __init__(self, df, seq_len=1913):
        self.df = df
        self.seq_len = seq_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        series = self.df.iloc[idx, -self.seq_len:].values
        return torch.tensor(series, dtype=torch.float32)

seq_len = 1913
dataset = TimeSeriesDataset(sales_train_validation, seq_len)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the PatchTST model
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

input_size = seq_len
output_size = 28
model = PatchTST(input_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data[:, -output_size:])
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'patchtst_model.pth')

