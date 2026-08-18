import torch
import torch.nn as nn

class FaultCNNLSTM(nn.Module):
    def __init__(self, in_channels=6, num_classes=11):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.fc_class = nn.Linear(128, num_classes) 
        self.fc_loc = nn.Linear(128, 1)    

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (hn, _) = self.lstm(x)
        h = hn[-1] 
        return self.fc_class(h), self.fc_loc(h).squeeze(-1)