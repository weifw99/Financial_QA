import torch
import torch.nn as nn
import torch.nn.functional as F

class BidirectionalGRU3L256_TCN_LN_GELU_Attention(torch.nn.Module):
    def __init__(self, num_features, num_timesteps):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=128, kernel_size=3, padding=64, dilation=64),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(128, 256, 3, padding=128, dilation=128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(256, 256, 3, padding=256, dilation=256),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.gru = nn.GRU(input_size=256, hidden_size=256, num_layers=3, bidirectional=True, batch_first=True)
        self.norm_tcn = nn.LayerNorm(256)
        self.norm_gru = nn.LayerNorm(512)
        self.residual_proj = nn.Linear(512, 512)
        self.attention = nn.Linear(512, 1)
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, num_timesteps, num_features)
        x = x.transpose(1, 2)  # (batch, features, timesteps)
        tcn_out = self.tcn(x)  # (batch, 256, timesteps)
        tcn_out = tcn_out.transpose(1, 2)  # (batch, timesteps, 256)
        tcn_out = self.norm_tcn(tcn_out)
        gru_out, _ = self.gru(tcn_out)

        # Residual connection
        residual = self.residual_proj(gru_out)
        gru_out = gru_out + residual

        gru_out = self.norm_gru(gru_out)

        # Attention pooling
        attention_weights = torch.softmax(self.attention(gru_out), dim=1)  # (batch, timesteps, 1)
        weighted_sum = torch.sum(gru_out * attention_weights, dim=1)  # (batch, 512)

        output = self.mlp(weighted_sum)
        return output

model_cls = BidirectionalGRU3L256_TCN_LN_GELU_Attention