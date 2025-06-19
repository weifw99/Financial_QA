import torch
import torch.nn as nn

class DeepCNNTimeSeriesWithAttention(torch.nn.Module):
    def __init__(self, num_features, num_timesteps=1):
        super(DeepCNNTimeSeriesWithAttention, self).__init__()
        
        # Convolutional layers
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm([64, num_timesteps]),
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LayerNorm([128, num_timesteps]),
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.LayerNorm([256, num_timesteps]),
            
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.LayerNorm([512, num_timesteps]),
            
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.LayerNorm([1024, num_timesteps])
        )
        
        # Self-attention layer
        self.attention = nn.MultiheadAttention(embed_dim=1024, num_heads=8)
        
        # MLP with residual connections
        self.mlp = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, num_timesteps, num_features)
        batch_size, timesteps, features = x.shape
        
        # Reshape to (batch_size, features, timesteps) for Conv1d
        x = x.permute(0, 2, 1)  # (B, F, T)
        
        # Apply convolutional layers
        conv_out = self.convs(x)  # (B, 1024, T)
        
        # Reshape back to (batch_size, timesteps, features)
        conv_out = conv_out.permute(0, 2, 1)  # (B, T, 1024)
        
        # Apply self-attention
        attn_output, _ = self.attention(
            conv_out.transpose(0, 1),  # (T, B, 1024)
            conv_out.transpose(0, 1),
            conv_out.transpose(0, 1)
        )
        attn_output = attn_output.transpose(0, 1)  # (B, T, 1024)
        
        # Combine with original features
        combined = conv_out + attn_output
        
        # Global average pooling over time dimension
        pooled = combined.mean(dim=1)  # (B, 1024)
        
        # MLP prediction
        output = self.mlp(pooled)  # (B, 1)
        return output

model_cls = DeepCNNTimeSeriesWithAttention