class AttentiveStatisticsPooling(nn.Module):

    def __init__(self, input_dim, attention_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        
        self.attention = nn.Sequential(
            nn.Conv1d(input_dim, attention_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(attention_dim),
            nn.Tanh(),
            nn.Conv1d(attention_dim, input_dim, kernel_size=1),
            nn.Softmax(dim=2)
        )
        
    def forward(self, x):

        batch_size, channels, freq, time = x.shape
        
        x_reshaped = x.view(batch_size, channels * freq, time)
        
        attn_weights = self.attention(x_reshaped)  
        
        mean = torch.sum(x_reshaped * attn_weights, dim=2)  
        residuals = x_reshaped - mean.unsqueeze(2)
        std = torch.sqrt(torch.sum(residuals**2 * attn_weights, dim=2) + 1e-6)
        
        pooled = torch.cat([mean, std], dim=1)
        
        return pooled
