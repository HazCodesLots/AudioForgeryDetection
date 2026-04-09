class TemporalAveragePooling(nn.Module):

    def __init__(self):
        super().__init__()
        
    def forward(self, x):

        batch_size, channels, freq, time = x.shape
        
        x_reshaped = x.view(batch_size, channels * freq, time)
        
        pooled = torch.mean(x_reshaped, dim=2)
        
        return pooled