class SelfAttentivePooling(nn.Module):
    def __init__(self, input_dim, attention_dim=512, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.dropout_rate = dropout_rate

        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)

        self.attn_dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(attention_dim, attention_dim)
        self.out_dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(attention_dim)
        self.input_proj = nn.Linear(input_dim, attention_dim)

    def forward(self, x):
        batch_size, channels, freq, time = x.shape
        x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(batch_size, freq * time, channels)

        Q = self.query(x_reshaped)
        K = self.key(x_reshaped)
        V = self.value(x_reshaped)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.attention_dim)

        out = self.out_proj(attended)
        out = self.out_dropout(out)

        x_proj = self.input_proj(x_reshaped)
        out = self.layer_norm(out + x_proj)

        pooled = out.mean(dim=1)
        return pooled