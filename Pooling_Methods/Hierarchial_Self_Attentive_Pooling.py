
class HierarchicalSelfAttentivePooling(nn.Module):
    def __init__(self, input_dim, attention_dim=512, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.dropout_rate = dropout_rate

        self.input_proj = nn.Linear(input_dim, attention_dim)

        self.freq_query = nn.Linear(attention_dim, attention_dim)
        self.freq_key = nn.Linear(attention_dim, attention_dim)
        self.freq_value = nn.Linear(attention_dim, attention_dim)
        self.freq_attn_dropout = nn.Dropout(dropout_rate)
        self.freq_out_proj = nn.Linear(attention_dim, attention_dim)
        self.freq_out_dropout = nn.Dropout(dropout_rate)
        self.freq_layer_norm = nn.LayerNorm(attention_dim)

        self.time_query = nn.Linear(attention_dim, attention_dim)
        self.time_key = nn.Linear(attention_dim, attention_dim)
        self.time_value = nn.Linear(attention_dim, attention_dim)
        self.time_attn_dropout = nn.Dropout(dropout_rate)
        self.time_out_proj = nn.Linear(attention_dim, attention_dim)
        self.time_out_dropout = nn.Dropout(dropout_rate)
        self.time_layer_norm = nn.LayerNorm(attention_dim)

    def _multihead_attention(self, Q, K, V, attn_dropout):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

    def forward(self, x):
        batch, channels, freq, time = x.shape
        x_reshaped = x.permute(0, 3, 2, 1).contiguous().view(batch * time, freq, channels)
        x_proj = self.input_proj(x_reshaped)

        Qf = self.freq_query(x_proj).view(batch * time, freq, self.num_heads, self.head_dim).transpose(1, 2)
        Kf = self.freq_key(x_proj).view(batch * time, freq, self.num_heads, self.head_dim).transpose(1, 2)
        Vf = self.freq_value(x_proj).view(batch * time, freq, self.num_heads, self.head_dim).transpose(1, 2)

        freq_attended, freq_attn_weights = self._multihead_attention(Qf, Kf, Vf, self.freq_attn_dropout)
        freq_attended = freq_attended.transpose(1, 2).contiguous().view(batch * time, freq, self.attention_dim)
        freq_out = self.freq_out_proj(freq_attended)
        freq_out = self.freq_out_dropout(freq_out)
        freq_out = self.freq_layer_norm(freq_out + x_proj)

        freq_pooled = freq_out.mean(dim=1).view(batch, time, self.attention_dim)

        Qt = self.time_query(freq_pooled).view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
        Kt = self.time_key(freq_pooled).view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
        Vt = self.time_value(freq_pooled).view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)

        time_attended, time_attn_weights = self._multihead_attention(Qt, Kt, Vt, self.time_attn_dropout)
        time_attended = time_attended.transpose(1, 2).contiguous().view(batch, time, self.attention_dim)
        time_out = self.time_out_proj(time_attended)
        time_out = self.time_out_dropout(time_out)
        time_out = self.time_layer_norm(time_out + freq_pooled)

        pooled = time_out.mean(dim=1)
        return pooled
