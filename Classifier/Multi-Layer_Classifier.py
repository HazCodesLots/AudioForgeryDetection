class MLPClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout_rate=0.3, num_classes=2):
        super().__init__()

        layers = []
        last_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.mlp(x)
        return logits