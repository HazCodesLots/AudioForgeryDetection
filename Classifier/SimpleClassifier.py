class SimpleClassifier(nn.Module):

    def __init__(self, input_dim, num_classes=2):

        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        
        logits = self.fc(x)
        return logits
