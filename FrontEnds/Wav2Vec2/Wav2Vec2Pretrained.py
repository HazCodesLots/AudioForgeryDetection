class Wav2Vec2Frontend(nn.Module):

    def __init__(self, model_name='facebook/wav2vec2-xls-r-300m', output_dim=768, freeze_encoder=False, use_conv_projection=True):
        super(Wav2Vec2Frontend, self).__init__()

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.hidden_dim = self.wav2vec2.config.hidden_size

        if freeze_encoder:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False

        self.use_conv_projection = use_conv_projection
        if use_conv_projection:
            self.projection = nn.Conv1d(self.hidden_dim, output_dim, kernel_size=1, bias=True)
        else:
            self.projection = nn.Linear(self.hidden_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)

        outputs = self.wav2vec2(x, output_hidden_states=False)
        features = outputs.last_hidden_state

        if self.use_conv_projection:
            features = features.transpose(1, 2)
            features = self.projection(features)
        else:
            features = self.projection(features)
            features = features.transpose(1,2)
        return features
