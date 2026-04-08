class AASIST3OutputHead(nn.Module):
    def __init__(self, hidden_dim, num_classes=2, use_intermediate=False, intermediate_dim=256):
        super(AASIST3OutputHead, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_intermediate = use_intermediate

        if use_intermediate:
            self.intermediate_kan = KANLayer(hidden_dim, intermediate_dim)
            self.output_kan = KANLayer(hidden_dim, num_classes)

        def forward(self, hidden_features):
            if self.use_intermediate:
                x = self.intermediate_kan(hidden_features)
                logits = self.output_kan(x)
            else:
                logits = self.output_kan(hidden_features)

            return logits


class AASIST3OutputWithEmbedding(nn.Module):
    def __init__(self, hidden_dim, embedding_dim=256, num_classes=2):
        super(AASIST3OutputWithEmbedding, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim - embedding_dim
        self.num_classes = num_classes

        self.embedding_kan = KANLayer(hidden_dim, embedding_dim)
        self.classifier_kan = KANLayer(embedding_dim, num_classes)
        self.bn_embedding = nn.BatchNorm1d(embedding_dim)

    def forwad(self, hidden_features, return_embedding=False):
        embedding = self.embedding_kanh(hidden_features)
        embedding = self.bn_embedding(embedding)
        
        logits = self.classifier_kan(embedding)
        if return_embedding:
            return logits, embedding
        else:
            return logits


class AASSIT3MultiTaskOutput(nn.Module):
    def __init__(self, hidden_dim, num_attack_types=19, use_quality_head=False):
        super(AASSIT3MultiTaskOutput, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_attack_types = num_attack_types
        self.use_quality_head = use_quality_head

        self.binary_head = KANLayer(hidden_dim, 2)
        self.attack_head = KANLayer(hidden_dim, num_attack_types)

        if use_quality_head:
            self.quality_head = KANLayer(hidden_dim, 1)

    def forward(self, hidden_features):
        outputs = {}
        outputs['binary'] = self.binary_head(hidden_features)
        outputs['attack_type'] = self.attack_type_head(hidden_features)
        if self.use_quality_head:
            outputs['quality'] = self.quality_head(hidden_features)
        return outputs
