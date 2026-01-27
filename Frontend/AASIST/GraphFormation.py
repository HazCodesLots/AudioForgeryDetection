class GraphPositionalEmbedding(nn.Module):
    def __init__(self, num_nodes, embedding_dim):

        super(PositionalEmbedding, self).__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim

        self.pos_embedding = nn.Parameter(torch.randn(1, num_nodes, embedding_dim))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, x):
        batch_size = x.size(0)
        pos_emb = self.pos_embedding.expand(batch_size, -1, -1)
        return x + pos_emb


class GraphFormation(nn.Module):

    def __init__(self, encoder_dim, num_temporal_nodes = 100, , num_spatial_nodes = 100, temporal_dim = 64, spatial_dim = 64, pool_ratio = 0.5, temperature = 1.0):

        super(GraphFormation, self).__init__()
        self.encoder_dim = encoder_dim
        self.num_temporal_nodes = num_temporal_nodes
        self.num_spatial_nodes = num_spatial_nodes
        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self.temporal_projection = nn.Linear(encoder_dim, temporal_dim)
        self.spatial_projection = nn.Linear(encoder_dim, spatial_dim)
        self.pe_temporal = PositionalEmbedding(num_temporal_nodes, temporal_dim)
        self.pe_spatial = PositionalEmbedding(num_spatial_nodes, spatial_dim)

        self.kan_gal_temporal = KAN_GAL(in_dim=temporal_dim, out_dim=temporal_dim, num_nodes=num_temporal_nodes, temperature=temperature)
        self.kan_gal_spatial = KAN_GAL(in_dim=spatial_dim, out_dim=spatial_dim, num_nodes=num_spatial_nodes, temperature=temperature)
        self.kan_pool_temporal = KAN_GraphPool(in_dim=temporal_dim, ratio=pool_ratio)
        self.kan_pool_spatial = KAN_GraphPool(in_dim=spatial_dim, ratio=pool_ratio)

        self.pooled_temporal_nodes = max(1, int(num_temporal_nodes * pool_ratio))
        self.pooled_spatial_nodes = max(1, int(num_spatial_nodes * pool_ratio))

    def _temporal_max_pooling(self, x):
        batch_size, channels, time = x.shape
        x_abs = torch.abs(x)
        pooled = F.adaptive_max_pool1d(x_abs, self.num_temporal_nodes)
        temporal_features = pooled.transpose(1,2)
        return temporal_features

    def _spatial_max_pooling(self, x):
        batch_size, channels, time = x.shape
        x_abs = torch.abs(x)
        x_transposed = x_abs.transpose(1,2)
        pooled = f.adaptive_max_pool1d(x_transposed.transpose(1,2), self.num_spatial_nodes)
        spatial_features = pooled.transpose(1,2)
        return spatial_features

    
    def forward(self, encoder_output):

        batch_size = encoder_output.size(0)

        temporal_features = self._temporal_max_pooling(encoder_output)
        temporal_features = self._temporal_projection(temporal_features)
        temporal_features = self.pe_temporal(temporal_features)
        temporal_graph = self.kan_gal_temporal(temporal_features)
        h_t = self.kan_pool_temporal(temporal_graph, k=self.pooled_temporal_nodes)
        
        spatial_features = self._spatial_max_pooling(encoder_output)
        spatial_features = self.spatial_projection(spatial_features)
        spatial_features = self.pe_spatial(spatial_features)
        spatial_graph = self.kan_gal_spatial(spatial_features)
        h_s = self.kan_pool_spatial(spatial_graph, k=self.pooled_spatial_nodes)

        return h_t, h_s
