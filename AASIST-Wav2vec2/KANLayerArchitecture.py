class KAN_GAL(nn.Module):

    def __init__(self, in_dim, out_dim, num_nodes, temperature=1.0, dropout=0.2):
        super(KAN_GAL, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes
        self.temprature = temperature

        self.dropout = nn.Dropout(dropout)
        self.kan_attention = KANLayer(in_dim, in_dim)
        self.W_att = nn.Parameter(torch.empty(in_dim, num_nodes))
        nn.init.xavier_uniform_(self.W_att)
        self.kan_attn_proj = KANLayer(in_dim, out_dim)
        self.kan_direct_proj = KANLayer(in_dim, out_dim)
        self.batch_norm = nn.BatchNorm1d(num_nodes)
    
    def forward(self, h):
        batch_size = h.size(0)
        h = self.dropout(h)
        h_expanded_i = h.unsqueeze(2)
        h_expanded_j = h.unsqueeze(1)

        node_products = h_expanded_i * h_expanded_j
        original_shape = node_products.shape
        node_products_flat = node_products.view(-1, self.in_dim)
        kan_out = self.kan_attention(node_products_flat)
        kan_out = kan_out.view(batch_size, self.num_nodes, self.num_nodes, self.in_dim)
        kan_out = torch.tanh(kan_out)
        attention_scores = torch.einsum('bnmd,dk->bnmk', kan_out, self.W_att)
        attention_scores = attention_scores.mean(dim=-1)
        attention_map = F.softmax(attention_scores / self.temprature, dim=-1)
        h_attended = torch.bmm(attention_map, h)
        h_attended_flat = h_attended.view(-1, self.in_dim)
        kan2_out = self.kan_attn_proj(h_attended_flat)
        kan2_out = kan2_out.view(batch_size, self.num_nodes, self.out_dim)

        h_flat = h.view(-1, self.in_dim)
        kan3_out = self.kan_direct_proj(h_flat)
        kan3_out = kan3_out.view(batch_size, self.num_nodes, self.out_dim)

        output = kan2_out + kan3_out
        output = output.transpose(1,2)
        output = self.batch_norm(output)
        output = output.transpose(1,2)
        return output

class KAN_GraphPool(nn.Module):

    def __init__(self, in_dim, ratio=0.5, dropout=0.2):
        super(KAN_GraphPool, self).__init__()
        self.in_dim = in_dim
        self.ratio = ratio
        self.dropout = nn.Dropout(dropout)
        self.kan_score = KANLayer(in_dim, 1)

    def forward(self, k=None):
        batch_size, num_nodes, in_dim = h.shape
        if k is None:
            k = max(1, int(num_nodes * self.ratio))
        h_drop = self.dropout(h)
        h_flat = h_drop.view(-1, in_dim)
        scores_flat = self.kan_score(h_flat)
        scores = scores_flat.view(batch_size, num_nodes)
        scores = torch.sigmoid(scores)
        h_gated = h * scores.unsqueeze(-1)
        tok_k_scores, top_k_indices = torch.topk(scores, k, dim=1)
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, in_dim)
        h_pooled = torch.gather(h_gated, 1, top_k_indices_expanded)
        return h_pooled

class KAN_HS_GAL(nn.Module):
    def __init__(self, temporal_dim, spatial_dim, num_nodes, stack_dim, temprature=1.0, dropout=0.2):
        super(KAN_HS_GAL, self).__init__()

        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self.num_nodes = num_nodes
        self.stack_dim = stack_dim
        self.hetero_dim = temporal_dim + spatial_dim
        self.temprature = temprature

        self.dropout = nn.Dropout(dropout)
        
        self.kan_temporal_proj = KANLayer(temporal_dim, temporal_dim)
        self.kan_spatial_proj = KANLayer(spatial_dim, spatial_dim)
        
        self.kan_primary_attn = KANLayer(self.hetero_dim, self.hetero_dim)
        
        self.W11 = nn.Parameter(torch.randn(self.hetero_dim))
        self.W12 = nn.Parameter(torch.randn(self.hetero_dim))
        self.W22 = nn.Parameter(torch.randn(self.hetero_dim))
        
        self.kan_stack_attn = KANLayer(self.hetero_dim, self.hetero_dim)
        
        self.W_m = nn.Parameter(torch.randn(self.hetero_dim, 1))
        nn.init.xavier_uniform_(self.W_m)
        
        self.kan_stack_update1 = KANLayer(self.hetero_dim, stack_dim)
        self.kan_stack_update2 = KANLayer(stack_dim, stack_dim)
        
        self.kan_hetero_update1 = KANLayer(self.hetero_dim, self.hetero_dim)
        self.kan_hetero_update2 = KANLayer(self.hetero_dim, self.hetero_dim)
        
        self.batch_norm = nn.BatchNorm1d(num_nodes)
    
    def forward(self, h_t, h_s, S):

        batch_size = h_t.size(0)
        
        h_t_proj = self.kan_temporal_proj(h_t.view(-1, self.temporal_dim))
        h_t_proj = h_t_proj.view(batch_size, self.num_nodes, self.temporal_dim)
        
        h_s_proj = self.kan_spatial_proj(h_s.view(-1, self.spatial_dim))
        h_s_proj = h_s_proj.view(batch_size, self.num_nodes, self.spatial_dim)
        
        h_st = torch.cat([h_t_proj, h_s_proj], dim=-1)
        
        h_st = self.dropout(h_st)
        
        h_st_i = h_st.unsqueeze(2)
        h_st_j = h_st.unsqueeze(1)
        node_products = h_st_i * h_st_j
        
        node_products_flat = node_products.view(-1, self.hetero_dim)
        primary_attn_flat = self.kan_primary_attn(node_products_flat)
        primary_attn = primary_attn_flat.view(batch_size, self.num_nodes, self.num_nodes, self.hetero_dim)
        A = torch.tanh(primary_attn)
        
        B = torch.zeros(batch_size, self.num_nodes, self.num_nodes, device=h_st.device)
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i < self.temporal_dim and j < self.temporal_dim:
                    weight = self.W11
                elif i >= self.temporal_dim and j >= self.temporal_dim:
                    weight = self.W22
                else:
                    weight = self.W12
                
                B[:, i, j] = (A[:, i, j, :] * weight).sum(dim=-1)
        
        B_hat = F.softmax(B / self.temperature, dim=-1)
        
        S_expanded = S.unsqueeze(1).expand(-1, self.num_nodes, -1)
        S_padded = F.pad(S_expanded, (0, self.hetero_dim - self.stack_dim))
        
        h_st_stack = h_st * S_padded
        
        h_st_stack_flat = h_st_stack.view(-1, self.hetero_dim)
        stack_attn_flat = self.kan_stack_attn(h_st_stack_flat)
        stack_attn = stack_attn_flat.view(batch_size, self.num_nodes, self.hetero_dim)
        stack_attn = torch.tanh(stack_attn)
        
        stack_scores = torch.matmul(stack_attn, self.W_m).squeeze(-1)
        A_m = F.softmax(stack_scores / self.temperature, dim=-1)
        
        h_st_weighted = (h_st * A_m.unsqueeze(-1)).sum(dim=1)
        
        S_update1 = self.kan_stack_update1(h_st_weighted)
        S_update2 = self.kan_stack_update2(S)
        S_new = S_update1 + S_update2
        
        h_st_attended = torch.bmm(B_hat, h_st)
        
        h_st_update1_flat = h_st_attended.view(-1, self.hetero_dim)
        h_st_update1 = self.kan_hetero_update1(h_st_update1_flat)
        h_st_update1 = h_st_update1.view(batch_size, self.num_nodes, self.hetero_dim)
        
        h_st_update2_flat = h_st.view(-1, self.hetero_dim)
        h_st_update2 = self.kan_hetero_update2(h_st_update2_flat)
        h_st_update2 = h_st_update2.view(batch_size, self.num_nodes, self.hetero_dim)
        
        h_st_new = h_st_update1 + h_st_update2
        
        h_st_new = h_st_new.transpose(1, 2)
        h_st_new = self.batch_norm(h_st_new)
        h_st_new = h_st_new.transpose(1, 2)
        
        I_t = torch.eye(self.num_nodes, device=h_st.device)[:, :self.temporal_dim]
        zeros_s = torch.zeros(self.num_nodes, self.spatial_dim, device=h_st.device)
        M_t = torch.cat([I_t, zeros_s], dim=1)
        
        zeros_t = torch.zeros(self.num_nodes, self.temporal_dim, device=h_st.device)
        I_s = torch.eye(self.num_nodes, device=h_st.device)[:, :self.spatial_dim]
        M_s = torch.cat([zeros_t, I_s], dim=1)
        
        h_t_new = torch.matmul(h_st_new, M_t.t())
        h_s_new = torch.matmul(h_st_new, M_s.t())
        
        return h_t_new, h_s_new, S_new
