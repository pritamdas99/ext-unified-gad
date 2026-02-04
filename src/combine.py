from encoder import *
from transformer import *

class GCNTemporalFusion(nn.Module):
    def __init__(self, in_dim, hid_dim=64, out_dim=128, n_layers_gcn=2, activation='relu', norm='batch',
                 n_heads=4, n_layers_attention=2, ff_dim=256, dropout=0.1):
        super().__init__()
        self.gcn = GCN(in_dim, hid_dim, out_dim, n_layers_gcn,
                       dropout, activation=activation, residual=True, norm=norm)
        self.temporal = TemporalTransformer(d_model=out_dim, n_heads=n_heads,
                                           n_layers=n_layers_attention, dim_feedforward=ff_dim,
                                           dropout=dropout)

    def forward(self, graph_seq, mrq_graph=None):
        """
        graph_seq: list of DGL graphs [G0, G1, ..., GT-1]
        Returns:
            fused_node_features: list of (N_t, D) node features per timestamp
        """
        H_nodes = []      # node embeddings per timestamp
        pooled_nodes = [] # pooled embeddings for transformer

        # ---- 1. GCN per timestamp ----
        for g in graph_seq:
            h_t = self.gcn(g)              # (N_t, D)
            H_nodes.append(h_t)
            pooled_nodes.append(h_t.mean(dim=0))  # (D, ) mean-pool per timestamp

        pooled_nodes = torch.stack(pooled_nodes).unsqueeze(0)  # (B=1, T, D)
        # ---- 2. Temporal transformer ----
        C = self.temporal(pooled_nodes).squeeze(0)  # (T, D)

        # ---- 3. Fuse temporal context back to nodes ----
        fused_node_features = []
        for t, h_t in enumerate(H_nodes):
            # broadcast transformer context to all nodes in timestamp
            h_fused = h_t + C[t]  # (N_t, D)
            fused_node_features.append(h_fused)

        return fused_node_features  # list of (N_t, D)