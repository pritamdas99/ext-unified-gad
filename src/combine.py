from encoder import *
from transformer import *

def SubgraphPooling(h, sg):
    with sg.local_scope():
        sg.ndata['h_tmp'] = h
        sg.update_all(fn.u_mul_e('h_tmp', 'pw', 'm'), fn.sum('m', 'h_tmp'))
        h = sg.ndata['h_tmp'] + h

        return h
class GCNTemporalFusion(nn.Module):
    def __init__(self, in_dim, hid_dim=64, out_dim=128, n_layers_gcn=2, activation='relu', norm='batch',
                 n_heads=4, n_layers_attention=2, ff_dim=256, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.gcn = GCN(in_dim, hid_dim, out_dim, n_layers_gcn,
                       dropout, activation=activation, residual=True, norm=norm)
        self.temporal = Transformer(d_model=out_dim, n_heads=n_heads,
                                           n_layers=n_layers_attention, dim_feedforward=ff_dim,
                                           dropout=dropout)

    def forward(self, total_nodes, graph_seq, mrq_graph=None):
        """
        graph_seq: list of DGL graphs [G0, G1, ..., GT-1]
        Returns:
            temporal_node_features: list of (T, N_t, D) node features per timestamp
        """
        H_nodes = []      # node embeddings per timestamp
        pooled_nodes = [] # pooled embeddings for transformer

        device = graph_seq[0].device
        mask_t = torch.ones(len(graph_seq), total_nodes, device=device)
        for t, g in enumerate(graph_seq):
            h_t = self.gcn(g, g.ndata['feature'])
            h_t = SubgraphPooling(h_t, mrq_graph[t])
            padded_ht = torch.zeros(total_nodes, self.in_dim, device=device)
            padded_ht[g.ndata[dgl.NID]] = h_t
            H_nodes.append(h_t)
            pooled_nodes.append(padded_ht) 
            mask_t[t, g.ndata[dgl.NID]] = 0  
       


        pooled_nodes = torch.stack(pooled_nodes)

        C = self.temporal(pooled_nodes, src_key_padding_mask=mask_t)

        h_sub_t = [C[t][g.ndata[dgl.NID]] for t, g in enumerate(graph_seq)] 
        return h_sub_t  