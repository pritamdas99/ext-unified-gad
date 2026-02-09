from encoder import *
from transformer import *
import numpy as np

def SubgraphPooling(h, sg):
    with sg.local_scope():
        sg.ndata['h_tmp'] = h
        sg.update_all(fn.u_mul_e('h_tmp', 'pw', 'm'), fn.sum('m', 'h_tmp'))
        h = sg.ndata['h_tmp'] + h

        return h
class GCNTemporalFusion(nn.Module):
    def __init__(self, original_graph, in_dim, hid_dim=64, out_dim=128, n_layers_gcn=2, activation='relu', norm='batch',
                 n_heads=4, n_layers_attention=2, ff_dim=256, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.original_graph = original_graph
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
        
        input = self.original_graph.ndata['feature']
            
        
        mask_t = torch.ones(len(graph_seq), total_nodes, device=device)
        for t, g in enumerate(graph_seq):
            input_t = input[g.ndata[dgl.NID]]
            h_t = self.gcn(g, input_t)
            h_t = SubgraphPooling(h_t, mrq_graph[t])
            input[g.ndata[dgl.NID]] = input_t
            H_nodes.append(input) 
            
        final_h = []
        for node_i in range(total_nodes):
            neighbours = np.array([node_i])
            i_n = np.array(self.original_graph.successsor(node_i))
            o_n = np.array(self.original_graph.predecessor(node_i))
            neighbours = np.concat(neighbours, i_n)
            neighbours = np.concat(neighbours, o_n)
            neighbours = np.unique(neighbours)
            pooled_nodes = []
            for t,g in enumerate(graph_seq):
                h_nodes_t = H_nodes[t]
                h_feat_t = h_nodes_t[neighbours.tolist()]
                pooled_nodes.append(h_feat_t)
                
            pooled_nodes = torch.stack(pooled_nodes)

            C = self.temporal(pooled_nodes, mask_t)
            idx = neighbours.tolist().index(node_i)
            for t in range(graph_seq):
                ar=C[t][idx]
                H_nodes[t][node_i] = ar
                
            for t,g in enumerate(graph_seq):
                h_sub = H_nodes[t][g.ndata[dgl.NID]]
                final_h.append(h_sub)
                
                
                
                
        return final_h           