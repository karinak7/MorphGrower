import torch
from torch_geometric.nn import MessagePassing, global_mean_pool
from utils.debug_config import DEBUG

class TGNNconv(MessagePassing):
    def __init__(self, in_channels, emd_channels, out_channels, gru):
        super().__init__(aggr='mean')
        self.linear = torch.nn.Linear(in_channels, emd_channels)
        self.gru = gru
        self.norm = torch.nn.ReLU()
        self.in_channel = in_channels
        self.emd_channel = emd_channels

    def forward(self, x, edge_index):
        """
        x: [no_nodes, 2*n_layers*hidden_dim] = [no_nodes, 256]
        edge_index: [D, no_nodes, no_nodes]
        """
        # x has shape [N, in_channels]
        # edge_index has shape [layer, 2, E[layer]]
        # encoder x
        x = self.linear(x) #[no_nodes, emd_channels]
        # propagating messages
        if DEBUG:
            print(f"[DEBUG] TGNNconv forward: edge_index.shape{edge_index.shape}")
            print(f"[DEBUG] TGNNconv forward: edge_index._indices().shape {edge_index._indices().shape}")
        #edge_index._indices(): [3, no_edges]. A set of edges: [(depth, row, col), ...]
        if edge_index._indices().shape[1]: #check if edge matrix is empty 
            for i in range(len(edge_index)): #for each depth - WHY LOOP FROM TOP OF TREE DOWN (i.e. SHALLOW TO DEEP)? 
                # get hidden messsage
                h_hat = self.propagate(edge_index=edge_index[i]._indices(), size=(x.size(0), x.size(0)), x=x)
                x = h_hat #update the node features 
        # output
        return self.norm(x)

    def message(self, x_j):
        """
        x_j = x[row]  # features from source nodes (children) [num_children, emd_channels]
        """
        if DEBUG: print(f"[DEBUG] TGNNconv message: x_j.shape {x_j.shape}")
        return x_j

    def update(self, h, x):
        """
        h: result of message aggregation [N, emd_channels]
        x: previous node features before aggregation [N, emd_channels]
        """
        #move batch size to dim=1
        h = h.reshape(1, x.shape[0], x.shape[1]) # [1, no_nodes, emd_channels]: [1, 3634, 64]
        x = x.reshape(1, x.shape[0], x.shape[1]) 
        if DEBUG:
            print(f"[DEBUG] TGNNconv update: h.shape: {h.shape}")
            print(f"[DEBUG] TGNNconv update: x.shape: {x.shape}")
        # use gru layer
        output, h_hat = self.gru(x, h) #h_hat: the updated hidden state for each node
        # for nodes without message from neighbour at current propagration -> keep original node features
        mask = (h == 0) # bool mask matrix
        #every position where h == 0, replace the value in h_hat with the corresponding value from x.
        h_hat[mask] = x[mask]
        return torch.squeeze(h_hat) # [no_nodes, emd_channels]


class TGNN(torch.nn.Module):
    def __init__(self, size, in_channel, emd_channel, out_channel):
        """
        in_channel: 256
        emd_channel: 64 -> project the input branch rep from dim 256 to dim 64
        out_channel: 64
        """
        super().__init__()
        self.conv0 = torch.nn.Conv2d(size, size, kernel_size=3, padding=0, stride=1)
        emd_channel = out_channel
        self.gru = torch.nn.GRU(emd_channel, out_channel)
        self.conv1 = TGNNconv(in_channel, emd_channel, out_channel, self.gru)
        self.conv2 = TGNNconv(out_channel, emd_channel, out_channel, self.gru)
        self.size = out_channel
        self.in_channel = in_channel
        self.emd_channel = emd_channel

    def forward(self, x, offset, edge):
        # x is [N*32*3], edge is [D*N*N], D=depth, N=no_nodes <- dim incorrect
        # x is [N, 256], edge is [D,N,N]
        if len(x.shape) == 1:
            x = x.unsqueeze(dim=0)
        if DEBUG: print(f"[DEBUG] TGNN.forward: x.shape: {x.shape}")
        e1 = self.conv1(x, edge)
        e2 = self.conv2(e1, edge)
        out = global_mean_pool(e2, offset)
        return out
