import torch
from torch.nn import Linear, LeakyReLU, Dropout, ReLU
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.weight = torch.nn.Parameter(torch.tensor([[2.0, 0.0, 0.0, 0.0],
                                                [0.0, 2.0, 0.0, 0.0],
                                                [0.0, 0.0, 2.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.5]], dtype=torch.float32))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        if isinstance(edge_index, type(None)):
            return x 
        else:
            # Step 1: Add self-loops to the adjacency matrix.
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

            # Step 2: Multiply weight matrix.
            x = torch.mm(x, self.weight)

            # Step 3: Compute normalization.
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            # We do not use (Degree matrix)^(-1/2).
            deg_inv_sqrt = deg.pow(0)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

            # Step 4-5: Start propagating messages.
            return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

class SCModel(torch.nn.Module):
    def __init__(self, dim_hidden, num_a_classes):
        super().__init__()
        self.dropout = Dropout(0.50)
        self.linear1 = Linear(768, dim_hidden)
        self.leakyrelu = LeakyReLU()
        self.linear2 = Linear(dim_hidden, num_a_classes)
    def forward(self, x):
        dropout_output = self.dropout(x)
        linear1_output = self.linear1(dropout_output)
        leakyrelu_output = self.leakyrelu(linear1_output)
        linear2_output = self.linear2(leakyrelu_output)
        return linear2_output, leakyrelu_output

class RVModel(torch.nn.Module):
    def __init__(self, dim_hidden1, num_a_classes, dim_hidden2, num_b_classes):
        super().__init__()
        self.conv = GCNConv()
        self.linear3 = Linear(dim_hidden1 + num_a_classes, dim_hidden2)
        self.relu = ReLU()
        self.linear4 = Linear(dim_hidden2, num_b_classes)
        self.mul_matrix_1 = torch.nn.Parameter(torch.tensor([[2.0, 0.0, 0.0, 0.0],
                                                [0.0, 2.0, 0.0, 0.0],
                                                [0.0, 0.0, 2.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.5]], dtype=torch.float32))
        self.bias_matrix_2 = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 0.0, 2.5]], dtype=torch.float32))
    def forward(self, edge_index, batch_idx, stance_features, relu_output):
        
        conv_output = self.conv(torch.mm(stance_features, self.mul_matrix_1), edge_index)
        bias_output = torch.sub(conv_output, self.bias_matrix_2)
        mean_pool_output = global_mean_pool(torch.cat((bias_output, relu_output), dim=1), batch_idx)
        linear3_output = self.linear3(mean_pool_output)
        linear4_output = self.linear4(self.relu(linear3_output))
        return linear4_output
    
class GCN(torch.nn.Module):
    def __init__(self, dim_hidden1, num_a_classes, dim_hidden2, num_b_classes):
        super().__init__()
        self.sc = SCModel(dim_hidden1, num_a_classes)
        self.rv = RVModel(dim_hidden1, num_a_classes, dim_hidden2, num_b_classes)
    def forward(self, x, edge_index, batch_idx):
        linear2_output, relu_output = self.sc(x)
        linear4_output = self.rv(edge_index, batch_idx, linear2_output, relu_output)
        return linear2_output, linear4_output
