import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):

    def __init__(self, c_in, c_out, num_heads=1, concat_heads=True, alpha=0.2):
        """
        Inputs:
            c_in - Dimensionality of input features
            c_out - Dimensionality of output features
            num_heads - Number of heads, i.e. attention mechanisms to apply in parallel. The
                        output features are equally split up over the heads if concat_heads=True.
            concat_heads - If True, the output of the different heads is concatenated instead of averaged.
            alpha - Negative slope of the LeakyReLU activation.
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            c_out = c_out // num_heads

        # Sub-modules and parameters needed in the layer
        self.projection = nn.Linear(c_in, c_out * num_heads)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * c_out)) # One per head
        self.leakyrelu = nn.LeakyReLU(alpha)

        # Initialization from the original implementation
        nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, node_feats, adj_matrix, print_attn_probs=False):
        """
        Inputs:
            node_feats - Input features of the node. Shape: [batch_size, c_in]
            adj_matrix - Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
            print_attn_probs - If True, the attention weights are printed during the forward pass (for debugging purposes)
        """
        dim_node_feats = node_feats.dim()
        dim = node_feats.shape
        #dim_2 = dim[1]
        if dim_node_feats == 2:
            node_feats = torch.reshape(node_feats, (1, dim[0], dim[1]))
        #else:

        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)

        # Apply linear layer and sort nodes by head
        node_feats = self.projection(node_feats)
        node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)

        # We need to calculate the attention logits for every edge in the adjacency matrix
        # Doing this on all possible combinations of nodes is very expensive
        # => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges
        edges = adj_matrix.nonzero(as_tuple=False) # Returns indices where the adjacency matrix is not 0 => edges
        #node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)
        edge_indices_row = edges[:, 0] * batch_size + edges[:, 1]
        edge_indices_col = edges[:, 0] * batch_size + edges[:, 2]

        """
        a_input = torch.cat([
            torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
            torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0)
        ],
            dim=-1)  # Index select returns a tensor with node_feats_flat being indexed at the desired positions along dim=0
        """


        #"""
        if batch_size == 1:
            node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)

            ### test_by WD
            #test_1 = torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0)
            #test_2 = torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0)

            #test_3 = torch.index_select(input=node_feats, index=edge_indices_row, dim=1)
            #test_4 = torch.index_select(input=node_feats, index=edge_indices_row, dim=1)
            ### end test by WD

            a_input = torch.cat([
                torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
                torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0)
            ], dim=-1)  # Index select returns a tensor with node_feats_flat being indexed at the desired positions along dim=0

            # Calculate attention MLP output (independent for each head)
            attn_logits = torch.einsum('bhc,hc->bh', a_input, self.a)
            attn_logits = self.leakyrelu(attn_logits)

            # Map list of attention values back into a matrix
            attn_matrix = attn_logits.new_zeros(adj_matrix.shape + (self.num_heads,)).fill_(-9e15)
            attn_matrix[adj_matrix[..., None].repeat(1, 1, 1, self.num_heads) == 1] = attn_logits.reshape(-1)

        else:
            a_input = torch.cat([
                torch.index_select(input=node_feats, index=edge_indices_row, dim=1),
                torch.index_select(input=node_feats, index=edge_indices_col, dim=1)
            ], dim=-1) # Index select returns a tensor with node_feats_flat being indexed at the desired positions along dim=0
            #a_test = edges.size(0)
            #a_input_test_flat = a_input.view(batch_size * edges.size(0), self.num_heads, -1)
            #a_input = a_input.view(batch_size * edges.size(0), self.num_heads, -1)
                #torch.reshape(a_input, (batch_size * num_nodes, self.num_heads, -1))
            attn_logits = torch.einsum('abhc,hc->abh', a_input, self.a)
            attn_logits = self.leakyrelu(attn_logits)
            shape_test = adj_matrix.shape
            #shape_test[0] = batch_size
            shape_test_1 = (batch_size, shape_test[1], shape_test[2])
            attn_matrix = attn_logits.new_zeros(shape_test_1  + (self.num_heads,)).fill_(-9e15)
            adj_matrix = torch.cat((batch_size*[adj_matrix]), dim=0)
            #attn_matrix_test = attn_logits_test.new_zeros(adj_matrix.shape + (self.num_heads,)).fill_(-9e15)
            attn_matrix[adj_matrix[..., None].repeat(1, 1, 1, self.num_heads) == 1] = attn_logits.reshape(-1)

            #attn_matrix_test[adj_matrix[..., None].repeat(1, 1, 1, self.num_heads) == 1] = attn_logits_test.reshape(-1)

        #"""

        # Calculate attention MLP output (independent for each head)
        # attn_logits = torch.einsum('bhc,hc->bh', a_input, self.a)
        # attn_logits = self.leakyrelu(attn_logits)

        ### test by WD
        #a_input_test_flat = a_input_test.view(batch_size * num_nodes, self.num_heads, -1)
        #attn_logits_test = torch.einsum('bhc,hc->bh', a_input_test_flat, self.a)
        #attn_logits_test = self.leakyrelu(attn_logits_test)
        ### end test by WD



        # Map list of attention values back into a matrix
        # attn_matrix = attn_logits.new_zeros(adj_matrix.shape+(self.num_heads,)).fill_(-9e15)
        # attn_matrix[adj_matrix[...,None].repeat(1,1,1,self.num_heads) == 1] = attn_logits.reshape(-1)


        # Weighted average of attention
        attn_probs = F.softmax(attn_matrix, dim=2)
        if print_attn_probs:
            print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))

        node_feats = torch.einsum('bijh,bjhc->bihc', attn_probs, node_feats)

        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)

        return node_feats