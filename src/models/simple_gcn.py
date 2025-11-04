import torch
import torch.nn as nn
from torch_geometric.data import Data

from modules.blocks import GCNConvLayer, NodeEmbedding


class SimpleGCN(nn.Module):
    """Simple Graph Convolutional Network for particle mobility prediction.

    Architecture:
    - Node embedding (one-hot encoding)
    - 2 GCN layers with ReLU activation
    - Final linear layer for regression

    Args:
        num_node_types: int, number of unique node types
        hidden_dim: int = 64, hidden dimension size
        output_dim: int = 10, number of output targets
        dropout_p: float = 0.1, dropout rate
    """

    def __init__(
        self,
        num_node_types: int,
        hidden_dim: int = 128,
        output_dim: int = 10,
        dropout_p: float = 0.4,
    ):
        super().__init__()

        self.num_node_types = num_node_types
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Node embedding layer
        self.node_embedding = NodeEmbedding()

        # GCN layers
        self.gcn1 = GCNConvLayer(
            in_channels=num_node_types,
            out_channels=hidden_dim,
            dropout_p=dropout_p,
            nonlinearity="relu",
        )

        self.gcn2 = GCNConvLayer(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            dropout_p=dropout_p,
            nonlinearity="relu",
        )

        # Final prediction layer
        self.predictor = nn.Linear(hidden_dim, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            data: PyTorch Geometric Data object with:
                - x: node features [n_nodes, 1]
                - edge_index: edge connectivity [2, n_edges]

        Returns:
            predictions: [n_nodes, output_dim]
        """
        # Get node features and edge index
        x = data.x  # [n_nodes, 1]
        edge_index = data.edge_index_th  # Use thermal edge index

        # Embed node features
        x = self.node_embedding(x).to(torch.float32)  # [n_nodes, num_node_types]

        # Pass through GCN layers
        x = self.gcn1(x, edge_index)  # [n_nodes, hidden_dim]
        x = self.gcn2(x, edge_index)  # [n_nodes, hidden_dim]

        # Final prediction
        predictions = self.predictor(x)  # [n_nodes, output_dim]

        return predictions

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_simple_gcn(num_node_types: int = 2, **kwargs) -> SimpleGCN:
    """Factory function to create a SimpleGCN model.

    Args:
        num_node_types: int, number of unique node types in the data
        **kwargs: additional arguments passed to SimpleGCN

    Returns:
        SimpleGCN model instance
    """
    return SimpleGCN(num_node_types=num_node_types, **kwargs)
