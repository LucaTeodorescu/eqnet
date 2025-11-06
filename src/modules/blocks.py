import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import (
    TensorProduct,
)
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from .radial import (
    BesselBasis,
    ChebychevBasis,
    GaussianBasis,
    PolynomialCutoff,
)


class RadialEmbeddingBlock(nn.Module):
    """Embed distances among nodes in a radial basis

    Args:
        r_max: float,
        num_bessel: int, [n_edges, n_basis]
        num_polynomial_cutoff: int = 6, [n_edges, 1]
        radial_type: str = "bessel", ["bessel", "gaussian", "chebyshev"]
        apply_cutoff: bool = True, [n_edges, 1]
    Returns:
        radial: torch.Tensor, [n_edges, n_basis] * cutoff, [n_edges, 1]
    """

    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int = 6,
        radial_type: str = "bessel",
        apply_cutoff: bool = True,
    ):
        super().__init__()
        if radial_type == "bessel":
            self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        elif radial_type == "gaussian":
            self.bessel_fn = GaussianBasis(r_max=r_max, num_basis=num_bessel)
        elif radial_type == "chebyshev":
            self.bessel_fn = ChebychevBasis(r_max=r_max, num_basis=num_bessel)
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel
        self.apply_cutoff = apply_cutoff

    def forward(self, edge_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # [n_edges, n_basis], [n_edges, 1]
        """Forward pass"""
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        radial = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        if hasattr(self, "apply_cutoff"):
            if not self.apply_cutoff:
                return radial, cutoff
        return radial * cutoff, None


class NodeEmbedding(nn.Module):
    """Embeds node features into a one-hot tensor:
    {0,1,2,...} -> {(1,0,0,...),(0,1,0,...),(0,0,1,...),...}

    Args:
        node_feat: torch.Tensor, [n_nodes, n_features], node features
        n_types: int, number of unique node features
    """

    def __init__(self):
        super().__init__()
        self._n_types = None

    def forward(self, node_feat: torch.Tensor) -> torch.Tensor:
        if self._n_types is None:
            self._n_types = int(node_feat.max().item() + 1)
        node_feat_one_hot = F.one_hot(node_feat[:, 0].to(torch.int64), self._n_types)
        return node_feat_one_hot.float()  # Convert to float for neural network layers

class ShEmbeddingBlock(nn.Module):
    """Embeds Distances into Spherical Harmonics basis

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self, irreps_sh):
        super().__init__()
        self.irreps_sh = irreps_sh

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        edge_attr = o3.spherical_harmonics(self.irreps_sh, edge_attr, normalize=True, normalization='component')
        return edge_attr

class GCNConvLayer(MessagePassing):
    """Graph Convolutional layer without equivariance, Kipf et Welling, 2017

    Args:
        in_channels: int, number of input channels
        out_channels: int, number of output channels
        dropout_p: float = 0.0, dropout rate
        aggr: str = "add", aggregation function
        nonlinearity: str = "relu", nonlinearity function,
        Returns:
        x: torch.Tensor, [n_nodes, out_channels]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_p: float = 0.0,
        aggr: str = "add",
        nonlinearity: str = "relu",
    ):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nonlinearity_type = nonlinearity
        if self.nonlinearity_type == "relu":
            self.nonlin = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.lin = nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,  # [n_nodes, in_channels]
        edge_index: torch.Tensor,  # [2, n_edges]
        edge_attr: torch.Tensor,  # [n_edges, n_features]
    ) -> torch.Tensor:  # [n_nodes, out_channels]
        """Forward pass"""

        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, fill_value=0, num_nodes=x.size(0))
        x = self.lin(x)
        x = self.dropout(x)
        if self.nonlinearity_type is not None:
            x = self.nonlin(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = MessagePassing.propagate(self, edge_index, x=x, norm=norm, edge_attr=edge_attr)

        return out

    def message(
        self,
        x_j: torch.Tensor,  # [n_edges, in_channels]
        norm: torch.Tensor,  # [n_edges, 1]
        edge_attr: torch.Tensor,  # [n_edges, n_features]
    ) -> torch.Tensor:  # [n_edges, out_channels]
        """Create messages"""
        w = self.edge_weight_mlp(edge_attr).view(-1, 1)
        return x_j * norm.view(-1, 1) * w


class EdgeWeightMLP(nn.Module):
    """Edge Weight MLP with dropout and BN"""

    def __init__(
        self,
        layers_size_list: list[int],
        dropout_p: float = 0.0,
    ):
        super().__init__()
        layers = []
        for i, (s1, s2) in enumerate(zip(layers_size_list, layers_size_list[1:], strict=False)):
            if i != len(layers_size_list) - 2:
                layers.append(nn.Linear(s1, s2))
                layers.append(nn.BatchNorm1d(s2, momentum=0.5))
                layers.append(nn.Dropout(dropout_p))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(s1, s2))

            self.net = nn.Sequential(*layers)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.network(edge_attr)
        return x


class EquiConvLayer(MessagePassing):
    """Equivariant Graph Convolutional layer"""

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_sh: o3.Irreps,
        irreps_out: o3.Irreps,
        num_rad_basis: int,
        MLPsize: list[int] = None,
        dropout_p: float = 0.0,
    ):
        if MLPsize is None:
            MLPsize = [16]
        super().__init__(aggr="add")
        self.irreps_in = irreps_in
        self.irreps_sh = irreps_sh
        self.irreps_out = irreps_out
        self.dropout_p = dropout_p

        self.prepareTensorProduct(self.irreps_in, self.irreps_sh, self.irreps_out)
        self.lin = o3.Linear(
            irreps_in=self.irreps_product.simplify(),
            irreps_out=self.irreps_out,
            internal_weights=True,
            shared_weights=True,
        )
        self.list_size_layers = [self.num_rad_basis] + MLPsize + [self.irreps_product.weight_numel]
        self.edge_mlp = EdgeWeightMLP(self.list_size_layers, dropout_p)

    def prepareTensorProduct(
        self,
        in1_irreps: o3.Irreps,
        in2_irreps: o3.Irreps,
        out_irreps: o3.Irreps,
    ) -> TensorProduct:
        """Prepare TensorProduct (no channel mixing: 'uvu')."""

        irreps_product_list = []
        instructions = []

        allowed = {ir for _, ir in out_irreps}

        for i, (mul, ir_in1) in enumerate(in1_irreps):
            for j, (_, ir_in2) in enumerate(in2_irreps):
                for ir_out in ir_in1 * ir_in2:
                    if ir_out in allowed:
                        k = len(irreps_product_list)
                        irreps_product_list.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))

        irreps_product = o3.Irreps(irreps_product_list)
        irreps_product, permutation, _ = irreps_product.sort()
        instructions = [(i1, i2, permutation[i_out], mode, train) for (i1, i2, i_out, mode, train) in instructions]

        self.tensor_product = TensorProduct(
            in1_irreps,
            in2_irreps,
            irreps_product,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )
        self.irreps_product = irreps_product
        return self.tensor_product

    def forward(
        self,
        x: torch.Tensor,
        edge_index,
        edge_attr,
        edge_features,
        batch,
    ):
        """Forward pass"""

        x = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_features=edge_features)
        return x

    def message(
        self,
        x_j,
        edge_attr,
        edge_features,
    ):
        """Create messages"""
        weights = self.edge_mlp(edge_features)
        message = self.tensor_product(x_j, edge_attr, weights)
        return message

    def update(self, message, x):
        """Update node features"""
        x = self.lin(message)
        return x
