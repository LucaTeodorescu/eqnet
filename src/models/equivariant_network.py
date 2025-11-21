import e3nn.nn as enn
import e3nn.o3 as o3
import torch
import torch.nn as nn

from ..modules.blocks import (
    EquiConvLayer,
    NodeEmbedding,
    RadialEmbeddingBlock,
    ShEmbeddingBlock,
)


class EquivariantNetwork(nn.Module):
    """Equivariant Network using SE(3) equivariant convolution

    Args:

    """

    def __init__(
        self,
        irreps_in,
        irreps_sh,
        irreps_hiddens,
        radial_basis_start,
        radial_basis_type,
        num_radial_basis,
        edge_cutoff,
        MLPsize,
        num_particle_types,
        dropout_p: float = 0.0,
        batchnorm: bool = True,
        skip_conn: bool = True,
        include_potential: bool = False,
        include_cage_size: bool = False,
        output_list: list[int] = None,
        bn_momentum: float = 0.5,
    ):
        super().__init__()
        # Architecture configuration
        self.irreps_in = irreps_in
        self.irreps_sh = irreps_sh
        self.irreps_hiddens = irreps_hiddens
        self.num_particle_types = num_particle_types
        self.num_layers = len(irreps_hiddens) - 1

        # Training configuration
        self.dropout_p = dropout_p
        self.batchnorm = batchnorm
        self.skip_conn = skip_conn

        # Task configuration
        self.include_cage_size = include_cage_size
        self.include_potential = include_potential
        if output_list is None:
            output_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.output_list = output_list
        self.ntargets = len(output_list)
        
        if self.skip_conn:
            self.irreps_hiddens_sc = [irreps_in + irr for irr in irreps_hiddens]
        else:
            self.irreps_hiddens_sc = irreps_hiddens
        self.nb_hidden_scalars = [o3.Irreps(str(irr[0])).num_irreps for irr in irreps_hiddens]

        # Embedding layers
        self.radial_embedding_layer = RadialEmbeddingBlock(r_max=edge_cutoff, num_bessel=num_radial_basis, radial_type=radial_basis_type)
        self.sh_embedding_layer = ShEmbeddingBlock(irreps_sh)
        self.node_embedding_layer = NodeEmbedding()

        # First convolution layer
        self.layer_0 = EquiConvLayer(
            irreps_in=irreps_in,
            irreps_sh=irreps_sh,
            irreps_out=self.irreps_hiddens[0],
            num_rad_basis=num_radial_basis,
            MLPsize=MLPsize,
            dropout_p=dropout_p,
        )
        if self.batchnorm:
            self.norm_0 = enn.BatchNorm(self.irreps_hiddens[0], momentum=bn_momentum)

        # Convolution layers
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.conv_layers.append(
                EquiConvLayer(
                    irreps_in=self.irreps_hiddens_sc[i],
                    irreps_sh=irreps_sh,
                    irreps_out=self.irreps_hiddens[i+1],
                    num_rad_basis=num_radial_basis,
                    MLPsize=MLPsize,
                    dropout_p=dropout_p,
                )
            )
            if self.batchnorm:
                self.norm_layers.append(enn.BatchNorm(self.irreps_hiddens[i+1], momentum=bn_momentum))

        self.setup_normReadout(self.irreps_hiddens[-1])

        # Output layer: independent readout layer for each particle type
        self.readout_per_type = nn.ModuleList([nn.Linear(self.irreps_hiddens[-1].num_irreps, self.ntargets) 
                                               for _ in range(num_particle_types)])

    def setup_normReadout(self, irreps_hiddens):
        """Setup norm layer for readout"""
        irreps_nonscalar = o3.Irreps(str(irreps_hiddens[1:]))
        self.outNorm = o3.Norm(irreps_nonscalar, squared=False)

    def forward(self, graph):
        """Forward pass"""
        x, edge_index, edge_attr, e_pot, delta_r = (
            graph.x,
            graph.edge_index,
            graph.edge_attr.float(),
            graph.e_pot.float(),
            graph.delta_r.float(),
        )
        #TODO: embedding or encoding? (name)
        edge_lengths = edge_attr.norm(dim=1, keepdim=True)
        edge_attr = self.sh_embedding_layer(edge_attr)
        edge_features, _ = self.radial_embedding_layer(edge_lengths)
        node_input = self.node_embedding_layer(x)

        if self.include_potential:
            node_input = torch.cat((node_input, e_pot), dim=-1)
        if self.include_cage_size:
            node_input = torch.cat((node_input, delta_r), dim=-1)

        if self.skip_conn:
            node_copy = node_input.detach().clone()

        x = self.layer_0(node_input, edge_index, edge_attr, edge_features)
        if self.batchnorm:
            x = self.norm_0(x)

        for layer_n in range(self.num_layers):
            node_features_in = torch.cat((node_copy, x), dim=-1) if self.skip_conn else x
            node_update = self.conv_layers[layer_n](node_features_in, edge_index, edge_attr, edge_features)

            if self.batchnorm:
                node_update = self.norm_layers[layer_n](node_update)

            # Residual connection
            if self.irreps_hiddens[layer_n + 1] == self.irreps_hiddens[layer_n]:
                x += node_update
            else:
                x = node_update

        # Output layer
        # Assumes that the irreps are ordered as [scalar, nonscalar, nonscalar, ...]
        nonscalar_norms = self.outNorm(x[:, self.nb_hidden_scalars[-1] :])
        x = torch.cat((x[:, : self.nb_hidden_scalars[-1]], 
                        nonscalar_norms), 
                        dim=-1) # concatenate scalar and nonscalar features

        # Readout layer - extract particle types from one-hot encoded node features
        particle_types = graph.x[:, 0].long()
        output = torch.zeros(x.shape[0], self.ntargets, device=x.device)
        for i in range(self.num_particle_types):
            mask = particle_types == i
            output[mask, :] = self.readout_per_type[i](x[mask, :])
        return output
