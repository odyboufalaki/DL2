import torch
import torch.nn as nn
from rff.layers import GaussianEncoding

from nn.probe_features import GraphProbeFeatures


def batch_to_graphs(
    weights,
    biases,
    weights_mean=None,
    weights_std=None,
    biases_mean=None,
    biases_std=None,
):
    device = weights[0].device
    bsz = weights[0].shape[0]
    num_nodes = weights[0].shape[1] + sum(w.shape[2] for w in weights)

    node_features = torch.zeros(bsz, num_nodes, biases[0].shape[-1], device=device)
    edge_features = torch.zeros(
        bsz, num_nodes, num_nodes, weights[0].shape[-1], device=device
    )

    row_offset = 0
    col_offset = weights[0].shape[1]  # no edge to input nodes
    for i, w in enumerate(weights):
        _, num_in, num_out, _ = w.shape
        w_mean = weights_mean[i] if weights_mean is not None else 0
        w_std = weights_std[i] if weights_std is not None else 1
        edge_features[
            :, row_offset : row_offset + num_in, col_offset : col_offset + num_out
        ] = (w - w_mean) / w_std
        row_offset += num_in
        col_offset += num_out

    row_offset = weights[0].shape[1]  # no bias in input nodes
    for i, b in enumerate(biases):
        _, num_out, _ = b.shape
        b_mean = biases_mean[i] if biases_mean is not None else 0
        b_std = biases_std[i] if biases_std is not None else 1
        node_features[:, row_offset : row_offset + num_out] = (b - b_mean) / b_std
        row_offset += num_out

    return node_features, edge_features


class GraphConstructor(nn.Module):
    def __init__(
        self,
        d_in,
        d_edge_in,
        d_node,
        d_edge,
        layer_layout,
        rev_edge_features=False,
        zero_out_bias=False,
        zero_out_weights=False,
        inp_factor=1,
        input_layers=1,
        sin_emb=False,
        sin_emb_dim=128,
        use_pos_embed=True,
        num_probe_features=0,
        inr_model=None,
        stats=None,
    ):
        super().__init__()
        self.rev_edge_features = rev_edge_features
        self.nodes_per_layer = layer_layout
        self.zero_out_bias = zero_out_bias
        self.zero_out_weights = zero_out_weights
        self.use_pos_embed = use_pos_embed
        self.stats = stats if stats is not None else {}
        self._d_node = d_node
        self._d_edge = d_edge

        self.pos_embed_layout = (
            [1] * layer_layout[0] + layer_layout[1:-1] + [1] * layer_layout[-1]
        )
        self.pos_embed = nn.Parameter(torch.randn(len(self.pos_embed_layout), d_node))

        if not self.zero_out_weights:
            proj_weight = []
            if sin_emb:
                proj_weight.append(
                    GaussianEncoding(
                        sigma=inp_factor,
                        input_size=d_edge_in
                        + (2 * d_edge_in if rev_edge_features else 0),
                        encoded_size=sin_emb_dim,
                    )
                )
                proj_weight.append(nn.Linear(2 * sin_emb_dim, d_edge))
            else:
                proj_weight.append(
                    nn.Linear(
                        d_edge_in + (2 * d_edge_in if rev_edge_features else 0), d_edge
                    )
                )

            for i in range(input_layers - 1):
                proj_weight.append(nn.SiLU())
                proj_weight.append(nn.Linear(d_edge, d_edge))

            self.proj_weight = nn.Sequential(*proj_weight)
        if not self.zero_out_bias:
            proj_bias = []
            if sin_emb:
                proj_bias.append(
                    GaussianEncoding(
                        sigma=inp_factor,
                        input_size=d_in,
                        encoded_size=sin_emb_dim,
                    )
                )
                proj_bias.append(nn.Linear(2 * sin_emb_dim, d_node))
            else:
                proj_bias.append(nn.Linear(d_in, d_node))

            for i in range(input_layers - 1):
                proj_bias.append(nn.SiLU())
                proj_bias.append(nn.Linear(d_node, d_node))

            self.proj_bias = nn.Sequential(*proj_bias)

        self.proj_node_in = nn.Linear(d_node, d_node)
        self.proj_edge_in = nn.Linear(d_edge, d_edge)

        if num_probe_features > 0:
            self.gpf = GraphProbeFeatures(
                d_in=layer_layout[0],
                num_inputs=num_probe_features,
                inr_model=inr_model,
                input_init=None,
                proj_dim=d_node,
            )
        else:
            self.gpf = None

    def forward(self, inputs):
        node_features, edge_features = batch_to_graphs(*inputs, **self.stats)
        mask = edge_features.sum(dim=-1, keepdim=True) != 0
        if self.rev_edge_features:
            rev_edge_features = edge_features.transpose(-2, -3)
            edge_features = torch.cat(
                [edge_features, rev_edge_features, edge_features + rev_edge_features],
                dim=-1,
            )
            mask = mask | mask.transpose(-3, -2)

        if self.zero_out_weights:
            edge_features = torch.zeros(
                (*edge_features.shape[:-1], self._d_edge),
                device=edge_features.device,
                dtype=edge_features.dtype,
            )
        else:
            edge_features = self.proj_weight(edge_features)
        if self.zero_out_bias:
            # only zero out bias, not gpf
            node_features = torch.zeros(
                (*node_features.shape[:-1], self._d_node),
                device=node_features.device,
                dtype=node_features.dtype,
            )
        else:
            node_features = self.proj_bias(node_features)

        if self.gpf is not None:
            probe_features = self.gpf(*inputs)
            node_features = node_features + probe_features

        node_features = self.proj_node_in(node_features)
        edge_features = self.proj_edge_in(edge_features)

        if self.use_pos_embed:
            pos_embed = torch.cat(
                [
                    # repeat(self.pos_embed[i], "d -> 1 n d", n=n)
                    self.pos_embed[i].unsqueeze(0).expand(1, n, -1)
                    for i, n in enumerate(self.pos_embed_layout)
                ],
                dim=1,
            )
            node_features = node_features + pos_embed
        return node_features, edge_features, mask
