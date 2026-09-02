"""Regression tests for actorder GPTQLinear checkpoint state."""

import torch
from torch import nn

from onecomp.quantizer.gptq.gptq_layer import GPTQLinear


def _make_actorder_layer(in_features, out_features, groupsize, perm):
    num_groups = in_features // groupsize
    return GPTQLinear(
        in_features=in_features,
        out_features=out_features,
        wbits=4,
        groupsize=groupsize,
        actorder=True,
        quantized_weight=torch.ones(out_features, in_features, dtype=torch.int32),
        scale=torch.ones(num_groups, out_features),
        zero=torch.ones(num_groups, out_features),
        perm=perm,
        bias=torch.zeros(out_features),
        device="cpu",
        pack_weights=False,
        use_gemlite=False,
    )


def test_actorder_network_state_dict_keeps_g_idx_but_not_perm():
    first_perm = torch.tensor([2, 0, 3, 1, 6, 4, 7, 5])
    second_perm = torch.tensor([1, 3, 0, 2])
    network = nn.Sequential(
        _make_actorder_layer(8, 4, 2, first_perm),
        nn.ReLU(),
        _make_actorder_layer(4, 3, 2, second_perm),
    )

    output = network(torch.randn(2, 8))

    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()
    for layer, perm in zip((network[0], network[2]), (first_perm, second_perm)):
        state_dict = layer.state_dict()
        assert "perm" not in state_dict
        assert "g_idx" in state_dict
        expected_g_idx = torch.argsort(perm) // layer.groupsize
        assert torch.equal(state_dict["g_idx"], expected_g_idx)
