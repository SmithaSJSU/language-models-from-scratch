"""Gate-by-gate, biased GRU cell using PyTorch's reset-after convention."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class GRUCellScratch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
        bound = 1 / math.sqrt(hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)

    def forward(self, x, h):
        # PyTorch packs gates in reset, update, candidate order.
        W_ir, W_iz, W_in = self.weight_ih.chunk(3, dim=0)
        W_hr, W_hz, W_hn = self.weight_hh.chunk(3, dim=0)
        b_ir, b_iz, b_in = self.bias_ih.chunk(3, dim=0)
        b_hr, b_hz, b_hn = self.bias_hh.chunk(3, dim=0)
        r = torch.sigmoid(F.linear(x, W_ir, b_ir) + F.linear(h, W_hr, b_hr))
        z = torch.sigmoid(F.linear(x, W_iz, b_iz) + F.linear(h, W_hz, b_hz))
        # Reset acts AFTER the hidden transformation, including its bias.
        n = torch.tanh(F.linear(x, W_in, b_in) + r * F.linear(h, W_hn, b_hn))
        return (1 - z) * n + z * h


def run_scratch_gru(cell, x, h0):
    """Single layer, batch-first input; h0 is [1, batch, hidden]."""
    h = h0[0]
    outputs = []
    for t in range(x.shape[1]):
        h = cell(x[:, t, :], h)
        outputs.append(h)
    return torch.stack(outputs, dim=1), h.unsqueeze(0)
