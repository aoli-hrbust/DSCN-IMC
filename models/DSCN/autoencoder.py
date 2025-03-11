# MIT License

# Copyright (c) 2025 Ao Li

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import List

from utils.torch_utils import *
from .backbone import (
    GCN_Encoder_SDIMC,
    NeuralMapper
)
from .ptsne_training import (
    get_q_joint,
    loss_function,
)


class ManifoldRegLoss(nn.Module):
    """
    t-SNE based manifold regularization loss.
    """

    def forward(self, inputs: dict):
        P_view: List[Tensor] = inputs["P_view"]
        H_common: Tensor = inputs["H_common"]
        M: Tensor = inputs["M"]
        viewNum: int = inputs["viewNum"]
        loss = 0

        for v in range(viewNum):
            h_common = H_common[M[:, v]]
            q_common = get_q_joint(h_common)
            loss += loss_function(p_joint=P_view[v], q_joint=q_common)
        loss = loss / viewNum
        return loss


class MultiviewEncoder(nn.Module):
    """
    The multi-view encoder part of the model.
    """

    def __init__(self, hidden_dims: int,
                 in_channels: List[int],
                 use_gcn: bool = True,
                 ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.use_gcn = use_gcn
        self.encoder_view = nn.ModuleList()
        self.viewNum = len(in_channels)
        for in_channel in in_channels:
            encoder = GCN_Encoder_SDIMC(
                view_dim=in_channel, clusterNum=self.hidden_dims
            ) if use_gcn else NeuralMapper(dim_input=in_channel, dim_emb=self.hidden_dims)

            self.encoder_view.append(encoder)

    def forward(self, inputs: dict):
        X_view: List[Tensor] = inputs["X_view"]
        M: Tensor = inputs["M"]
        if self.use_gcn:
            S_view: List[Tensor] = inputs["S_view"]

        # Encoding
        sampleNum, viewNum = M.shape
        H_view = [None] * viewNum
        for v in range(viewNum):
            if self.use_gcn:
                h_tilde = self.encoder_view[v](X_view[v], S_view[v])
            else:
                h_tilde = self.encoder_view[v](X_view[v])
            H_view[v] = h_tilde

        # Fusion
        H_common = torch.zeros(sampleNum, self.hidden_dims).to(M.device)
        for v in range(viewNum):
            H_common[M[:, v]] += H_view[v]
        H_common = H_common / torch.sum(M, 1, keepdim=True)
        H_common = F.normalize(H_common)

        inputs["H_common"] = H_common
        inputs["H_view"] = H_view
        return inputs





