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


from torch_geometric.nn.dense import DenseGCNConv

from utils.torch_utils import F, Tensor, nn, torch


class NeuralMapper(nn.Module):


    def __init__(self, dim_input, dim_emb=2):
        super().__init__()
        # 这个4层MLP的结构和CDIMC的是一样的，不过多了BN。
        dim1 = int(round(dim_input * 0.8))
        dim2 = int(round(dim_input * 0.5))
        self.linear_1 = nn.Linear(dim_input, dim1)
        self.bn_1 = nn.BatchNorm1d(dim1)
        self.linear_2 = nn.Linear(dim1, dim1)
        self.bn_2 = nn.BatchNorm1d(dim1)
        self.linear_3 = nn.Linear(dim1, dim2)
        self.bn_3 = nn.BatchNorm1d(dim2)
        self.linear_4 = nn.Linear(dim2, dim_emb)
        self.relu = nn.ReLU()
        self.apply(weights_init)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.bn_1(x)
        x = self.linear_2(self.relu(x))
        x = self.bn_2(x)
        x = self.linear_3(self.relu(x))
        x = self.bn_3(x)
        x = self.linear_4(self.relu(x))
        return x

class GCN_Encoder_SDIMC(nn.Module):
    """
    GCN_Encoder(
        GCN(mv, 0.8mv),
        BN(),
        ReLU(),
        GCN(0.8mv, c),
        BN(),
    ), where mv is view_dim, c is clusterNum.
    """

    def __init__(self, view_dim, clusterNum):
        super().__init__()
        self.mv = view_dim
        self.c = clusterNum
        self.mv_ = int(round(0.8 * self.mv))
        self.conv1 = DenseGCNConv(in_channels=self.mv, out_channels=self.mv_)
        # self.bn1 = nn.BatchNorm1d(num_features=self.mv_)
        self.relu = nn.ReLU()
        self.conv2 = DenseGCNConv(in_channels=self.mv_, out_channels=self.c)
        # self.bn2 = nn.BatchNorm1d(num_features=self.c)

    def forward(self, x, a):
        x = self.conv1(x, a).squeeze()
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x, a).squeeze()
        # x = self.bn2(x)
        return x

class ViewAE(nn.Module):

    def __init__(self, n_input, n_z):
        super(ViewAE, self).__init__()
        self.dim1 = int(round(0.8 * n_input))
        self.dim2 = 1500
        self.n_input = n_input
        self.n_z = n_z

        self.encoder = nn.Sequential(
            nn.Linear(n_input, self.dim1),
            nn.ReLU(),
            nn.BatchNorm1d(self.dim1),
            nn.Linear(self.dim1, self.dim1),
            nn.ReLU(),
            nn.BatchNorm1d(self.dim1),
            nn.Linear(self.dim1, self.dim2),
            nn.ReLU(),
            nn.BatchNorm1d(self.dim2),
            nn.Linear(self.dim2, n_z),
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_z, n_z),  # [k,k]
            nn.ReLU(),
            nn.BatchNorm1d(n_z),
            nn.Linear(n_z, self.dim2),  # [k,1500]
            nn.ReLU(),
            nn.BatchNorm1d(self.dim2),
            nn.Linear(self.dim2, self.dim1),
            nn.ReLU(),
            nn.BatchNorm1d(self.dim1),
            nn.Linear(self.dim1, n_input),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
                nn.init.xavier_uniform_(m.weight)

def weights_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear,)):
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0.01)
    elif isinstance(m, (nn.BatchNorm1d,)):
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0)









