""" Components of the model (GPU-ready) """
import torch
import torch.nn as nn
import torch.nn.functional as F


def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_normal_(self.weight)

        # SAFE bias handling
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, adj):
        # x: [N, in_features] (dense)
        # adj: [N, N] (can be dense or sparse COO)
        support = x @ self.weight  # [N, out_features]

        if adj.is_sparse:
            output = torch.sparse.mm(adj, support)
        else:
            output = adj @ support

        if self.bias is not None:
            output = output + self.bias
        return output


class GCN_E(nn.Module):
    def __init__(self, in_dim, hgcn_dim, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim,     hgcn_dim[0])
        self.gc2 = GraphConvolution(hgcn_dim[0], hgcn_dim[1])
        self.gc3 = GraphConvolution(hgcn_dim[1], hgcn_dim[2])
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc2(x, adj)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc3(x, adj)
        x = F.leaky_relu(x, 0.25)
        return x


class Classifier_1(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        return self.clf(x)


class VCDN(nn.Module):
    def __init__(self, num_view, num_cls, hvcdn_dim):
        super().__init__()
        self.num_cls = num_cls
        self.model = nn.Sequential(
            nn.Linear(pow(num_cls, num_view), hvcdn_dim),
            nn.LeakyReLU(0.25),
            nn.Linear(hvcdn_dim, num_cls)
        )
        self.model.apply(xavier_init)

    def forward(self, in_list):
        # in_list: list of logits per view (B, num_cls)
        num_view = len(in_list)
        in_list = [torch.sigmoid(t) for t in in_list]

        # Outer-product chaining across views
        x = torch.reshape(
            torch.matmul(in_list[0].unsqueeze(-1), in_list[1].unsqueeze(1)),
            (-1, pow(self.num_cls, 2), 1)
        )
        for i in range(2, num_view):
            x = torch.reshape(
                torch.matmul(x, in_list[i].unsqueeze(1)),
                (-1, pow(self.num_cls, i + 1), 1)
            )

        vcdn_feat = torch.reshape(x, (-1, pow(self.num_cls, num_view)))
        output = self.model(vcdn_feat)
        return output


def init_model_dict(num_view, num_class, dim_list, dim_he_list, dim_hc,
                    gcn_dropout=0.5, device=None):
    """
    Build models and (optionally) move them to GPU if `device='cuda'`.
    """
    model_dict = {}
    for i in range(num_view):
        e = GCN_E(dim_list[i], dim_he_list, gcn_dropout)
        c = Classifier_1(dim_he_list[-1], num_class)
        if device is not None:
            e = e.to(device)
            c = c.to(device)
        model_dict[f"E{i+1}"] = e
        model_dict[f"C{i+1}"] = c

    if num_view >= 2:
        vc = VCDN(num_view, num_class, dim_hc)
        if device is not None:
            vc = vc.to(device)
        model_dict["C"] = vc
    return model_dict


def init_optim(num_view, model_dict, lr_e=1e-4, lr_c=1e-4, weight_decay=0.0):
    """
    Initialize optimizers (works on CPU or GPU transparently).
    """
    optim_dict = {}
    for i in range(num_view):
        params = list(model_dict[f"E{i+1}"].parameters()) + \
                 list(model_dict[f"C{i+1}"].parameters())
        optim_dict[f"C{i+1}"] = torch.optim.Adam(params, lr=lr_e, weight_decay=weight_decay)

    if num_view >= 2:
        optim_dict["C"] = torch.optim.Adam(model_dict["C"].parameters(), lr=lr_c, weight_decay=weight_decay)
    return optim_dict


# Optional: slightly faster matmul kernels in PyTorch 2.x on Ampere+
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
