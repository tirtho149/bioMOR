"""
Adapted Utils for MOGONET
Added Jaccard similarity support for binary mutation data
(GPU-safe: device-aware tensors & sparse ops)
"""
import os
import numpy as np
import torch
import torch.nn.functional as F

# ---- Device setup ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ----------------------

def cal_sample_weight(labels, num_class, use_sample_weight=True):
    """Calculate sample weights - same as original"""
    if not use_sample_weight:
        return np.ones(len(labels)) / len(labels)
    count = np.zeros(num_class)
    for i in range(num_class):
        count[i] = np.sum(labels == i)
    sample_weight = np.zeros(labels.shape)
    for i in range(num_class):
        sample_weight[np.where(labels == i)[0]] = count[i] / np.sum(count)
    return sample_weight


def one_hot_tensor(y, num_dim):
    """One-hot encoding (device-aware)"""
    y_onehot = torch.zeros(y.shape[0], num_dim, device=y.device, dtype=torch.float32)
    y_onehot.scatter_(1, y.view(-1, 1), 1.0)
    return y_onehot


def cosine_distance_torch(x1, x2=None, eps=1e-8):
    """Cosine distance - same as original, device-aware"""
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def jaccard_distance_torch(x1, x2=None, eps=1e-8):
    """
    Jaccard distance for binary data
    """
    x2 = x1 if x2 is None else x2
    # Ensure binary data
    x1_bin = (x1 > 0.5).float()
    x2_bin = (x2 > 0.5).float()

    # Intersection / union
    intersection = torch.mm(x1_bin, x2_bin.t())
    x1_sum = x1_bin.sum(dim=1, keepdim=True)
    x2_sum = x2_bin.sum(dim=1, keepdim=True)
    union = x1_sum + x2_sum.t() - intersection
    union = union.clamp(min=eps)

    jaccard_sim = intersection / union
    jaccard_dist = 1 - jaccard_sim
    return jaccard_dist


def to_sparse(x: torch.Tensor) -> torch.Tensor:
    """
    Convert a dense tensor to a sparse COO tensor (device-aware).
    """
    if x.is_sparse:
        return x
    # indices: [2, nnz], values: [nnz]
    idx = (x != 0).nonzero(as_tuple=False).t()
    if idx.numel() == 0:
        # all zeros
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long, device=x.device),
            torch.empty((0,), dtype=x.dtype, device=x.device),
            x.size(),
            device=x.device,
        )
    vals = x[idx[0], idx[1]]
    return torch.sparse_coo_tensor(idx, vals, x.size(), device=x.device)


def cal_adj_mat_parameter(edge_per_node, data, metric="cosine"):
    """Calculate adjacency threshold parameter (kNN-like)"""
    if metric == "cosine":
        dist = cosine_distance_torch(data, data)
    elif metric == "jaccard":
        dist = jaccard_distance_torch(data, data)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    flat = torch.sort(dist.reshape(-1,)).values
    # select kth distance as threshold
    parameter = flat[edge_per_node * data.shape[0]]
    return parameter.detach().cpu().numpy().item()


def graph_from_dist_tensor(dist, parameter, self_dist=True):
    """Binary graph mask from distance (<= parameter)"""
    if self_dist:
        assert dist.shape[0] == dist.shape[1], "Input is not pairwise dist matrix"
    g = (dist <= parameter).float()
    if self_dist:
        diag_idx = torch.arange(g.shape[0], device=g.device)
        g[diag_idx, diag_idx] = 0.0
    return g


def _build_adj_from_dist(data, metric):
    """Helper: return (dist, adj_sim) on the same device as data"""
    if metric == "cosine":
        dist = cosine_distance_torch(data, data)
        adj = 1 - dist
    elif metric == "jaccard":
        dist = jaccard_distance_torch(data, data)
        adj = 1 - dist
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return dist, adj


def gen_adj_mat_tensor(data, parameter, metric="cosine"):
    """
    Build normalized (row-stochastic) sparse adjacency with self-loops.
    data: [N, D] on device
    """
    dist, adj = _build_adj_from_dist(data, metric)

    # Mask by threshold
    g = graph_from_dist_tensor(dist, parameter, self_dist=True)
    adj = adj * g

    # Symmetrize by max
    adj_T = adj.t()
    adj = adj + adj_T * (adj_T > adj).float() - adj * (adj_T > adj).float()

    # Add self-loops and row-normalize
    I = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
    adj = F.normalize(adj + I, p=1, dim=1)

    # Convert to sparse
    adj = to_sparse(adj)
    return adj


def gen_test_adj_mat_tensor(data, tr_idx, te_idx, parameter, metric="cosine"):
    """
    Build adjacency matrix for train+test subset ONLY.
    
    Args:
        data: Full dataset tensor
        tr_idx: Training indices (list)
        te_idx: Test/validation indices (list)
        parameter: Adjacency threshold
        metric: "cosine" or "jaccard"
    
    Returns:
        Sparse adjacency matrix of shape [len(tr_idx) + len(te_idx), len(tr_idx) + len(te_idx)]
    """
    # Convert to lists if needed
    tr = tr_idx if isinstance(tr_idx, list) else list(tr_idx)
    te = te_idx if isinstance(te_idx, list) else list(te_idx)
    
    num_tr = len(tr)
    num_te = len(te)
    N = num_tr + num_te
    
    # Extract data subsets
    data_tr = data[tr]
    data_te = data[te]
    
    # Create adjacency matrix
    adj = torch.zeros((N, N), device=data.device, dtype=torch.float32)

    # Compute distances
    if metric == "cosine":
        dist_tr2te = cosine_distance_torch(data_tr, data_te)
        dist_te2tr = cosine_distance_torch(data_te, data_tr)
    elif metric == "jaccard":
        dist_tr2te = jaccard_distance_torch(data_tr, data_te)
        dist_te2tr = jaccard_distance_torch(data_te, data_tr)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Similarities
    sim_tr2te = 1 - dist_tr2te
    sim_te2tr = 1 - dist_te2tr

    # Threshold masks
    g_tr2te = graph_from_dist_tensor(dist_tr2te, parameter, self_dist=False)
    g_te2tr = graph_from_dist_tensor(dist_te2tr, parameter, self_dist=False)

    # Fill adjacency
    adj[:num_tr, num_tr:] = sim_tr2te * g_tr2te
    adj[num_tr:, :num_tr] = sim_te2tr * g_te2tr

    # Symmetrize
    adj_T = adj.t()
    adj = adj + adj_T * (adj_T > adj).float() - adj * (adj_T > adj).float()

    # Add self-loops & normalize
    I = torch.eye(N, device=adj.device, dtype=adj.dtype)
    adj = F.normalize(adj + I, p=1, dim=1)

    # Convert to sparse
    adj = to_sparse(adj)
    return adj


def save_model_dict(folder, model_dict):
    """Save model dictionary"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    for module in model_dict:
        torch.save(model_dict[module].state_dict(), os.path.join(folder, module + ".pth"))


def load_model_dict(folder, model_dict):
    """Load model dictionary (device-aware)"""
    map_loc = device
    for module in model_dict:
        pth = os.path.join(folder, module + ".pth")
        if os.path.exists(pth):
            model_dict[module].load_state_dict(torch.load(pth, map_location=map_loc))
        else:
            print(f"WARNING: Module {module} from model_dict is not loaded!")
        model_dict[module].to(device)
    return model_dict
