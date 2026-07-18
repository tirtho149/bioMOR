# Gene / pathway downsampling: the models and theory

How the model reduces `N` genes (thousands) down to `M` marker/pathway tokens
(hundreds), and the theory behind each mechanism. All code lives in
`recursive_marker_transformer/marker.py`.

There are **two families**: *learned* downsampling (data-driven marker genes) and
*fixed biological* downsampling (Reactome pathways). Both expose the same
`weights()` / `selected_indices()` interface, so the rest of the model is identical
regardless of which one is chosen.

---

## 1. Learned gene downsampling — differentiable subset selection

### (a) Concrete / Gumbel-Softmax feature selection — `ConcreteSelector`

**Theory:** the **Concrete / Gumbel-Softmax relaxation** of a discrete choice.
Cited in the code: Jang, Gu & Poole 2017 (*Categorical Reparameterization with
Gumbel-Softmax*) and Balin, Abid & Zou 2019 (*Concrete Autoencoders*, differentiable
feature selection).

**Idea:** hard "pick K of N genes" is non-differentiable in *which* genes are
picked. Replace each pick with a temperature-annealed Gumbel-softmax sample over all
`N` genes: hot temperature -> near-uniform (gradient reaches every gene), cold ->
near one-hot. At eval, take the hard `argmax` gene. This lets the model learn *which*
genes to keep, unlike hard top-k which can only re-rank a frozen set.

```python
# marker.py:72
class ConcreteSelector(nn.Module):
    """Concrete / Gumbel-softmax differentiable feature selection
    (Balin, Abid & Zou 2019; Jang, Gu & Poole 2017)."""
    def __init__(self, n_genes, n_markers, temp_start=1.0, temp_end=0.1):
        ...
        # Peaked init: each selector starts ~one-hot on a distinct random gene,
        # so training begins at random-selection quality and *improves*.
        logits = 0.01 * torch.randn(self.n_markers, n_genes)
        spike = torch.randperm(n_genes)[: self.n_markers]
        logits[torch.arange(self.n_markers), spike] = 10.0
        self.logits = nn.Parameter(logits)

    def weights(self, gene_identity=None):
        if self.training:
            u = torch.rand_like(self.logits).clamp_(1e-9, 1.0)
            g = -torch.log(-torch.log(u))                                  # Gumbel noise
            return torch.softmax((self.logits + g) / self.temp.clamp_min(1e-4), dim=1)
        idx = self.logits.argmax(dim=1)                                    # hard at eval
        return torch.zeros_like(self.logits).scatter_(1, idx.unsqueeze(1), 1.0)
```

- **`marker.py:107-109`** — the Gumbel-softmax sample (`g = -log(-log(u))`, then
  temperature-scaled softmax over genes).
- **`marker.py:110-111`** — hard one-hot argmax at inference (discrete, interpretable
  marker).
- **`marker.py:92-95`** — peaked init (start ~one-hot, improve from random quality).

### (b) Cross-attention slot bottleneck — `SlotRouter`

**Theory:** the **induced-point / set-bottleneck** family — Set Transformer (induced
points), Perceiver, and Slot Attention. `M` learnable query "slots" cross-attend over
the `N` gene embeddings (keys); a temperature-annealed softmax over genes routes each
slot; at eval each slot collapses to its arg-max gene.

```python
# marker.py:117
class SlotRouter(nn.Module):
    """Cross-attention 'slot' router (Set Transformer induced points / Perceiver /
    Slot Attention)."""
    def _logits(self, gene_identity):
        k = self.key(gene_identity)                          # (N, d)
        return (self.queries @ k.t()) * self.scale           # (M, N)

    def weights(self, gene_identity):
        logits = self._logits(gene_identity)
        if self.training:
            return torch.softmax(logits / self.temp.clamp_min(1e-4), dim=1)
        idx = logits.argmax(dim=1)
        return torch.zeros_like(logits).scatter_(1, idx.unsqueeze(1), 1.0)
```

- **`marker.py:145-152`** — `M` queries x `N` gene keys -> softmax over genes
  (temperature-annealed).

### (c) Original SMART hard top-K + straight-through gate — `MarkerHead` / `MarkerModule`

**Theory:** hard top-K selection made trainable with a **straight-through-style soft
gate** `sigmoid(score)` (Bengio-style straight-through estimator). A per-gene MLP
scores every gene; the top-K become marker tokens; every non-marker gene is folded
into its **nearest marker by cosine similarity**, compressing attention from
**O(N^2) -> O(M^2)**.

```python
# marker.py:42
class MarkerHead(nn.Module):
    """Per-gene importance score from the (batch-independent) gene identity."""
    def __init__(self, d_model):
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                 nn.Linear(d_model, 1))
    def forward(self, gene_identity):
        return self.net(gene_identity).squeeze(-1)          # (N,)

# marker.py:229 (MarkerModule.select, mode="learnable")
scores     = self.head(gene_identity)                       # (N,)
marker_idx = torch.topk(scores, m).indices                  # hard top-K
gate       = torch.sigmoid(scores[marker_idx])              # (M,) differentiable gate
```

- **`marker.py:231`** — hard top-K choice (non-differentiable in *which*).
- **`marker.py:232`** — `sigmoid(score)` soft gate so gradients still flow to the
  scoring head (straight-through style; docstring `marker.py:27-31`).

---

## 2. Fixed biological (pathway) downsampling — P-NET pooling — `PathwayPooler`

**Theory:** this is **P-NET** (Elmarakeby et al., *Nature* 2021) -- a biologically
informed sparse gene->pathway layer -- recast as interpretable tokens. The biology
(Reactome membership) is **fixed**, not learned; only a per-pathway selectivity gate
trains.

```python
# marker.py:160
class PathwayPooler(nn.Module):
    """Fixed Reactome gene->pathway membership pooling -> M *pathway tokens*.
    ... This is P-NET's (Elmarakeby et al. 2021) gene->pathway sparse layer,
    recast as interpretable tokens ..."""
    def __init__(self, membership, pool="mean"):
        P = membership.float()                                  # (N_genes, M_pathways)
        if pool == "sum":
            W = P.t().contiguous()                             # (M, N) raw -> burden
        elif pool == "mean":
            col = P.sum(dim=0).clamp_min(1.0)                  # members per pathway
            W = (P / col).t().contiguous()                    # (M, N) rows ~sum to 1
        self.register_buffer("Wn", W, persistent=True)
        self.gate = nn.Parameter(torch.zeros(self.n_markers))  # sigmoid(0)=0.5

    def weights(self, gene_identity=None):
        """(M, N) pooling weights = per-pathway gate x normalised membership."""
        return torch.sigmoid(self.gate).unsqueeze(1) * self.Wn
```

- **`marker.py:187-191`** — `sum` pooling = mutation burden (raw membership); `mean`
  pooling = scale-free (row-normalised, for dense assays like CNV/expression).
- **`marker.py:195,199`** — the only learnable part: per-pathway gate
  `sigmoid(g_m)` (the "selective" property, at pathway granularity). Membership is
  fixed biology.

---

## 3. Cross-cutting mechanisms (both families)

**Temperature annealing** — a geometric schedule from near-uniform (explore all
genes) to near-hard (sharp selection) over training:

```python
# marker.py:100 (ConcreteSelector) / marker.py:141 (SlotRouter)
def set_progress(self, p):
    p = min(1.0, max(0.0, float(p)))
    self.temp.fill_(self.temp_start * (self.temp_end / self.temp_start) ** p)
```

**Peaked initialization** (`marker.py:88-95`) — each selector starts ~one-hot on a
random gene, so training begins at random-selection quality and improves, avoiding the
slow "uniform mush" cold start.

**Recursive refinement** — during the recursive MoR passes, marker tokens are
re-scored and re-gated, so a marker that stops being informative gets down-weighted
(the closed feedback loop):

```python
# marker.py:57
class RefineHead(nn.Module):
    """Per-token gate from the *current* contextual marker embedding (B, M, d)."""
    def forward(self, tokens):
        return self.net(tokens).squeeze(-1)                 # (B, M)
```

---

## One-line summary

**Learned downsampling** = Gumbel-Softmax/Concrete relaxation (or an attention
induced-point bottleneck) with temperature annealing + straight-through hard selection
at eval. **Biological downsampling** = P-NET's fixed Reactome gene->pathway pooling
with a learnable per-pathway selectivity gate.

## References

- Jang, Gu & Poole (2017). *Categorical Reparameterization with Gumbel-Softmax.* ICLR.
- Balin, Abid & Zou (2019). *Concrete Autoencoders: Differentiable Feature Selection
  and Reconstruction.* ICML.
- Lee et al. (2019). *Set Transformer.* ICML (induced points).
- Jaegle et al. (2021). *Perceiver.* ICML.
- Locatello et al. (2020). *Object-Centric Learning with Slot Attention.* NeurIPS.
- Elmarakeby et al. (2021). *Biologically informed deep neural network for prostate
  cancer discovery.* Nature (P-NET).
- Bengio, Leonard & Courville (2013). *Estimating or Propagating Gradients Through
  Stochastic Neurons* (straight-through estimator).
