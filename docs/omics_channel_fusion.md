# How mutation and CNV values are fused (multi-omics input)

Traced from the actual code (`pathway_data.py`, `marker.py`, `model.py`, `embedding.py`).

## Short version

Mutation and CNV are **not** merged into one number up front. They are carried as two
parallel channels, pooled into pathway tokens **separately**, and then fused by a
**single shared linear layer** (`value_proj`, a `Linear(2 -> d_model)`) whose output is
added to each token's identity embedding. It is a learned **early fusion at the input
embedding** — not a concatenation-then-MLP, and not attention.

## Step by step

**1. Load as two aligned channels.** `load_cohort` reads `mutation_data.csv` (patients x
genes, binary 0/1) and `cnv_data.csv` (patients x genes, integers in {-2..+2}). Both are
aligned to the same patient set and the same shared gene set, then stacked on a new last
axis: `X` has shape `(N_patients, G_genes, C=2)`, where channel 0 = mutation, channel 1 =
CNV. At this point there is no fusion — just two parallel per-gene layers.

**2. Pool genes -> pathway tokens, each channel independently.** With
`marker_mode=pathway`, a fixed binary Reactome membership matrix (G genes x M pathways)
collapses genes into M pathway tokens. The **same** pooling weights `w` are applied to each
channel separately:

    sel_value[b, m, c] = sum_gene  w[m, gene] * X[b, gene, c]   ->  shape (B, M, 2)

So every pathway token ends up with a **2-number summary**:
`[mutation-value-of-pathway, CNV-value-of-pathway]`.

- With `--pathway_pool sum` (the setting used for `mut_cnv`), the mutation channel becomes
  the pathway's **mutation burden** = count of mutated member genes. Sum is chosen because
  binary mutation is sparse; mean-pooling washes it to a near-constant and the model
  collapses.
- A learnable per-pathway gate `sigmoid(g_m)` scales each pathway's weight, letting the
  model down-weight uninformative pathways (the "selective" part). The membership itself is
  fixed biology; only the gate trains.

**3. The fusion itself.** A single linear layer `value_proj = nn.Linear(n_channels=2,
d_model)` is applied to each token's 2-vector:

    value[b, m, :] = W * [mut_m, cnv_m] + bias      # W is (d_model x 2)

This is the actual fusion: **each output embedding dimension is a learned weighted mix of
the mutation value and the CNV value** (plus bias). Mutation and CNV become one
`d_model`-dimensional vector per token. The **same `W` is shared across all tokens** — the
model learns one global weighting of "how much mutation vs. copy-number matters," rather
than combining them per-gene by hand.

**4. Add to token identity.** The fused omics vector is added to the pathway **identity**
embedding:

    token_m = pathway_identity_m + value_proj([mut_m, cnv_m])   # then LayerNorm + dropout

Identity encodes *which* pathway the token is; the fused value encodes *how mutated /
copy-altered* that pathway is in this patient. These M fused tokens then flow into the
recursive MoR stack.

## Notes

- **3M (`mut_cnv_expr`)** is identical with `C=3`: `value_proj` is `Linear(3 -> d_model)`,
  fusing mutation, CNV, and expression the same way.
- The raw mutation and CNV magnitudes are **not** pre-normalized relative to each other —
  the linear layer learns the relative scaling.
- Because fusion is one shared linear map added to identity, the model is deliberately
  simple/interpretable here (it mirrors P-NET's gene->pathway sparse layer recast as
  tokens), with the heavier modeling happening downstream in the recursive transformer.

## Exact code locations

### `model.py` — pathway / multimodal fusion (selector branch)

```python
441        if self.selector is not None:
442            # Soft selection (Concrete or cross-attention router): soft (train) /
443            # hard (eval) selection over ALL genes -> M marker tokens directly, so
444            # gradients reach every gene (the property hard top-k routing lacks).
445            w = self.selector.weights(gene_identity)           # (M, N)
446            sel_ident = w @ gene_identity                      # (M, d)
447            if x.dim() == 3:
448                # Multimodal: pool each gene-aligned channel by the same marker
449                # weights, then fuse the C channels at the shared value projection.
450                sel_value = torch.einsum("bnc,mn->bmc", x, w)  # (B, M, C)
451                cluster = sel_ident.unsqueeze(0) + self.embed.value_proj(sel_value)
452            else:
453                sel_value = x @ w.t()                          # (B, M) selected expression
454                cluster = sel_ident.unsqueeze(0) + self.embed.value_proj(sel_value.unsqueeze(-1))
```

- **`model.py:450`** — pool each channel by the shared pathway weights -> `(B, M, C)`.
- **`model.py:451`** — fuse the C channels through `value_proj`, add the pathway identity.

Optional per-channel biological smoothing that runs *before* pooling (each omics channel is
denoised along the same gene-gene graph):

```python
408        elif self._bio_prop and self._bio_have_graph and x.dim() == 3:
409            # Multi-modal (B,G,C): smooth EACH omics channel along the same gene-gene
410            # graph, so mutation and copy-number both denoise along the biological network
411            # (this is what lets a fixed aggregated network act as a prior on P-NET too).
412            lam = torch.sigmoid(self.bio_prop_logit)
413            xs = x
414            for _ in range(max(1, self._bio_hops)):
415                xs = (1.0 - lam) * xs + lam * torch.einsum("bgc,gh->bhc", xs, self.bio_operator)
416            x = xs
```

### `marker.py` — the gene->pathway pooling weights (`PathwayPooler`)

```python
182    def __init__(self, membership: torch.Tensor, pool: str = "mean"):
183        super().__init__()
184        P = membership.float()                                  # (N, M)
185        self.n_genes, self.n_markers = P.shape
186        self.pool = pool
187        if pool == "sum":
188            W = P.t().contiguous()                             # (M, N) raw -> burden
189        elif pool == "mean":
190            col = P.sum(dim=0).clamp_min(1.0)                  # members per pathway
191            W = (P / col).t().contiguous()                    # (M, N) rows ~sum to 1
192        else:
193            raise ValueError(f"Unknown pathway pool: {pool!r}")
194        self.register_buffer("Wn", W, persistent=True)
195        self.gate = nn.Parameter(torch.zeros(self.n_markers))  # sigmoid(0)=0.5
196
197    def weights(self, gene_identity: torch.Tensor = None) -> torch.Tensor:
198        """(M, N) pooling weights = per-pathway gate x normalised membership."""
199        return torch.sigmoid(self.gate).unsqueeze(1) * self.Wn
```

- **`marker.py:187-191`** — `sum` pooling = mutation burden (raw membership); `mean`
  pooling = row-normalised (for dense assays).
- **`marker.py:197-199`** — final pooling weights `w` = per-pathway gate `sigmoid(g_m)` x
  fixed membership `Wn`. This is the `w` used at `model.py:450`.

### `embedding.py` — the shared fusion layer (`value_proj`)

```python
45        self.value_proj = nn.Linear(n_channels, d_model)
...
50    def forward(self, x: torch.Tensor) -> torch.Tensor:
51        """x: (B, N) scalar or (B, N, C) multichannel -> (B, N, d_model) tokens."""
52        if x.dim() == 2:
53            x = x.unsqueeze(-1)                              # (B, N, 1)
54        gene = self.gene_emb(self.gene_ids)                  # (N, d)
55        value = self.value_proj(x)                           # (B, N, d)
56        tokens = gene.unsqueeze(0) + value                   # broadcast over batch
```

- **`embedding.py:45`** — the fusion layer: `Linear(n_channels -> d_model)` (2 for
  `mut_cnv`, 3 for `mut_cnv_expr`).
- **`embedding.py:55-56`** — `value = value_proj(x)` fuses the channels; `tokens =
  gene_identity + value` adds it to the identity embedding (the gene-token analogue of the
  pathway-token fusion at `model.py:451`).

### `pathway_data.py` — where the channels are stacked

```python
213    X = chans[0] if len(chans) == 1 else np.stack(chans, axis=-1)   # (N, G, C)
```

- **`pathway_data.py:213`** — stacks the per-modality matrices into `(N, G, C)`; channel
  order follows `CHANNEL_SETS` (`mut_cnv` -> `["mut", "cnv"]`).
