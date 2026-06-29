# ============================================================================
# SMART -- self-contained paper whose narrative MATCHES the new content: a
# systematic reproduction of the Mixture-of-Recursions (Bae et al. 2025) design
# decisions with SMART on the genomap single-cell suite + pathway/P-NET multi-omics.
# build() returns the full LaTeX; the results section is \input{mor_tables} (all 14
# MoR tables + adaptive-depth + pathway + figures), rendered from the fresh runs.
# ============================================================================
from __future__ import annotations


def build() -> str:
    return r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{placeins}
\usepackage{amsmath,amssymb}
\usepackage{times}
\usepackage[hidelinks]{hyperref}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf}
\title{\textbf{SMART: A Selective Marker-guided Adaptive Recursive Transformer\\
for Genomic Classification}\\[4pt]
\large Reproducing the Mixture-of-Recursions design space on the genomap suite}
\author{Koushik Howlader\textsuperscript{1} \and Tirtho Roy\textsuperscript{1}
\and Md Tauhidul Islam\textsuperscript{2} \and Wei Le\textsuperscript{1}\\[2pt]
\textsuperscript{1}Iowa State University \quad \textsuperscript{2}Stanford University}
\date{}

\begin{document}
\maketitle

\begin{abstract}
SMART is a recursive transformer for genomic classification that (i) compresses
thousands of genes into a small set of \emph{interpretable tokens} -- learned
marker genes for single-cell expression, or curated Reactome \emph{pathway} tokens
for multi-omics -- and (ii) processes them with a single weight-shared transformer
block applied recursively, where a Mixture-of-Recursions (MoR) router gives each
token its own recursion depth. This pairing lets us ask whether the design
decisions that make MoR effective for language models transfer to a
non-autoregressive biological set classifier. We therefore reproduce the entire
MoR ablation suite -- all fourteen tables of Bae et al.~(2025) -- with SMART on the
genomap single-cell datasets (Islam \& Xing, 2023) and on pathway-informed
multi-omics cohorts, with no TCGA. We study three claims: the \emph{adaptive
recursion loop}, \emph{token reduction}, and \emph{parameter reduction}. Weight
sharing yields an exact $K\times$ parameter reduction (e.g.\ $4\times$ at $K{=}4$)
while matching the untied baseline; adaptive per-token depth cuts recursion compute
by roughly a third at parity with fixed depth; and a handful of interpretable
tokens recover most of the full-gene accuracy. We report every result honestly,
including where adaptive routing does not help, so the tables read as a transparent
map of which MoR ingredients matter for genomics.
\end{abstract}

\section{Introduction}
Transcriptomic and multi-omics classifiers must map tens of thousands of genes to a
phenotype from few labelled samples. SMART addresses this with two ideas. First,
\textbf{token reduction}: rather than attend over all genes, it selects a small set
of tokens that carry biological meaning -- learned marker genes, or fixed Reactome
pathway tokens that pool their member genes. Second, an \textbf{adaptive recursion
loop}: a single transformer block is applied $K$ times with weight sharing, and a
Mixture-of-Recursions router assigns each token its own depth, so compute is spent
where it helps. Weight sharing also gives a direct \textbf{parameter reduction}.

These are precisely the levers studied by the Mixture-of-Recursions paper for
language models. Because SMART is a one-shot set encoder (no autoregression, no
KV-cache across positions), it is not obvious that MoR's conclusions carry over. We
test this directly by reproducing \emph{all fourteen} MoR tables on biological data.
Two of them (uptraining, KV-cache sharing) have no literal analogue for a set
encoder; we reproduce them as documented analogues -- warm-start and step-cache --
and label them as such rather than fabricate numbers.

\paragraph{Contributions.}
(1) A faithful reproduction of the full MoR design-space study with SMART on the
genomap suite and pathway multi-omics, with no TCGA. (2) The three SMART claims --
adaptive recursion, token reduction, parameter reduction -- quantified by the
corresponding MoR experiments (Section~\ref{sec:results}). (3) Honest negative
results: adaptive routing is dataset-dependent and the curated-prior gains are
modest, reported transparently.

\section{The SMART model}
\textbf{Tokens.} A gene-identity embedding plus a value projection embeds each
input feature; SMART then forms $M$ tokens. For single-cell expression the tokens
are learned marker slots (a cross-attention router over genes); for multi-omics the
tokens are the curated Reactome pathways, each pooling its member genes'
mutation/CNV/expression channels (mean pooling for dense assays, burden/sum pooling
for sparse mutation). Either way the tokens are interpretable gene or pathway
identities, and the cost of the downstream attention is $O(M^2)$ with $M\!\ll\!$
\#genes.

\textbf{Recursive stack.} One pre-norm transformer block is applied for $K$
recursion steps. Parameter sharing across the $K$ steps follows the MoR schemes
(Cycle, Sequence, Middle-Cycle, Middle-Sequence; Table~\ref{tab:mor-t1}), between
the fully-shared and fully-independent extremes.

\textbf{Adaptive depth (MoR routing).} An expert-choice router keeps a capacity
top-$k$ of tokens at each step, funnelling survivors deeper; a token's survival
depth is an intrinsic importance signal. A token-choice variant lets each token pick
one depth up front. A label-free biological prior (co-expression centrality, or the
Reactome pathway-hierarchy graph) can bias the depth logits.

\textbf{Step-cache.} As a set-encoder analogue of MoR's recursion-wise KV-cache
sharing, the step-1 attention keys/values can be reused across recursions rather
than recomputed.

\section{Datasets}
\textbf{Genomap suite (6).} The genomap-paper datasets, processed to genomap
features: Tabula Muris, pancreas, common\_class and prototype (the Code Ocean
capsule), plus Baron and Segerstolpe rebuilt from raw counts via genomap's own
construction. \textbf{Pathway/P-NET cohorts (6).} Reactome pathway-informed
multi-omics: prostate, bladder (BLCA), stomach (STAD), breast (BRCA), and the
pan-cancer metastatic-vs-primary and 32-class cancer-type tasks. All experiments
follow the genomap-paper protocol; no TCGA bulk data is used.

\section{Results}\label{sec:results}
We reproduce every MoR table; each is rendered from the fresh result directories and
named with the MoR table(s) it reproduces and the claim it supports. The headline
findings: \emph{(parameter reduction)} the weight-shared recursive stack uses an
exact $1/K$ of the parameters of an untied stack of the same depth (e.g.\ a $4\times$
reduction at $K{=}4$) while matching its accuracy (Tables~\ref{tab:mor-t6},
\ref{tab:mor-t3}); \emph{(adaptive depth)} expert-choice routing leaves a mean depth
well below $K$ and saves roughly a third of the recursion compute at parity with
fixed depth on most datasets, though the benefit is dataset-dependent
(Table~\ref{tab:mor-adaptive}, Fig.~\ref{fig:mor-depth}); \emph{(token reduction)} a
few hundred marker/pathway tokens recover most full-gene accuracy, and learned
selection is competitive with or better than random/variance panels
(Table~\ref{tab:mor-t4}). The same architecture ablation transfers to the pathway
multi-omics cohorts (Table~\ref{tab:mor-pathway}).

\FloatBarrier
\input{mor_tables}
\FloatBarrier

\section{Discussion}
The MoR ingredients transfer unevenly. Parameter reduction is unconditional: weight
sharing is essentially free here. Adaptive depth helps when the task has slack to
allocate -- on saturated datasets fixed depth or even a single pass is already
enough, and on the weakest dataset (Segerstolpe, whose genomap features are
under-resolved) no architecture choice rescues it; we report these cases rather than
hide them. Curated biological priors (co-expression, Reactome hierarchy) give small,
mostly within-noise gains, consistent with their being warm-start regularisers
rather than accuracy drivers. The two LLM-specific tables (warm-start for uptraining,
step-cache for KV sharing) are reproduced as honest analogues.

\section{Conclusion}
Treating SMART as a biological instantiation of Mixture-of-Recursions, we reproduced
the complete MoR design-space study on the genomap suite and pathway multi-omics.
The result is a transparent account of which recursive-transformer design decisions
matter for genomic classification: parameter sharing always, adaptive depth when the
task allows, interpretable token reduction throughout.

\end{document}
"""


if __name__ == "__main__":
    print(build())
