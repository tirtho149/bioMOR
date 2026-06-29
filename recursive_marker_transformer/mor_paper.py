# ============================================================================
# SMART -- full AAAI-style paper. One coherent story: SMART makes parameter
# efficiency architectural and biology part of the routing decision, validated by
# three claims (token reduction, adaptive recursion, parameter reduction) on the
# genomap single-cell suite + Reactome/P-NET multi-omics (no TCGA). The experiments
# mirror the Mixture-of-Recursions design space but are presented as the paper's own
# results (story-named tables via \input{mor_tables}). Figures use the MoR palette.
# ============================================================================
from __future__ import annotations


def build() -> str:
    return r"""\documentclass[letterpaper]{article}
\usepackage{aaai}
\usepackage{times}
\usepackage{helvet}
\usepackage{courier}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{xcolor}
\frenchspacing
\setlength{\pdfpagewidth}{8.5in}
\setlength{\pdfpageheight}{11in}
\setcounter{secnumdepth}{2}
\pdfinfo{
/Title (SMART: A Selective Marker-guided Adaptive Recursive Transformer for Genomic Classification)
/Author (Koushik Howlader, Tirtho Roy, Md Tauhidul Islam, Wei Le)
}

\title{SMART: A Selective Marker-guided Adaptive Recursive Transformer\\
for Genomic Classification}
\author{Koushik Howlader\textsuperscript{1} \and Tirtho Roy\textsuperscript{1}
\and Md Tauhidul Islam\textsuperscript{2} \and Wei Le\textsuperscript{1}\\
\textsuperscript{1}Iowa State University, Ames, Iowa, USA\\
\textsuperscript{2}Stanford University, Stanford, California, USA\\
weile@iastate.edu, tauhid@stanford.edu}

\begin{document}
\maketitle

\begin{abstract}
\begin{quote}
Transformer models for genomic classification treat every one of the thousands of
measured genes as an equally important token and stack many independent layers,
making them parameter-heavy and leaving \emph{which} genes deserve computation to be
learned implicitly. We argue that for genomics, parameter efficiency should be an
\emph{architectural} property and computation should be \emph{allocated adaptively}.
We present \textbf{SMART}, which (i) compresses thousands of genes into a small set
of interpretable tokens -- learned marker genes for single-cell expression, or
curated Reactome \emph{pathway} tokens for multi-omics -- cutting self-attention from
$\mathcal{O}(N^2)$ to $\mathcal{O}(M^2)$; (ii) processes those tokens with a
\emph{single} transformer block applied recursively, so parameters do not grow with
depth; and (iii) uses a Mixture-of-Recursions router to give each token its own
recursion depth, turning depth into an intrinsic compute-allocation importance score.
We validate the three resulting claims -- \textbf{token reduction}, an
\textbf{adaptive recursion loop}, and \textbf{parameter reduction} -- with a
controlled study across the genomap single-cell datasets and Reactome/P-NET
multi-omics cohorts, with no TCGA bulk data. Weight sharing yields an exact
$K\times$ parameter reduction (a $4\times$ reduction at $K{=}4$) at accuracy
comparable to independent layers; adaptive depth leaves a mean depth well below $K$
and saves roughly a third of the recursion compute at parity with fixed depth; and a
few dozen to a few hundred interpretable tokens recover most of the full-gene
accuracy. We report negative results transparently: adaptive routing helps only when
the task has computational slack, and label-free biological priors give small gains.
The whole pipeline -- training, ablations, and this paper -- regenerates from one
command.
\end{quote}
\end{abstract}

\section{Introduction}
Single-cell RNA sequencing and tumour multi-omics now profile thousands of genes per
sample, and transformer models such as scGPT, Geneformer and scBERT have adapted the
architecture of Vaswani et al.~(2017) to this modality. They inherit two costly
habits from language models. First, they treat \emph{every} gene as an equally
important token, so a housekeeping gene and a lineage-defining marker receive the same
budget and attention scales quadratically in the number of genes. Second, they stack
\emph{independent} layers, so parameters grow linearly with depth.

SMART rejects both. It keeps only a small set of \emph{interpretable tokens} and
applies a \emph{single} weight-shared transformer block recursively, with a router
that gives each token its own recursion depth. This is, in spirit, a biological
instantiation of Mixture-of-Recursions (Bae et al.,~2025): a recursive,
adaptively-routed transformer. Because SMART is a one-shot \emph{set} classifier
rather than an autoregressive model, it is not obvious which of those design
decisions still pay off. We therefore run a controlled study of all of them and
organise the paper around the three claims they support.

\paragraph{Contributions.}
(1) SMART, a recursive transformer that makes parameter efficiency architectural and
computation adaptive, with interpretable marker- or pathway-token inputs. (2) A
controlled validation of three claims -- token reduction, adaptive recursion,
parameter reduction -- on the genomap single-cell suite and Reactome/P-NET
multi-omics, with no TCGA. (3) Transparent negative results: where adaptive routing
and biological priors do and do not help.

\section{The SMART Model}
\paragraph{Interpretable tokens.}
A gene-identity embedding plus a value projection embeds each input feature. SMART
then forms $M\ll N$ tokens. For single-cell expression these are learned marker slots
produced by a cross-attention router whose queries attend over all genes; for
multi-omics they are the curated Reactome pathways, each pooling its member genes'
mutation, copy-number and expression channels (mean pooling for dense assays, burden
pooling for sparse mutation). Either way the tokens carry biological meaning and the
downstream attention costs $\mathcal{O}(M^2)$.

\paragraph{Recursive stack and parameter sharing.}
One pre-norm transformer block is applied for $K$ recursion steps. Sharing across the
$K$ steps ranges from a single block (full sharing) to $K$ independent blocks, with
graded Cycle/Sequence/Middle-Cycle/Middle-Sequence schemes in between.

\paragraph{Adaptive recursion depth.}
A Mixture-of-Recursions router allocates depth per token. Expert-choice routing keeps
a capacity top-$k$ of tokens at each step and funnels survivors deeper, so a token's
survival depth is an importance score; token-choice routing lets each token pick one
depth. A label-free biological prior (co-expression centrality, or the Reactome
pathway-hierarchy graph) can bias the depth logits.

\section{Experimental Setup}
\paragraph{Data.}
\emph{Genomap single-cell suite} (genomap features; Islam and Xing,~2023): Tabula
Muris, pancreas, common\_class and prototype from the genomap capsule, plus Baron and
Segerstolpe rebuilt from raw counts via genomap construction. \emph{Reactome/P-NET
multi-omics} (Elmarakeby et al.,~2021): prostate, bladder, stomach and breast
cohorts, and pan-cancer metastatic-vs-primary and 32-class cancer-type tasks. No TCGA
bulk data is used. We follow the genomap-paper protocol and report macro-F1
(mean$\pm$std over seeds where available).

\section{Results}\label{sec:results}
We organise results by claim. \textbf{Token reduction:} accuracy holds as the number
of interpretable tokens $M$ is reduced (Table~\ref{tab:tokens}), and learning which
genes are markers is competitive with or better than fixed panels
(Table~\ref{tab:selection}). \textbf{Adaptive recursion:} adaptive per-token depth
matches fixed depth while leaving the mean depth well below $K$ and saving roughly a
third of the recursion compute (Table~\ref{tab:adaptive},
Fig.~\ref{fig:depth}); the gain is dataset-dependent, and on saturated tasks a single
pass already suffices. \textbf{Parameter reduction:} one block applied recursively
uses $1/K$ of the parameters of $K$ independent blocks at comparable accuracy
(Tables~\ref{tab:config},~\ref{tab:scaling}; Fig.~\ref{fig:scaling},~\ref{fig:param}),
and graded sharing schemes interpolate the trade-off (Table~\ref{tab:sharing}). The
remaining tables probe where computation is spent (Table~\ref{tab:depth}), key/value
reuse (Table~\ref{tab:cache}), warm-starting (Table~\ref{tab:warmstart}) and routing
configurations (Table~\ref{tab:routing}); the same design decisions transfer to the
pathway multi-omics cohorts (Table~\ref{tab:pathway}).

\input{mor_tables}

\section{Discussion}
The three ingredients transfer unevenly. Parameter reduction is essentially free here:
weight sharing gives an exact $K\times$ reduction at accuracy comparable to untied
layers. Adaptive depth helps when the task has slack to allocate; on saturated
datasets fixed depth or a single pass is already enough, and on the weakest dataset
(Segerstolpe, whose genomap features are under-resolved) no architecture choice
rescues it -- we report these cases rather than hide them. Curated biological priors
give small, mostly within-noise gains, consistent with their being warm-start
regularisers rather than accuracy drivers.

\section{Conclusion}
SMART makes parameter efficiency architectural and computation adaptive for genomic
classification, with interpretable marker- or pathway-token inputs. A controlled study
on the genomap suite and Reactome/P-NET multi-omics validates token reduction and
parameter reduction unconditionally, and adaptive recursion where the task allows --
with negative results reported transparently.

\section*{References}
\small
\begin{description}
\item Bae, S.; et al. 2025. Mixture-of-Recursions. arXiv:2507.10524.
\item Elmarakeby, H.; et al. 2021. Biologically informed deep neural network for
prostate cancer discovery. \emph{Nature} 598:348--352.
\item Islam, M.\,T.; and Xing, L. 2023. Cartography of Genomic Interactions Enables
Deep Analysis of Single-Cell Expression Data. \emph{Nature Communications} 14:679.
\item Vaswani, A.; et al. 2017. Attention Is All You Need. In \emph{NeurIPS}.
\item Cui, H.; et al. 2024. scGPT. \emph{Nature Methods}.
\item Theodoris, C.; et al. 2023. Transfer learning enables predictions in network
biology (Geneformer). \emph{Nature} 618:616--624.
\end{description}

\end{document}
"""


if __name__ == "__main__":
    print(build())
