#!/usr/bin/env python3
"""Shared relevance filter: is a paper genuine *gene/genomic classification* and not image-based?

Word-boundary aware so short tokens like 'gene'/'dna'/'rna' don't match
'generate'/'general'/'learning'. Used by both the OpenAlex and OpenReview harvesters.
"""
import re

# Short tokens that MUST match as whole words (else 'gene' hits 'generate').
# NOTE: 'variant'/'mutation' deliberately EXCLUDED -- they are ML jargon
# ('model variant', 'mutation' in evolutionary algorithms) and flooded the set
# with point-cloud / time-series papers. Only unambiguously genomic tokens here.
_SHORT = re.compile(r"\b(gene|genes|dna|rna|codon|codons|allele|alleles|exon|intron|genotype|genotypes|nucleotide|nucleotides)\b")

# Longer / multiword signals where a plain substring is safe and specific.
_LONG = [
    "genomic", "genome", "genomics", "transcriptom", "epigenom", "chromosom",
    "methylation", "single-cell", "single cell",
    "cell type", "cell-type", "gene expression", "gene regulatory", "regulatory network",
    "scrna", "rna-seq", "rna seq", "biological sequence", "protein sequence",
    "dna sequence", "gene ontology", "microbiome", "phenotype",
]

# Image-/vision-based work to exclude (user wants sequence/expression, not imaging).
_IMAGE = [
    "image", "imaging", "vision", "microscop", "histopath", "histolog",
    "radiolog", "mri", "ct scan", "x-ray", "photograph", "pixel", "genomap",
    "whole-slide", "pathology slide", "remote sensing", "video",
]


def is_gene(text):
    t = (text or "").lower()
    if _SHORT.search(t):
        return True
    return any(s in t for s in _LONG)


def is_image_based(text):
    t = (text or "").lower()
    return any(s in t for s in _IMAGE)


def keep(title, abstract, drop_image=True):
    text = (title or "") + " " + (abstract or "")
    if not is_gene(text):
        return False
    if drop_image and is_image_based(text):
        return False
    return True
