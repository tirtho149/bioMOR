"""
Convert the Graph_Transformer-style pathway files into P-NET's expected
Reactome layout under `_database/pathways/Reactome/`.

Source files:
  filtered_pathways.csv     cols: Pathway_ID, Pathway_Name, Genes
                            (Genes is a comma-separated string per row)
  HumanPathwaysRelation.csv cols: Parent, Child  (optional; if provided,
                            the real Reactome hierarchy is preserved instead
                            of synthesizing a flat DUMMY-target topology)

Output files (written to `_database/pathways/Reactome/`):
  ReactomePathways.txt              TSV: reactome_id, pathway_name, species
  ReactomePathways.gmt              TSV: pathway_name, pathway_id, descrip, gene1, gene2, …
                                    (loaded by GMT.load_data with pathway_col=1, genes_col=3)
  ReactomePathwaysRelation.txt      TSV: child, parent

Topologies produced
-------------------
Two modes, selected by whether `--relations` is provided:

(A) FLAT (no --relations):
    Each pathway points at a single synthetic node HSA-DUMMY-TARGET. After
    P-NET adds 'root' (in_degree=0 nodes become root's children), every
    pathway becomes a direct child of root. Use `n_hidden_layers=1`.

(B) HIERARCHICAL (--relations supplied):
    Real Reactome parent→child edges are kept, restricted to relations where
    BOTH endpoints are in the kept pathway set. Pathways with no parent in
    the kept set become roots (direct children of P-NET's synthetic 'root').
    P-NET's `complete_network(G, n_levels=K)` pads any branch shorter than K
    with `_copy` nodes, so `n_hidden_layers=5` works on any actual depth.

Usage
-----
    cd /lustre/hdd/LAS/weile-lab/howlader/GraphPath_baselines/baseline_pnet
    # Flat (legacy):
    python preprocessing/build_reactome_files.py \\
        --pathways /path/to/filtered_pathways.csv \\
        --out      _database/pathways/Reactome
    # Hierarchical (recommended when relations file is available):
    python preprocessing/build_reactome_files.py \\
        --pathways  data/reactome_latest/filtered_pathways_curated.csv \\
        --relations data/reactome_latest/HumanPathwaysRelation.csv \\
        --out       _database/pathways/Reactome

The `--out` directory will be created if it doesn't exist.
"""
import argparse
import os
import re
import sys

import pandas as pd


def _hsa_id(raw_id) -> str:
    """Return a Reactome-style HSA-prefixed pathway ID.

    The downstream Reactome loader filters with
    `hierarchy[hierarchy['child'].str.contains('HSA')]`, so every pathway
    ID we emit MUST contain the substring 'HSA' (case-sensitive).
    """
    s = str(raw_id).strip()
    if 'HSA' in s:
        return s
    # Keep ASCII identifier-safe characters; replace others with '-'.
    safe = re.sub(r'[^A-Za-z0-9_]+', '-', s)
    return 'HSA-{}'.format(safe)


def _parse_gene_list(cell) -> list:
    if pd.isna(cell):
        return []
    toks = re.split(r'[,\s;|]+', str(cell).strip())
    return [t for t in toks if t]


def build(pathways_csv: str, out_dir: str,
          relations_csv: str = None,
          species: str = 'Homo sapiens',
          dummy_target: str = 'HSA-DUMMY-TARGET',
          gmt_description: str = 'converted_from_filtered_pathways') -> None:
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(pathways_csv)
    needed = {'Pathway_ID', 'Pathway_Name', 'Genes'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            'filtered_pathways.csv is missing columns {}. Got: {}'.format(
                missing, list(df.columns))
        )

    # ------------------------------------------------------------------
    # ReactomePathways.txt
    # cols (tab-separated, no header expected by the loader):
    #     reactome_id   pathway_name   species
    # The loader assigns these names regardless of any header in the file,
    # so we omit the header.
    # ------------------------------------------------------------------
    names_rows = []
    gmt_rows = []
    flat_rel_rows = []
    kept_ids = set()
    skipped_empty = 0
    for _, row in df.iterrows():
        rid = _hsa_id(row['Pathway_ID'])
        pname = str(row['Pathway_Name']).strip()
        genes = _parse_gene_list(row['Genes'])
        if not genes:
            skipped_empty += 1
            continue
        names_rows.append((rid, pname, species))
        # GMT layout matching the loader: pathway_col=1, genes_col=3.
        # Column 0 = display name, 1 = id, 2 = description, 3+ = genes
        gmt_rows.append([pname, rid, gmt_description] + genes)
        flat_rel_rows.append((rid, dummy_target))
        kept_ids.add(rid)

    if not gmt_rows:
        raise ValueError('No usable pathways found in {}.'.format(pathways_csv))

    # ------------------------------------------------------------------
    # Build the child→parent relation rows.
    #   - No --relations  : every pathway → DUMMY (flat).
    #   - With --relations: real Reactome parent→child edges, restricted to
    #     pairs where both endpoints are in `kept_ids`. Pathways with no
    #     kept parent stay edge-less here; P-NET's loader will detect them
    #     as in_degree==0 nodes and wire them to its synthetic 'root'.
    # ------------------------------------------------------------------
    hierarchical = relations_csv is not None
    if hierarchical:
        rel_df = pd.read_csv(relations_csv)
        rel_needed = {'Parent', 'Child'}
        rel_missing = rel_needed - set(rel_df.columns)
        if rel_missing:
            raise ValueError(
                'relations file is missing columns {}. Got: {}'.format(
                    rel_missing, list(rel_df.columns))
            )
        rel_rows = []
        seen = set()
        for _, r in rel_df.iterrows():
            parent = _hsa_id(r['Parent'])
            child = _hsa_id(r['Child'])
            if parent in kept_ids and child in kept_ids:
                key = (child, parent)
                if key in seen:
                    continue
                seen.add(key)
                rel_rows.append(key)
        n_orphans = sum(1 for pid in kept_ids
                        if not any(c == pid for c, _ in rel_rows))
    else:
        rel_rows = flat_rel_rows
        n_orphans = len(kept_ids)

    # Write ReactomePathways.txt — TSV without header (loader sets columns).
    names_path = os.path.join(out_dir, 'ReactomePathways.txt')
    pd.DataFrame(names_rows).to_csv(names_path, sep='\t', index=False, header=False)

    # Write ReactomePathways.gmt — TSV without header, variable-width rows.
    gmt_path = os.path.join(out_dir, 'ReactomePathways.gmt')
    with open(gmt_path, 'w') as f:
        for r in gmt_rows:
            f.write('\t'.join(str(c) for c in r) + '\n')

    # Write ReactomePathwaysRelation.txt — TSV WITH header. The loader does
    # `df.columns = ['child', 'parent']` AFTER reading, so the header text
    # doesn't matter for column resolution; pandas just needs *something*
    # to read as the header.
    rel_path = os.path.join(out_dir, 'ReactomePathwaysRelation.txt')
    pd.DataFrame(rel_rows, columns=['child', 'parent']).to_csv(
        rel_path, sep='\t', index=False, header=True)

    print('Wrote {} pathways to:'.format(len(gmt_rows)))
    print('  {}'.format(names_path))
    print('  {}'.format(gmt_path))
    print('  {}'.format(rel_path))
    if hierarchical:
        print('Hierarchical mode: kept {} parent→child edges; {} pathways '
              'have no kept parent (will attach to P-NET root).'.format(
                  len(rel_rows), n_orphans))
    else:
        print('Flat mode: every pathway → {} (single-layer topology).'.format(
            dummy_target))
    if skipped_empty:
        print('Skipped {} pathway rows that had no gene members.'.format(skipped_empty))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pathways', required=True,
                    help='Path to filtered_pathways.csv (Pathway_ID, Pathway_Name, Genes)')
    ap.add_argument('--relations', default=None,
                    help='Optional path to HumanPathwaysRelation.csv '
                         '(Parent, Child). When provided, the real Reactome '
                         'parent→child hierarchy is preserved (restricted to '
                         'pairs where both endpoints are in the kept set). '
                         'When omitted, a flat DUMMY-target topology is emitted.')
    ap.add_argument('--out', required=True,
                    help='Output directory (will be created). Typical: '
                         '_database/pathways/Reactome under the repo root.')
    ap.add_argument('--species', default='Homo sapiens')
    ap.add_argument('--dummy_target', default='HSA-DUMMY-TARGET',
                    help='Synthetic node name used only in flat mode (when '
                         '--relations is not given). Must contain "HSA" to '
                         'survive P-NET\'s species filter.')
    args = ap.parse_args()

    if not os.path.exists(args.pathways):
        sys.exit('Pathways file not found: {}'.format(args.pathways))
    if args.relations and not os.path.exists(args.relations):
        sys.exit('Relations file not found: {}'.format(args.relations))

    build(args.pathways, args.out,
          relations_csv=args.relations,
          species=args.species, dummy_target=args.dummy_target)


if __name__ == '__main__':
    main()
