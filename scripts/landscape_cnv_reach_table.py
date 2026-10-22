"""Export gene-by-CNV reach tables and figures from a saved landscape.

Direct reach uses the individual CNV NPS threshold. Tables include every
landscape gene, per-CNV coverage, and the distribution of CNV reach counts."""
from cnv_overlap.results import resolve_path, report_run, finalize_manifest
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
YES, NO = '#2a6fb0', '#ffffff'


def load(run):
    """Return the per-gene reach table (all landscape genes, most-reached first) and the CNV names."""
    reached = pd.read_csv(resolve_path(run/'02_cnv_in_landscape/propagation_reached_gene_cnv.tsv'), sep='\t')
    landscape = pd.read_csv(resolve_path(run/'01_schema_gwas_landscape/genes.tsv'), sep='\t', usecols=['entrez'])
    if set(reached.entrez) != set(landscape.entrez):
        raise ValueError('Reach table and landscape genes disagree')
    cnvs = [c for c in reached.columns if c not in ('entrez', 'symbol', 'n_cnvs_reached', 'cnvs_reached')]
    reached['n_cnvs_reached'] = reached[cnvs].sum(axis=1)
    order = reached.sort_values(['n_cnvs_reached', 'symbol'], ascending=[False, True])
    return order.reset_index(drop=True), cnvs


def plot(table, cnvs, out, network):
    """Draw the gene x CNV grid: filled cell = the CNV reaches the gene (CNV NPS > 1)."""
    counts = table[cnvs].sum()
    fig, ax = plt.subplots(figsize=(1.2+.55*len(cnvs), max(4, .14*len(table)+2)), constrained_layout=True)
    ax.imshow(table[cnvs].to_numpy(int), aspect='auto', cmap=ListedColormap([NO, YES]),
              vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks(range(len(cnvs)), [f'{c}\n({counts[c]})' for c in cnvs], rotation=45, ha='left', fontsize=8)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(table)), [f'{s} ({k})' for s, k in zip(table.symbol, table.n_cnvs_reached)],
                  fontsize=4.5)
    ax.set_xticks([x-.5 for x in range(1, len(cnvs))], minor=True)
    ax.set_yticks([y-.5 for y in range(1, len(table))], minor=True)
    ax.grid(which='minor', color='#dddddd', linewidth=.3)
    ax.tick_params(which='minor', length=0)
    unreached = int((table.n_cnvs_reached == 0).sum())
    ax.set_title(f'{network}: SCZ landscape genes reached by each CNV\n'
                 f'Reached: CNV NPS > 1 ({len(table)} genes, {unreached} reached by no CNV)', fontsize=9)
    for ext in ('png', 'pdf'):
        fig.savefig(out/f'gene_cnv_reach_matrix.{ext}', dpi=200)
    plt.close(fig)


def export(run, out, *, network):
    """Generate reach artifacts in an explicitly supplied unfinished output directory."""
    out.mkdir(parents=True, exist_ok=True)
    table, cnvs = load(run)
    export = table[['symbol', 'entrez']].assign(**{c: table[c].map({True: 'yes', False: 'no'}) for c in cnvs},
                                                n_cnvs_reached=table.n_cnvs_reached)
    export.to_csv(resolve_path(out/'gene_cnv_reach_table.tsv'), sep='\t', index=False)
    pd.DataFrame([{'cnv': c, 'landscape_genes_reached': int(table[c].sum()),
                   'fraction_of_landscape': table[c].mean()} for c in cnvs]).to_csv(
        resolve_path(out/'cnv_reach_summary.tsv'), sep='\t', index=False)
    table.n_cnvs_reached.value_counts().sort_index().rename_axis('n_cnvs_reached').rename('genes').reset_index().to_csv(
        resolve_path(out/'reach_count_distribution.tsv'), sep='\t', index=False)
    plot(table, cnvs, out, network)
    (out/'SOURCE.txt').write_text(f'Source landscape run: {run}\n'
                                  'Landscape: SCHEMA NPS > 1, GWAS NPS > 1, SCHEMA x GWAS NPS >= 3 (NetColoc)\n'
                                  'Reach: CNV NPS > 1\n')
    return out


def main():
    """Read a landscape run and export the full gene x CNV reach table and figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, help='Report collection root; defaults to the network landscape reports')
    args = parser.parse_args()
    run = args.run.resolve()
    out, identity, reused = report_run(run, 'gene-cnv-reach', output_root=args.out)
    if not reused:
        export(run, out, network=identity['source'])
        finalize_manifest(out/'manifest.json', {'status': 'complete', 'identity': identity})
    print(out)


if __name__ == '__main__':
    main()
