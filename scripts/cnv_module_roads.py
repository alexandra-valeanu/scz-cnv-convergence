"""Which schizophrenia-landscape modules each CNV reaches ("many roads lead to Rome").

Reads a finished landscape run; nothing is propagated or permuted here. The landscape
is already the SCHEMA x GWAS colocalized network, so by default (--reach direct) a CNV
reaches a landscape gene when its own propagation z exceeds the individual threshold
(02_cnv_in_landscape/propagation_reached_gene_cnv.tsv). --reach joint additionally
requires the CNV x SCHEMA and CNV x GWAS product thresholds (cnvs_both; the definition
of 02_cnv_in_landscape/cnv_module_coverage.tsv and the shared core).
Landscape genes without a Leiden module (isolates) are grouped as "No module".

Outputs (in results/figure_review/cnv_module_roads/):
- cnv_module_profile.tsv: per CNV x module, reached genes, share of the CNV's
  reached landscape genes, share of the module, ratio to the module's share of
  the landscape (descriptive, not a test), and genes reached by no other CNV.
- cnv_summary.tsv: per CNV, reached genes, genes private to it, main module.
- gene_modules_reached.tsv: per landscape gene, module and the CNVs reaching it.
- cnv_module_roads.png/.pdf: share heatmap and private/shared bars.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'results/PCNET2.2_brain/schizophrenia_landscape/landscape-ae8b5001c838'
DEFAULT_OUT = ROOT / 'results/figure_review/cnv_module_roads'
NO_MODULE = 'No module'
REACH_LABEL = {'direct': 'CNV propagation reach', 'joint': 'CNV also colocalizes with SCHEMA and GWAS'}


def short_name(process):
    return process.split(' (GO:')[0]


def load(run, reach):
    direct = pd.read_csv(run / '02_cnv_in_landscape/propagation_reached_gene_cnv.tsv', sep='\t')
    modules = pd.read_csv(run / '01_schema_gwas_landscape/gene_modules.tsv', sep='\t',
                          usecols=['entrez', 'leiden_cluster'])
    names = pd.read_csv(run / '01_schema_gwas_landscape/module_names.tsv', sep='\t')
    if reach == 'joint':
        joint = pd.read_csv(run / '01_schema_gwas_landscape/genes.tsv', sep='\t',
                            usecols=['entrez', 'symbol', 'cnvs_both'])
        both = joint.pop('cnvs_both').fillna('').str.split(',')
        reached = joint.assign(**{cnv: both.map(lambda c, cnv=cnv: cnv in c)
                                  for cnv in direct.columns[2:]})
    else:
        reached = direct
    genes = reached.merge(modules, on='entrez', how='left')
    genes['module'] = np.where(genes['leiden_cluster'].isna(), NO_MODULE,
                               'C' + genes['leiden_cluster'].astype('Int64').astype(str))
    labels = {f'C{r.leiden_cluster}': f'C{r.leiden_cluster} {short_name(r.biological_process)}'
              for r in names.itertuples()}
    labels[NO_MODULE] = NO_MODULE
    cnvs = [c for c in reached.columns if c not in ('entrez', 'symbol')]
    return genes, cnvs, labels


def profile(genes, cnvs):
    hits = genes[cnvs].astype(bool)
    n_reaching = hits.sum(axis=1)
    module_size = genes['module'].value_counts()
    n_landscape = len(genes)
    rows = []
    for cnv in cnvs:
        mine = hits[cnv]
        total = int(mine.sum())
        for module, size in module_size.items():
            in_module = genes['module'] == module
            n = int((mine & in_module).sum())
            share = n / total if total else np.nan
            rows.append(dict(cnv=cnv, module=module, module_genes=int(size), reached_genes=n,
                             share_of_cnv_reach=share, share_of_module=n / size,
                             ratio_to_module_share=share / (size / n_landscape),
                             private_genes=int((mine & in_module & (n_reaching == 1)).sum())))
    return pd.DataFrame(rows), n_reaching


def summarise(table, genes, cnvs, n_reaching):
    rows = []
    for cnv, sub in table.groupby('cnv', sort=False):
        main = sub.sort_values('reached_genes', ascending=False).iloc[0]
        private = genes.loc[genes[cnv].astype(bool) & (n_reaching == 1), 'symbol']
        rows.append(dict(cnv=cnv, reached_landscape_genes=int(sub['reached_genes'].sum()),
                         private_genes=int(sub['private_genes'].sum()),
                         main_module=main['module'], main_module_share=main['share_of_cnv_reach'],
                         modules_reached=int((sub['reached_genes'] > 0).sum()),
                         private_gene_symbols=','.join(sorted(private))))
    out = pd.DataFrame(rows)
    out['private_fraction'] = out['private_genes'] / out['reached_landscape_genes']
    return out.sort_values('reached_landscape_genes', ascending=False)


def plot(table, summary, labels, out, reach):
    cnv_order = list(summary['cnv'])
    module_order = [m for m in labels if m in set(table['module'])]
    share = table.pivot(index='cnv', columns='module', values='share_of_cnv_reach').loc[cnv_order, module_order]
    count = table.pivot(index='cnv', columns='module', values='reached_genes').loc[cnv_order, module_order]
    size = table.drop_duplicates('module').set_index('module')['module_genes']

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.2), width_ratios=[3, 1.3],
                                 gridspec_kw=dict(wspace=0.35))
    im = ax.imshow(share.values * 100, cmap='Blues', vmin=0, vmax=100, aspect='auto')
    for i, j in np.ndindex(share.shape):
        value = share.values[i, j] * 100
        ax.text(j, i, f'{value:.0f}%\n({count.values[i, j]})', ha='center', va='center',
                fontsize=8, color='white' if value > 55 else 'black')
    ax.set_xticks(range(len(module_order)),
                  [f'{labels[m]}\n{size[m]} genes' for m in module_order],
                  rotation=35, ha='right', fontsize=8)
    ax.set_yticks(range(len(cnv_order)), cnv_order)
    ax.set_title(f'Where each CNV lands in the schizophrenia landscape ({REACH_LABEL[reach]})\n'
                 '% of the CNV\'s reached landscape genes in each module (gene count)', fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='% of CNV\'s reached genes')

    y = np.arange(len(cnv_order))
    shared = summary['reached_landscape_genes'] - summary['private_genes']
    bx.barh(y, summary['private_genes'], color='#d95f02', label='Reached by this CNV only')
    bx.barh(y, shared, left=summary['private_genes'], color='#bdbdbd',
            label='Also reached by another CNV')
    bx.set_yticks(y, [])
    bx.set_ylim(len(y) - 0.5, -0.5)
    bx.set_xlabel('Landscape genes reached')
    bx.set_title('Own road or shared?', fontsize=10)
    bx.legend(fontsize=8, loc='lower right', frameon=False)
    bx.spines[['top', 'right']].set_visible(False)

    fig.savefig(out / 'cnv_module_roads.png', dpi=200, bbox_inches='tight')
    fig.savefig(out / 'cnv_module_roads.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--reach', choices=('direct', 'joint'), default='direct')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    genes, cnvs, labels = load(args.run, args.reach)
    table, n_reaching = profile(genes, cnvs)
    table['module_name'] = table['module'].map(labels)
    summary = summarise(table, genes, cnvs, n_reaching)

    reached_by = genes[cnvs].astype(bool).apply(lambda r: ','.join(r.index[r]), axis=1)
    gene_table = genes[['entrez', 'symbol', 'module']].assign(
        module_name=genes['module'].map(labels), n_cnvs=n_reaching, cnvs=reached_by)

    table.to_csv(args.out / 'cnv_module_profile.tsv', sep='\t', index=False)
    summary.to_csv(args.out / 'cnv_summary.tsv', sep='\t', index=False)
    gene_table.sort_values(['module', 'n_cnvs'], ascending=[True, False]).to_csv(
        args.out / 'gene_modules_reached.tsv', sep='\t', index=False)
    (args.out / 'SOURCE.txt').write_text(f'Source landscape run: {args.run.relative_to(ROOT)}\n'
                                       f'Reach definition: {args.reach}\n')
    plot(table, summary, labels, args.out, args.reach)


if __name__ == '__main__':
    main()
