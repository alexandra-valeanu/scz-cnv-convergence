"""Summarize CNV reach across modules of a saved schizophrenia landscape.

Direct reach uses the individual CNV NPS threshold. Joint reach also requires
both CNV–target product thresholds. Exports module profiles, private/shared
gene counts, and figures; isolates are grouped as "No module"."""
from cnv_overlap.results import resolve_path, report_run, finalize_manifest
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / 'results/PCNET2.2_brain/4_scz_landscape/landscape-ae8b5001c838'
NO_MODULE = 'No module'
REACH_LABEL = {'direct': 'CNV propagation reach', 'joint': 'CNV also colocalizes with SCHEMA and GWAS'}


def short_name(process):
    """Remove the trailing GO identifier from a biological-process label."""
    return process.split(' (GO:')[0]


def load(run, reach):
    """Return landscape genes, CNV columns and module labels for direct or joint reach."""
    direct = pd.read_csv(resolve_path(run / '02_cnv_in_landscape/propagation_reached_gene_cnv.tsv'), sep='\t')
    modules = pd.read_csv(resolve_path(run / '01_schema_gwas_landscape/gene_modules.tsv'), sep='\t',
                          usecols=['entrez', 'leiden_cluster'])
    names = pd.read_csv(resolve_path(run / '01_schema_gwas_landscape/module_names.tsv'), sep='\t')
    if reach == 'joint':
        joint = pd.read_csv(resolve_path(run / '01_schema_gwas_landscape/genes.tsv'), sep='\t',
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
    """Return per-CNV module coverage and per-gene CNV counts; ratios are descriptive."""
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
    """Summarize reached and private genes and the most-reached module for each CNV."""
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
    """Write module-share and private/shared-gene figures for the selected reach definition."""
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

    fig.savefig(resolve_path(out / 'cnv_module_roads.png'), dpi=200, bbox_inches='tight')
    fig.savefig(resolve_path(out / 'cnv_module_roads.pdf'), bbox_inches='tight')
    plt.close(fig)


def export(run, out, *, reach="direct"):
    """Generate module-road artifacts in an explicit unfinished output directory."""
    out.mkdir(parents=True, exist_ok=True)
    genes, cnvs, labels = load(run, reach)
    table, n_reaching = profile(genes, cnvs)
    table['module_name'] = table['module'].map(labels)
    summary = summarise(table, genes, cnvs, n_reaching)

    reached_by = genes[cnvs].astype(bool).apply(lambda r: ','.join(r.index[r]), axis=1)
    gene_table = genes[['entrez', 'symbol', 'module']].assign(
        module_name=genes['module'].map(labels), n_cnvs=n_reaching, cnvs=reached_by)

    table.to_csv(resolve_path(out / 'cnv_module_profile.tsv'), sep='\t', index=False)
    summary.to_csv(resolve_path(out / 'cnv_summary.tsv'), sep='\t', index=False)
    gene_table.sort_values(['module', 'n_cnvs'], ascending=[True, False]).to_csv(
        resolve_path(out / 'gene_modules_reached.tsv'), sep='\t', index=False)
    (out / 'SOURCE.txt').write_text(f'Source landscape run: {run}\n'
                                       f'Reach definition: {reach}\n')
    plot(table, summary, labels, out, reach)

    return out

def main():
    """Read a landscape run and export descriptive module coverage tables and figures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, help='Report collection root; defaults to the network landscape reports')
    parser.add_argument('--reach', choices=('direct', 'joint'), default='direct')
    args = parser.parse_args()
    out, identity, reused = report_run(args.run, 'cnv-module-roads',
        settings={'reach': args.reach}, output_root=args.out)
    if not reused:
        export(args.run, out, reach=args.reach)
        finalize_manifest(out/'manifest.json', {'status': 'complete', 'identity': identity})
    print(out)


if __name__ == '__main__':
    main()
