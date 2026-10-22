"""Separate annotation panels, biological-process module names, and orsum summaries."""
from cnv_overlap.results import resolve_path
from pathlib import Path
import re
from importlib.metadata import version
import subprocess

import numpy as np
import pandas as pd

from cnv_overlap.enrichment import load_gmt
from brain_pair_enrichment import collapse_terms
from shared_network_orsum import find_orsum
from cnv_overlap.provenance import file_hash


def analysis_tools():
    """Return plotting/clustering versions and the installed orsum version and source hashes."""
    binary = find_orsum()
    return {'versions': {name: version(name) for name in ('matplotlib', 'igraph', 'leidenalg')},
            'orsum_version': subprocess.check_output([str(binary), '--version'], text=True).strip(),
            'orsum_code': {n: file_hash(binary.parent/n)
                           for n in ('orsum.py', 'termCombinationLib.py', 'plotFunctions.py')}}


def cluster_colors(clusters):
    """Distinct colors for every module; gray is reserved for unassigned genes."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    clusters = sorted(set(clusters))
    cmap = plt.get_cmap('tab20' if len(clusters) <= 20 else 'turbo')
    slots = list(range(0, 20, 2)) + list(range(1, 20, 2))
    return {c: to_hex(cmap(slots[i] if len(clusters) <= 20 else (i+.5)/len(clusters)))
            for i, c in enumerate(clusters)}


def module_names(genes, results, fdr=.05):
    """Label each module by its lowest-q significant GO biological-process term."""
    rows = []
    for cluster, group in genes.groupby('leiden_cluster'):
        hits = results[(results.cluster.astype(str) == f'C{cluster}')
                       & (results.library == 'GO Biological Process') & (results.q_group < fdr)]
        hits = hits.sort_values(['q_group', 'p', 'term'])
        rows.append({'leiden_cluster': int(cluster), 'n_genes': len(group),
                     'biological_process': hits.iloc[0].term if len(hits) else 'No enriched biological process',
                     'bp_q': hits.iloc[0].q_group if len(hits) else np.nan})
    return pd.DataFrame(rows, columns=['leiden_cluster', 'n_genes', 'biological_process', 'bp_q'])


def panel(results, library, queries, out, fdr, *, all_terms=False):
    """Plot ranked terms for one annotation library and write PDF and PNG panels."""
    import matplotlib.pyplot as plt
    import textwrap
    data = results[results.library == library]
    chosen = data if all_terms else data[data.q_group < fdr]
    terms = chosen.groupby('term').q_group.min().sort_values().head(20).index
    fig, ax = plt.subplots(figsize=(max(7, .6*len(terms)+4), max(4, .35*len(queries)+3)), layout='constrained')
    if len(terms):
        q = data.pivot(index='cluster', columns='term', values='q_group').reindex(index=queries, columns=terms)
        score = -np.log10(q.clip(lower=1e-300))
        image = ax.imshow(np.ma.masked_invalid(score), aspect='auto', cmap='Blues', vmin=0)
        ax.set_xticks(range(len(terms)), [textwrap.fill(t, 28) for t in terms], rotation=55, ha='right', fontsize=8)
        ax.set_yticks(range(len(queries)), queries, fontsize=8)
        for (i, j), value in np.ndenumerate(q.to_numpy()):
            if value < fdr:
                ax.text(j, i, '*', ha='center', va='center')
        fig.colorbar(image, ax=ax, label='−log10 BH q; * passes FDR')
    else:
        ax.text(.5, .5, 'No terms pass FDR', ha='center', transform=ax.transAxes)
        ax.axis('off')
    ax.set_title(library + '\nDescriptive enrichment after selection')
    stem = re.sub(r'[^a-z0-9]+', '_', library.lower()).strip('_')
    for ext in ('pdf', 'png'):
        fig.savefig(Path(out)/f'{stem}.{ext}', dpi=180)
    plt.close(fig)


def summarize_enrichment(results, paths, out, fdr=.05):
    out = Path(out)
    queries = results.cluster.drop_duplicates().tolist()
    for library in ('GO Biological Process', 'SynGO 2024', 'MitoCarta 3.0', 'HPA broad brain cell type'):
        panel(results, library, queries, out, fdr, all_terms=library == 'HPA broad brain cell type')
    libraries = {name: load_gmt(path) for name, path in paths.items() if str(path).endswith('.gmt')}
    collapsed = collapse_terms(results.rename(columns={'cluster': 'locus', 'q_group': 'q_all_pair_terms'}),
                              libraries, out/'orsum', binary=find_orsum(), fdr=fdr, min_size=5)
    collapsed.to_csv(resolve_path(out/'orsum_mapping.tsv'), sep='\t', index=False)
    (out/'orsum_status.txt').write_text('Completed; original p/q retained. Libraries with no significant terms require no orsum invocation.\n')
