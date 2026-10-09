"""Exploratory notebook analyses; own code hash preserves the core cache identity."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cnv_overlap.propagation import Propagator
from cnv_overlap.provenance import atomic_path, digest, file_hash, run_dir, write_json
from cnv_overlap.statistics import bh, rng_for

SCZ_TARGETS = ('schema', 'gwaspri')


def recurrence_statistics(reached, target_masks):
    counts = np.asarray(reached, dtype=bool).sum(axis=0)
    # Each gene contributes the number of CNV pairs reaching that same gene.
    weights = counts * (counts - 1) // 2
    values = {'global_convergence': int(weights.sum())}
    values.update({name: int(weights[mask].sum()) for name, mask in target_masks.items()})
    return counts, values


def consecutive_windows(coords, nodes, chrom, size):
    """Gene-count-matched windows on the CNV's chromosome, in network coordinates."""
    c = coords[coords.entrez.isin(nodes)].copy()
    c['chrom'] = c.chrom.astype(str).str.removeprefix('chr')
    genes = c[c.chrom == str(chrom).removeprefix('chr')].sort_values(['start', 'end', 'entrez']).entrez.to_numpy()
    if len(genes) < size:
        raise ValueError(f'Chromosome {chrom} has fewer mapped network genes than the seed count')
    return genes


def run_convergence(inputs, *, config, cache_dir, output_root, reps=1000,
                    null_model='degree', scz_entities=None, progress=print):
    """Joint seed randomizations; no selection of genes by pair significance.

    Gene p_maxT is a maximum-count tail under the complete joint seed null.
    It is not a claim of strong FWER control under arbitrary partial alternatives.
    Region windows preserve chromosome and seed count, not degree or physical span.
    Every new degree profile receives its own propagation reference distribution.
    """
    if null_model not in ('degree', 'region') or reps < 2:
        raise ValueError('Choose degree or region and at least two draws')
    A, nodes, symbols, entities, coords, regions = inputs
    prop = Propagator(A, nodes, cache_dir, config.propagation)
    names = sorted(n for n, e in entities.items() if e['kind'] == 'cnv')
    if len(names) < 2:
        raise ValueError('Convergence requires at least two CNVs')
    seeds = {n: sorted(set(entities[n]['seeds']) & prop.pos.keys()) for n in names}
    scz = {n: sorted(set(e['seeds']) & prop.pos.keys()) for n, e in sorted((scz_entities or {}).items())
           if n in SCZ_TARGETS}
    identity = {'algorithm': 'joint_recurrence_v1', 'code': file_hash(__file__),
                'propagation': prop.metadata, 'config': asdict(config), 'seeds': seeds,
                'raw_entities': entities, 'scz': scz, 'raw_scz': scz_entities or {},
                'coordinates': coords.to_json(orient='records'), 'regions': regions.to_json(orient='records'),
                'symbols': {str(k): v for k, v in symbols.items()}, 'reps': reps, 'null_model': null_model}
    out = run_dir(output_root, identity, 'joint-recurrence', null_model, f'reps{reps}')
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'manifest.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old.get('status') == 'complete':
            if any(not (out / f).exists() or file_hash(out / f) != h for f, h in old['artifacts'].items()):
                raise ValueError('Convergence artifacts changed')
            return out
    write_json(manifest, {'status': 'running', 'identity': identity})
    threshold = config.thresholds.individual
    reached = np.stack([np.asarray(prop.nps(seeds[n])) > threshold for n in names])
    targets = {n: np.asarray(prop.nps(s)) > threshold for n, s in scz.items()}
    observed_counts, observed_stats = recurrence_statistics(reached, targets)
    signatures = {n: prop.signature(prop.seed_indices(seeds[n])) for n in names}
    windows = {}
    if null_model == 'region':
        r = regions.set_index('entity')
        windows = {n: consecutive_windows(coords, prop.nodes, r.loc[n, 'chrom'], len(seeds[n])) for n in names}
    # Fixed reference profiles are reused for degree-matched draws; do not cache
    # thousands of one-use random NPS vectors. Region draws compute their own refs.
    def random_nps(draw):
        cols = prop.seed_indices(draw)
        mu, sd, _ = prop.reference(cols)
        return (np.log(prop.heat(draw)) - mu) / sd
    counts_path = out / 'null_counts.npy'
    # Checkpoint blocks allow long production calculations to resume.
    blocks = []
    for start in range(0, reps, 25):
        stop = min(start + 25, reps)
        path = out / f'draws-{start:06d}.npz'
        stamp = Path(str(path) + '.json')
        if path.exists() and stamp.exists() and json.loads(stamp.read_text())['sha256'] == file_hash(path):
            blocks.append(path)
            continue
        block_counts, block_stats = [], []
        for rep in range(start, stop):
            rng = rng_for(config.propagation.seed, 'joint_convergence', null_model, rep)
            random_reached = []
            for n in names:
                if null_model == 'degree':
                    draw = prop.nodes[np.concatenate([rng.choice(group, size=k, replace=False)
                        for group, k in zip(prop.bins, signatures[n]) if k])]
                else:
                    genes = windows[n]
                    j = rng.integers(len(genes) - len(seeds[n]) + 1)
                    draw = genes[j:j + len(seeds[n])]
                random_reached.append(random_nps(draw) > threshold)
            counts, stats = recurrence_statistics(random_reached, targets)
            block_counts.append(counts)
            block_stats.append([stats[k] for k in observed_stats])
        with atomic_path(path) as tmp:
            np.savez_compressed(tmp, counts=np.asarray(block_counts, dtype=np.uint16), stats=block_stats)
        write_json(stamp, {'sha256': file_hash(path)})
        blocks.append(path)
        progress(f'{null_model} convergence: {stop}/{reps} joint draws')
    null_counts, null_stats = [], []
    for path in blocks:
        with np.load(path) as data:
            null_counts.append(data['counts'])
            null_stats.append(data['stats'])
    null_counts = np.concatenate(null_counts)
    null_stats = np.concatenate(null_stats)
    with atomic_path(counts_path) as tmp:
        np.save(tmp, null_counts)
    max_counts = null_counts.max(axis=1)
    genes = pd.DataFrame({'entrez': prop.nodes, 'symbol': [symbols.get(int(g), str(g)) for g in prop.nodes],
        'n_cnvs': observed_counts, 'null_mean_n_cnvs': null_counts.mean(axis=0),
        'p_empirical': (1 + (null_counts >= observed_counts).sum(axis=0)) / (reps + 1),
        'p_maxT': (1 + (max_counts[:, None] >= observed_counts).sum(axis=0)) / (reps + 1),
        'reaching_cnvs': [','.join(np.asarray(names)[reached[:, j]]) for j in range(len(prop.nodes))]})
    for name, mask in targets.items():
        genes[f'reached_by_{name}'] = mask
    genes.to_csv(out / 'gene_convergence.tsv', sep='\t', index=False)
    rows = []
    for j, (name, value) in enumerate(observed_stats.items()):
        null = null_stats[:, j]
        rows.append({'endpoint': name, 'observed': value, 'null_mean': null.mean(),
            'excess': value - null.mean(), 'p_empirical': (1 + (null >= value).sum()) / (reps + 1),
            'null_low': np.quantile(null, .025), 'null_high': np.quantile(null, .975),
            'reps': reps, 'p_resolution': 1 / (reps + 1)})
    tests = pd.DataFrame(rows)
    tests['q_scz'] = np.nan
    mask = tests.endpoint.isin(SCZ_TARGETS)
    tests.loc[mask, 'q_scz'] = bh(tests.loc[mask, 'p_empirical'])
    tests.to_csv(out / 'convergence_tests.tsv', sep='\t', index=False)
    loo = []
    for i, name in enumerate(names):
        counts, stats = recurrence_statistics(np.delete(reached, i, axis=0), targets)
        loo.append({'omitted_cnv': name, 'genes_reached_by_at_least_two': int((counts >= 2).sum()), **stats})
    pd.DataFrame(loo).to_csv(out / 'leave_one_out.tsv', sep='\t', index=False)
    pd.DataFrame({'entity': names, 'raw_seeds': [len(set(entities[n]['seeds'])) for n in names],
                  'retained_seeds': [len(seeds[n]) for n in names],
                  'retained_with_coordinates': [len(set(seeds[n]) & set(coords.entrez)) for n in names]}
                 ).to_csv(out / 'seed_qc.tsv', sep='\t', index=False)
    write_json(manifest, {'status': 'complete', 'identity': identity,
        'interpretation': 'Exploratory joint seed null; maxT calibrated under complete null; LOO descriptive; fixed SCZ profiles',
        'endpoint_order': list(observed_stats),
        'artifacts': {p.name: file_hash(p) for p in out.iterdir() if p.is_file() and p != manifest}})
    return out


def disorder_gene_partition(candidates, primary):
    """All pairs, irrespective of p/q. Categories describe threshold membership."""
    scz = candidates[candidates.b.isin(SCZ_TARGETS)].groupby('a').entrez.agg(set).to_dict()
    secondary = candidates[~candidates.b.isin(SCZ_TARGETS)].copy()
    secondary['scz_shared'] = [g in scz.get(a, set()) for a, g in zip(secondary.a, secondary.entrez)]
    secondary['category'] = np.where(secondary.scz_shared, 'shared_with_scz', 'secondary_only_at_threshold')
    summary = secondary.groupby(['a', 'b', 'category']).size().unstack(fill_value=0)
    pairs = primary.loc[~primary.b.isin(SCZ_TARGETS), ['a', 'b']].drop_duplicates()
    summary = pairs.merge(summary.reset_index(), on=['a', 'b'], how='left').fillna(0)
    for col in ('shared_with_scz', 'secondary_only_at_threshold'):
        if col not in summary:
            summary[col] = 0
        summary[col] = summary[col].astype(int)
    return secondary, summary


def export_versioned_report(run_dir_path, inputs, *, output_dir, gmt_paths=(), fdr=.05):
    """Select a report revision by content; preserve existing report directories."""
    from cnv_overlap.reporting import export_report
    gmt_paths = tuple(Path(p) for p in gmt_paths)
    identity = {'run_manifest_sha256': file_hash(Path(run_dir_path)/'manifest.json'),
                'fdr': fdr, 'gmt': {str(p): file_hash(p) for p in gmt_paths}}
    return export_report(run_dir_path, inputs, output_dir=run_dir(output_dir, identity, 'report', f'fdr{fdr}'),
                         gmt_paths=gmt_paths, fdr=fdr)


def scz_overlap_membership(nodes, entities, profiles, config, min_cnvs=3):
    """Count CNVs satisfying BOTH CNV–target pair masks at each identical gene."""
    from cnv_overlap.statistics import hit_mask
    if not isinstance(min_cnvs, int) or isinstance(min_cnvs, bool) or min_cnvs < 1:
        raise ValueError('min_cnvs must be a positive integer')
    names = sorted(n for n, e in entities.items() if e['kind'] == 'cnv')
    targets = [n for n in SCZ_TARGETS if n in entities and entities[n]['kind'] == 'target']
    if not names or not targets:
        raise ValueError('Need CNVs and at least one SCHEMA/GWAS target')
    pairs = {}
    for name in names:
        for target in targets:
            mask = hit_mask(profiles[name], profiles[target], config.thresholds)
            if config.overlap == 'remove':
                mask &= ~np.isin(nodes, list(set(entities[name]['seeds']) & set(entities[target]['seeds'])))
            pairs[name, target] = mask
    table = pd.DataFrame({'entrez': nodes})
    for target in targets:
        masks = np.stack([pairs[n, target] for n in names])
        table[f'n_cnvs_{target}'] = masks.sum(axis=0)
        table[f'cnvs_{target}'] = [','.join(np.asarray(names)[masks[:, j]]) for j in range(len(nodes))]
    both = {}
    if len(targets) == 2:
        both = {n: pairs[n, 'schema'] & pairs[n, 'gwaspri'] for n in names}
        masks = np.stack(list(both.values()))
        table['n_cnvs_both'] = masks.sum(axis=0)
        table['cnvs_both'] = [','.join(np.asarray(names)[masks[:, j]]) for j in range(len(nodes))]
        table['shared_core'] = table.n_cnvs_both >= min_cnvs
    return table, pairs, both


def export_scz_networks(inputs, *, config, cache_dir, output_root, min_cnvs=3, progress=print):
    """All pair networks and same-gene intersections, without significance filtering."""
    import matplotlib.pyplot as plt
    import networkx as nx
    from cnv_overlap.interpretation import candidate_network
    from cnv_overlap.reporting import write_graphml

    A, nodes, symbols, entities, *_ = inputs
    entities = {n: e for n, e in entities.items()
                if e['kind'] == 'cnv' or (e['kind'] == 'target' and n in SCZ_TARGETS)}
    prop = Propagator(A, nodes, cache_dir, config.propagation)
    identity = {'algorithm': 'scz_pair_intersections_v1', 'code': file_hash(__file__),
                'propagation': prop.metadata, 'thresholds': asdict(config.thresholds),
                'overlap': config.overlap, 'entities': entities, 'min_cnvs': min_cnvs,
                'symbols': {str(k): v for k, v in symbols.items()}}
    out = run_dir(output_root, identity, 'scz-pair-intersections', f'min{min_cnvs}cnvs')
    manifest = out/'manifest.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous.get('status') == 'complete':
            for name, checksum in previous['artifacts'].items():
                if not (out/name).exists() or file_hash(out/name) != checksum:
                    raise ValueError('SCZ network artifacts changed')
            return out
    retained = {n: sorted(set(e['seeds']) & prop.pos.keys()) for n, e in entities.items()}
    profiles = {n: np.asarray(prop.nps(retained[n])) for n in entities}
    table, pairs, both = scz_overlap_membership(nodes, entities, profiles, config, min_cnvs)
    table.insert(1, 'symbol', [symbols.get(int(g), str(g)) for g in nodes])
    for name in sorted(entities):
        table[f'nps_{name}'] = profiles[name]
        table[f'seed_{name}'] = np.isin(nodes, entities[name]['seeds'])
    cnv_names = sorted(n for n, e in entities.items() if e['kind'] == 'cnv')
    table['seed_cnvs'] = [','.join(n for n in cnv_names if row[f'seed_{n}']) for row in table.to_dict('records')]
    table['is_cnv_seed'] = table.seed_cnvs.ne('')
    table['is_any_seed'] = table[[f'seed_{n}' for n in entities]].any(axis=1)
    table['gene_origin'] = np.where(table.is_any_seed, 'input_seed', 'propagated_only')
    out.mkdir(parents=True, exist_ok=True)
    write_json(manifest, {'status': 'running', 'identity': identity})
    table.to_csv(out/'gene_membership.tsv', sep='\t', index=False)
    pd.DataFrame([{'entity': n, 'raw_seeds': len(set(e['seeds'])),
                   'retained_seeds': len(retained[n]),
                   'dropped_seeds': ','.join(map(str, sorted(set(e['seeds']) - set(retained[n]))))}
                  for n, e in entities.items()]).to_csv(out/'seed_qc.tsv', sep='\t', index=False)
    selections = [('pair', a, b, mask) for (a, b), mask in pairs.items()]
    selections += [('cnv_both', n, 'schema+gwaspri', mask) for n, mask in both.items()]
    if both:
        selections.append(('shared_core', f'at_least_{min_cnvs}_cnvs', 'schema+gwaspri', table.shared_core.to_numpy()))
    rows = []
    for kind, cnv, target, mask in selections:
        key = digest([kind, cnv, target])[:12]
        directory = out/'networks'/key
        directory.mkdir(parents=True, exist_ok=True)
        query = table.loc[mask].copy()
        graph, membership = candidate_network(A, nodes, query)
        query = query.merge(membership.drop(columns='interpretation'), on='entrez', how='left')
        for record in query.to_dict('records'):
            gene = record.pop('entrez')
            graph.nodes[gene].update(record)
        write_graphml(graph, directory/'network.graphml')
        query.to_csv(directory/'genes.tsv', sep='\t', index=False)
        pd.DataFrame(graph.edges(), columns=['source', 'target']).to_csv(directory/'edges.tsv', sep='\t', index=False)
        sizes = [len(c) for c in nx.connected_components(graph)]
        rows.append({'kind': kind, 'cnv': cnv, 'target': target, 'nodes': len(graph),
                     'edges': graph.number_of_edges(), 'components': len(sizes),
                     'largest_component': max(sizes, default=0), 'isolates': nx.number_of_isolates(graph),
                     'directory': str(directory.relative_to(out))})
        fig, ax = plt.subplots(figsize=(16, 12) if kind == 'shared_core' else (10, 7))
        if not graph:
            ax.text(.5, .5, 'No genes meet these overlap criteria', ha='center', transform=ax.transAxes)
        else:
            display_graph = graph
            labelled = list(graph) if kind == 'shared_core' else sorted(graph, key=lambda g: (-graph.degree(g), g))[:8]
            labels = {g: graph.nodes[g]['symbol'] for g in labelled}
            node_sizes = 50
            if len(graph) > 150 and kind != 'shared_core':
                groups = query.groupby('community').entrez.agg(set).tolist()
                display_graph = nx.Graph()
                mapping = {g: i for i, group in enumerate(groups) for g in group}
                display_graph.add_nodes_from(range(len(groups)))
                display_graph.add_edges_from((mapping[u], mapping[v]) for u, v in graph.edges() if mapping[u] != mapping[v])
                labels = {i: f'C{i+1}: {len(group)} genes' for i, group in enumerate(groups)}
                node_sizes = [80 + 20*np.sqrt(len(group)) for group in groups]
            isolates = sorted(nx.isolates(display_graph))
            connected = display_graph.subgraph([g for g in display_graph if g not in set(isolates)])
            pos = nx.spring_layout(connected, seed=42, k=.6, iterations=100) if connected else {}
            pos.update({g: (x, -1.4) for g, x in zip(isolates, np.linspace(-1, 1, len(isolates)))})
            if kind == 'shared_core':
                colors = draw_shared_cnv_pies(ax, graph, pos, cnv_names)
                from matplotlib.colors import to_hex
                pd.DataFrame([{'cnv': n, 'color': to_hex(c)} for n, c in colors.items()]).to_csv(
                    directory/'cnv_colors.tsv', sep='\t', index=False)
            else:
                from matplotlib.collections import LineCollection
                ax.add_collection(LineCollection([(pos[a], pos[b]) for a, b in display_graph.edges()],
                                                 colors='#aaaaaa', linewidths=.4))
                nx.draw_networkx_nodes(display_graph, pos, ax=ax, node_size=node_sizes, node_color='#58a6a6')
                nx.draw_networkx_labels(display_graph, pos, ax=ax, labels=labels, font_size=7)
        title_cnv = f'At least {min_cnvs} CNVs' if kind == 'shared_core' else cnv
        title_target = target.replace('schema', 'SCHEMA').replace('gwaspri', 'GWAS').replace('+', ' + ')
        ax.set_title(f'{title_cnv} × {title_target}\n{len(graph)} genes; {graph.number_of_edges()} edges; descriptive overlap'
                     + ('\nCommunity summary (full gene graph exported)' if len(graph) > 150 and kind != 'shared_core' else ''))
        ax.axis('off')
        for suffix in ('png', 'pdf'):
            fig.savefig(directory/f'network.{suffix}', dpi=160, bbox_inches='tight')
        plt.close(fig)
        progress(f'{cnv} × {target}: {len(graph)} genes, {graph.number_of_edges()} edges')
    pd.DataFrame(rows).to_csv(out/'network_index.tsv', sep='\t', index=False)
    write_json(manifest, {'status': 'complete', 'identity': identity,
        'joint_status': 'available' if both else 'unavailable: requires schema and gwaspri',
        'interpretation': 'Descriptive induced networks; same CNVs satisfy both pair masks; no gene/module p-values; no genomic colocalization claim',
        'artifacts': {str(p.relative_to(out)): file_hash(p) for p in out.rglob('*') if p.is_file() and p != manifest}})
    return out


def draw_shared_cnv_pies(ax, graph, pos, cnv_names):
    """Reference-inspired pies: convergence membership inside, input seed rings outside."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Patch, Wedge
    from matplotlib.lines import Line2D
    import networkx as nx

    colors = {n: plt.get_cmap('tab10')(i % 10) for i, n in enumerate(cnv_names)}
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection([(pos[a], pos[b]) for a, b in graph.edges()],
                                     colors='#9aa1aa', linewidths=.6, alpha=.45))
    radius = .026
    # Spread labels as in the reference plot, retaining leader lines to each gene.
    genes = list(graph)
    label_positions = np.array([pos[g] for g in genes], dtype=float) + [0, .07]
    for _ in range(100):
        moved = False
        for i in range(len(genes)):
            for j in range(i + 1, len(genes)):
                delta = label_positions[i] - label_positions[j]
                distance = np.linalg.norm(delta / [.19, .075])
                if distance < 1:
                    direction = delta / max(np.linalg.norm(delta), 1e-9) if np.any(delta) else np.array([0., 1.])
                    label_positions[i] += direction * .025
                    label_positions[j] -= direction * .025
                    moved = True
        if not moved:
            break
    label_pos = dict(zip(genes, label_positions))
    for gene, attrs in graph.nodes(data=True):
        center = np.asarray(pos[gene])
        members = [n for n in cnv_names if n in attrs['cnvs_both'].split(',')]
        for i, name in enumerate(members):
            ax.add_patch(Wedge(center, radius, 90 + 360*i/len(members), 90 + 360*(i+1)/len(members),
                               facecolor=colors[name], edgecolor='white', linewidth=.5, zorder=3))
        ax.add_patch(Circle(center, radius, facecolor='none', edgecolor='#777777', linewidth=.6, zorder=4))
        ring_radius = radius
        for flag, color in [('is_cnv_seed', '#111111'), ('seed_schema', '#d62728'), ('seed_gwaspri', '#7b2cbf')]:
            if attrs.get(flag, False):
                ring_radius += .004
                ax.add_patch(Circle(center, ring_radius, facecolor='none', edgecolor=color, linewidth=2, zorder=5))
        label = attrs['symbol']
        if attrs.get('is_cnv_seed', False):
            label += '\nseed: ' + attrs['seed_cnvs']
        ax.annotate(label, center, xytext=label_pos[gene],
                    arrowprops=dict(arrowstyle='-', color='#888888', linewidth=.5, shrinkB=8),
                    ha='center', va='bottom', fontsize=8, zorder=6,
                    fontweight='bold' if attrs.get('is_any_seed', False) else 'normal',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.75, pad=.3))
    cnv_legend = ax.legend(handles=[Patch(facecolor=colors[n], label=n) for n in cnv_names],
        title='Pie slices: CNVs meeting BOTH target masks\nEqual slices indicate membership, not effect size',
        loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9, title_fontsize=10)
    ax.add_artist(cnv_legend)
    handles = [Line2D([], [], marker='o', linestyle='', markerfacecolor='white',
                      markeredgecolor=color, markeredgewidth=width, markersize=12, label=label)
               for color, width, label in [('#777777', .6, 'Propagated only: no input seed membership'),
                   ('#111111', 2, 'CNV seed (CNV named below gene)'),
                   ('#d62728', 2, 'SCHEMA seed'), ('#7b2cbf', 2, 'GWAS-prioritized seed')]]
    ax.legend(handles=handles, title='Seed status: rings can combine\nBold label = any input seed',
              loc='upper left', bbox_to_anchor=(1.01, .52), frameon=False, fontsize=9, title_fontsize=10)
    ax.set_aspect('equal')
    ax.margins(.12)
    return colors
