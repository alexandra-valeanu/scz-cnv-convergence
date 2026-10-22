"""Run upstream orsum on ranked significant terms, separately by annotation source."""
from cnv_overlap.results import finalize_manifest, resolve_path
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pandas as pd

from cnv_overlap.enrichment import load_gmt
from cnv_overlap.provenance import file_hash, run_dir, write_json
from shared_network_enrichment import plot_enrichment, plot_gene_term_heatmap

SOURCES = ('Reactome pathways', 'GO Cellular Component', 'GO Biological Process')


def find_orsum(executable='orsum.py'):
    """Resolve orsum on PATH, else beside the running interpreter (unactivated .venv)."""
    found = shutil.which(str(executable)) or shutil.which(str(executable), path=str(Path(sys.executable).parent))
    if found is None:
        raise FileNotFoundError('orsum.py not found; run scripts/install_orsum.sh .venv or supply --orsum-executable')
    return Path(found).resolve()


def collapse_enrichment(enrichment_dir, *, output_root, executable='orsum.py'):
    enrichment_dir = Path(enrichment_dir)
    manifest = json.loads((enrichment_dir/'manifest.json').read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('Enrichment must be complete before orsum')
    for name, checksum in manifest['artifacts'].items():
        if not (enrichment_dir/name).exists() or file_hash(enrichment_dir/name) != checksum:
            raise ValueError(f'Enrichment artifact changed: {name}')
    binary = find_orsum(executable)
    version = subprocess.check_output([str(binary), '--version'], text=True).strip()
    implementation = {n: file_hash(binary.parent/n) for n in ['orsum.py', 'termCombinationLib.py', 'plotFunctions.py']}
    fdr = manifest['identity']['fdr']
    min_size = manifest['identity']['min_size']
    identity = {'enrichment_manifest': file_hash(enrichment_dir/'manifest.json'), 'code': file_hash(__file__),
                'plot_code': file_hash(Path(__file__).with_name('shared_network_enrichment.py')),
                'orsum_version': version, 'orsum_files': implementation, 'sources': SOURCES,
                'fdr': fdr, 'minTermSize': min_size, 'maxRepSize': 1000000, 'maxTermSize': 1000000}
    out = run_dir(output_root, identity, 'orsum', f'fdr{fdr}')
    if (out/'manifest.json').exists():
        previous = json.loads((out/'manifest.json').read_text())
        if previous.get('status') == 'complete':
            for name, checksum in previous['artifacts'].items():
                if not (out/name).exists() or file_hash(out/name) != checksum:
                    raise ValueError(f'Orsum artifact changed: {name}')
            return out
    all_terms = pd.read_csv(resolve_path(enrichment_dir/'enrichment_all_terms.tsv'), sep='\t')
    out.mkdir(parents=True, exist_ok=True)
    finalize_manifest(out/'manifest.json', {'status': 'running', 'identity': identity})
    representative_frames, mapping_frames, counts = [], [], []
    for library in SOURCES:
        path = Path(manifest['libraries'][library])
        if file_hash(path) != manifest['identity']['gmt'][library]:
            raise ValueError(f'Annotation snapshot changed: {library}')
        definitions = load_gmt(path)
        selected = all_terms[(all_terms.library == library) & (all_terms.q_all_libraries < fdr)].sort_values(['p','term']).copy()
        # Preserve original annotation sets, replacing GMT keys with stable IDs and
        # using readable names in the description column for upstream HTML output.
        def term_id(term):
            description = definitions[term][0]
            matches = re.findall(r'GO:\d+', term)
            if matches:
                return matches[-1]
            if str(description).startswith('R-HSA-'):
                return str(description)
            return term  # supports custom GMT fixtures without native IDs
        selected['term_id'] = [term_id(t) for t in selected.term]
        if selected.term_id.duplicated().any():
            raise ValueError(f'Duplicate stable term IDs: {library}')
        directory = out/library.lower().replace(' ', '_')
        directory.mkdir(parents=True, exist_ok=True)
        gmt = directory/'annotations.gmt'
        ids = [term_id(t) for t in definitions]
        if len(set(ids)) != len(ids):
            raise ValueError(f'Duplicate annotation IDs: {library}')
        with gmt.open('w') as f:
            for term, (_, genes) in definitions.items():
                f.write('\t'.join([term_id(term), term, *sorted(genes)])+'\n')
        ranked = directory/'ranked_terms.txt'
        ranked.write_text(''.join(t+'\n' for t in selected.term_id))
        selected.to_csv(resolve_path(directory/'uncollapsed.tsv'), sep='\t', index=False)
        if selected.empty:
            reps = selected.copy()
            mapping = pd.DataFrame(columns=['Representative','Represented'])
            (directory/'status.txt').write_text('No terms pass combined FDR; orsum not invoked.\n')
        else:
            command = [str(binary), '--gmt', str(gmt.resolve()), '--files', str(ranked.resolve()),
                       '--fileAliases', 'shared_core', '--outputFolder', str((directory/'upstream').resolve()),
                       '--minTermSize', str(min_size), '--maxRepSize', '1000000', '--maxTermSize', '1000000',
                       '--numberOfTermsToPlot', '50']
            result = subprocess.run(command, capture_output=True, text=True, env={**os.environ, 'MPLBACKEND':'Agg'})
            (directory/'command.json').write_text(json.dumps(command, indent=2)+'\n')
            (directory/'stdout.log').write_text(result.stdout)
            (directory/'stderr.log').write_text(result.stderr)
            if result.returncode:
                raise RuntimeError(f'orsum failed for {library}; see {directory}/stderr.log')
            mapping = pd.read_csv(resolve_path(directory/'upstream/filteredResultIDMapping.tsv'), sep='\t')
            if set(mapping.Represented) != set(selected.term_id):
                raise ValueError(f'Orsum did not account for every input term: {library}')
            reps = selected[selected.term_id.isin(mapping.Representative)].copy()
            if not set(mapping.Representative) <= set(selected.term_id):
                raise ValueError('Unknown representative from orsum')
        reps['n_represented_terms'] = reps.term_id.map(mapping.groupby('Representative').size()).fillna(0).astype(int)
        reps.to_csv(resolve_path(directory/'representative_terms.tsv'), sep='\t', index=False)
        mapping['library'] = library
        names = selected.set_index('term_id').term.to_dict()
        mapping['representative_name'] = mapping.Representative.map(names)
        mapping['represented_name'] = mapping.Represented.map(names)
        mapping.to_csv(resolve_path(directory/'term_mapping.tsv'), sep='\t', index=False)
        representative_frames.append(reps)
        mapping_frames.append(mapping)
        counts.append({'library': library, 'significant_terms': len(selected), 'representatives': len(reps),
                       'collapsed_terms': len(selected)-len(reps), 'directory': str(directory.relative_to(out))})
    representatives = pd.concat(representative_frames, ignore_index=True)
    representatives.to_csv(resolve_path(out/'representative_terms.tsv'), sep='\t', index=False)
    pd.concat(mapping_frames, ignore_index=True).to_csv(resolve_path(out/'term_mapping.tsv'), sep='\t', index=False)
    pd.DataFrame(counts).to_csv(resolve_path(out/'collapse_summary.tsv'), sep='\t', index=False)
    plot_enrichment(representatives, out, fdr=fdr, libraries=SOURCES, prefix='collapsed_enrichment', collapsed=True)
    plot_gene_term_heatmap(representatives, pd.read_csv(resolve_path(enrichment_dir/'query_genes.tsv'), sep='\t'), out)
    finalize_manifest(out/'manifest.json', {'status':'complete', 'identity':identity,
        'enrichment_dir':str(enrichment_dir.resolve()), 'executable':str(binary),
        'interpretation':'Upstream orsum on full original annotation sets, separately per source; original p/q retained; no re-testing after collapse',
        'artifacts':{str(p.relative_to(out)):file_hash(p) for p in out.rglob('*') if p.is_file() and p != out/'manifest.json'}})
    return out
