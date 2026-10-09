# Literature evidence for CNV × disease comparisons

`cnv_disease_evidence.tsv` lists published links between the schizophrenia CNVs
and non-psychiatric diseases. The publication workflow joins it to the CNV × disease
tests of each network (`publication_pipeline.py`, setting `disease_evidence`).

## Columns

| Column | Meaning |
| --- | --- |
| `cnv` | CNV name, as in `input/processed/cnv_disorder_overlap_seeds.tsv` |
| `target` | Disease target name in the seed file, or a new name for a disease we do not test |
| `literature_dosage` | Dosage studied in the source (`deletion`, `duplication`, or explicitly mixed syndrome mechanisms) |
| `evidence_status` | `cohort association`, `case-level evidence`, `related phenotype` or `related phenotype only` |
| `finding` | One-line summary of the published finding |
| `source_url` | PubMed or PMC link |
| `caveat` | Interval, dosage or phenotype mismatch to state when the row is cited |
| `review_date` | Date of the check |

## How to read the joined tables

- Network propagation uses the CNV's genes without dosage, because we have no
  deletion/duplication assignment. A literature row about a deletion therefore
  does not confirm or contradict a colocalization. Each joined row has
  `dosage_match = unresolved: propagated CNV is unsigned`.
- A significant comparison with no literature row is `not reviewed`. That is
  not evidence of absence.
- A row whose disease we do not test (`tested_in_this_network = False`) is a lead
  for the reverse direction: a disease seen in the syndrome that could be added.
  Adding it requires a new frozen seed set and rerunning BH across the larger
  disease family. Do not add a test after looking at the results.
- `related phenotype only` rows are there to stop over-reading. For example,
  aortic dissection does not support a coronary artery disease result.

## Sources

The review is preliminary, not systematic. Citations were checked against NCBI
records on 2026-10-07.

| Source | Reference |
| --- | --- |
| PMC7565196 | 22q11.2 microdeletion and increased risk for type 2 diabetes. *EClinicalMedicine* 2020 |
| PMC11196336 | Endocrine manifestations in adults with 22q11.2 deletion syndrome. *J Endocrinol Invest* 2024 |
| PMC5292049 | Obesity in adults with 22q11.2 deletion syndrome. *Genet Med* 2017 |
| PMC2880448 | A new highly penetrant form of obesity due to deletions on chromosome 16p11.2. *Nature* 2010 |
| nature10406 | Mirror extreme BMI phenotypes associated with gene dosage at the chromosome 16p11.2 locus. *Nature* 2011 |
| PMC5650138 | Impaired glucose metabolism in subjects with the Williams-Beuren syndrome. *PLoS One* 2017 |
| PMID21698135 | Recurrent chromosome 16p13.1 duplications are a risk factor for aortic dissections. *PLoS Genet* 2011 |
| PMC4228180 | Constitutional and somatic deletions of the Williams-Beuren syndrome critical region in non-Hodgkin lymphoma. *J Hematol Oncol* 2014 |
| PMID30343275 | Medical consequences of pathogenic CNVs in adults: analysis of the UK Biobank. *J Med Genet* 2019 |

The UK Biobank study covers many CNVs and traits. It has not yet been extracted
row by row into the TSV.

Additional reviewed sources include the [GREM1 duplication study](https://pubmed.ncbi.nlm.nih.gov/22561515/)
and a [Williams–Beuren cardiovascular case report](https://academic.oup.com/ejcts/article/58/1/196/5682457).
The GREM1 finding concerns a different CNV: the frozen GRCh37 coordinate table
places SCG5 and GREM1 outside this project's 15q13.3 interval. The Williams finding
concerns congenital arterial/coronary anatomy, which does not establish an
association with the common CAD GWAS phenotype. Both distinctions are recorded
in the evidence table.

## Review coverage

`search_log.tsv` records targeted searches and unresolved leads. It is not a
systematic review and does not support claims that an association is novel or
absent. The publication workflow exports `literature_review_queue.tsv` for
significant comparisons lacking reviewed evidence.

The Prader–Willi glucose-metabolism cohort (PMID27381990) is included with an
explicit imprinting and genotype caveat: its deletion/UPD evidence cannot be
transferred to an unsigned 15q11.2-q13.1 set. Obesity, underweight, aortic dissection
and lymphoma remain phenotype leads outside the frozen disease panel. Adding
them requires new curated gene sets and a fresh multiple-testing family.
