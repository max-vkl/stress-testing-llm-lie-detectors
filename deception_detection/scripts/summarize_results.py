"""Aggregate the paper's five AUROC columns from complete per-probe CSV files."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

PROBES = ('ours', 'marks_burger_truth', 'marks_burger_lie', 'goldowsky_dill_repe',
          'goldowsky_dill_roleplay', 'zou_repe', 'cooney_dyl', 'cundy_doluschat',
          'macdiarmid_cp', 'goldowsky_dill_sd')
CONFOUNDERS = ('probability_score_learning_16turn', 'persona_stance_nora', 'compliance_brackets')
METRICS = ('on_policy', 'shared_persona_prefill', 'likelihood', 'persona_belief', 'compliance')

def summarize_table(path: Path) -> dict[str, float]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    values = {}
    for row in rows:
        key = row.get('', row.get('dataset', ''))
        if key in values:
            raise ValueError(f'Duplicate dataset in {path}: {key}')
        value = float(row['auroc'])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'Invalid AUROC in {path}: {key}')
        values[key] = value
    on = {k:v for k,v in values.items() if k.startswith('personas__paper_ready_') and not k.endswith('SysPrompt')}
    shared = {k:v for k,v in values.items() if k.startswith('personas__paper_ready_') and k.endswith('_deceptOnlySysPrompt')}
    if len(on) != 15 or set(on) != {k.removesuffix('_deceptOnlySysPrompt') for k in shared}:
        raise ValueError(f'Expected matching sets of 15 persona conditions in {path}')
    numbers = [mean(on.values()), mean(shared.values())]
    numbers.extend(values['confounder_experiments__'+key] for key in CONFOUNDERS)
    return dict(zip(METRICS, numbers, strict=True))

def summarize_run(run: Path) -> dict:
    rows = []
    for probe in PROBES:
        path = run/'runs'/probe/'results_table.csv'
        if path.is_file(): rows.append({'detector':probe, **summarize_table(path)})
    if not rows: raise ValueError(f'No completed result tables in {run}')
    report = {'state':'complete' if len(rows)==10 else 'partial', 'completed':len(rows),
              'expected':10, 'aggregation':'Equal-weight mean over 15 personas; validation excluded.',
              'rows':rows, 'pending':[p for p in PROBES if p not in {r['detector'] for r in rows}]}
    (run/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    with (run/'main_table.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['detector',*METRICS]);writer.writeheader();writer.writerows(rows)
    return report

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run_directory',type=Path)
    print(json.dumps(summarize_run(parser.parse_args().run_directory),indent=2))
