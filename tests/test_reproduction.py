"""CPU-only checks for the public shell interfaces, data closure, and aggregation."""
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from deception_detection.scripts.summarize_results import summarize_run, summarize_table
from deception_detection.scripts.validate_reproduction import ROOT, validate

class ReproductionTests(unittest.TestCase):
    def test_curated_data_closure_and_checksums(self):
        self.assertEqual(validate()['rollouts'],108)

    def test_python_source_is_not_hidden_by_output_ignore_rules(self):
        files=[str(p.relative_to(ROOT)) for p in (ROOT/'deception_detection').rglob('*.py')]
        p=subprocess.run(['git','check-ignore','--no-index',*files],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(p.returncode,1,p.stdout)

    def test_dry_run_from_other_directory_does_not_write_outputs(self):
        before=set((ROOT/'results').rglob('*')) if (ROOT/'results').exists() else set()
        result=subprocess.run(['bash',str(ROOT/'reproduce_results.sh'),'--dry-run','--fresh','--stop-after','ours'],cwd='/tmp',capture_output=True,text=True,check=True)
        self.assertIn('--caching_off=True',result.stdout)
        self.assertEqual(result.stdout.count('--caching_off='),1)
        self.assertIn('[ours]',result.stdout)
        self.assertNotIn('[marks_burger_truth]',result.stdout)
        after=set((ROOT/'results').rglob('*')) if (ROOT/'results').exists() else set()
        self.assertEqual(before,after)

    def test_complete_dry_run_has_ten_probes_and_dyl_preparation(self):
        out=subprocess.check_output(['bash',str(ROOT/'reproduce_results.sh'),'--dry-run'],text=True)
        self.assertEqual(out.count('deception_detection.scripts.experiment run'),10)
        self.assertIn('[prepare_dyl]',out)
        self.assertIn('--batch-size 1',out)

    def test_invalid_stop_target_fails_before_starting(self):
        p=subprocess.run(['bash',str(ROOT/'reproduce_results.sh'),'--stop-after','unknown'],capture_output=True,text=True)
        self.assertEqual(p.returncode,2)

    def test_pipeline_failure_is_propagated_and_next_probe_not_started(self):
        with tempfile.TemporaryDirectory() as folder:
            # Exercise the unchanged runner against a fake Python executable in a disposable copy.
            import shutil
            root=Path(folder);shutil.copy(ROOT/'reproduce_results.sh',root/'reproduce_results.sh')
            shutil.copytree(ROOT/'deception_detection/scripts/configs',root/'deception_detection/scripts/configs')
            fake=root/'python';fake.write_text('#!/bin/bash\ncase "$*" in *validate_reproduction*) exit 0;; *) exit 37;; esac\n');fake.chmod(0o755)
            p=subprocess.run(['bash',str(root/'reproduce_results.sh')],env={**os.environ,'PAPER_MAIN_PYTHON':str(fake)},capture_output=True,text=True)
            self.assertEqual(p.returncode,37)
            self.assertNotIn('[marks_burger_truth]',p.stdout)

    def make_table(self, root):
        table=root/'runs/ours/results_table.csv';table.parent.mkdir(parents=True)
        rows=[]
        for i in range(15):
            rows.extend([(f'personas__paper_ready_p{i}',0.8),(f'personas__paper_ready_p{i}_deceptOnlySysPrompt',0.7)])
        rows += [('confounder_experiments__'+n,v) for n,v in [('probability_score_learning_16turn',1),('persona_stance_nora',0.9),('compliance_brackets',0.6)]]
        rows.append(('training_val',0.0))
        with table.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['','n_honest','n_ambiguous','n_deceptive','auroc'])
            for n,v in rows:w.writerow([n,10,0,10,v])
        return table

    def test_aggregation_excludes_validation_and_marks_partial(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.make_table(root);r=summarize_run(root)
            self.assertEqual(r['state'],'partial');self.assertEqual(r['completed'],1)
            self.assertAlmostEqual(r['rows'][0]['on_policy'],0.8)
            self.assertAlmostEqual(r['rows'][0]['shared_persona_prefill'],0.7)
            self.assertEqual(len(r['pending']),9)

    def test_incomplete_and_nonfinite_results_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            table=self.make_table(Path(folder));t=table.read_text()
            table.write_text(t.replace('personas__paper_ready_p0,','missing,',1))
            with self.assertRaises(ValueError):summarize_table(table)
            table.write_text(t.replace('0.8','nan',1))
            with self.assertRaises(ValueError):summarize_table(table)

if __name__=='__main__':unittest.main()
