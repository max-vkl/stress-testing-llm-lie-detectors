"""Import the real public execution path before reserving GPU time."""
import importlib
from pathlib import Path
import pytest
import yaml
from deception_detection.experiment import ExperimentConfig
from deception_detection.scripts.validate_reproduction import LAYERS, ROOT

def test_reproduction_entrypoints_import_without_optional_services():
    for name in ('experiment', 'prepare_dyl_followups', 'import_dolus_chat',
                 'prepare_reproduction', 'validate_reproduction', 'summarize_results'):
        importlib.import_module('deception_detection.scripts.'+name)

@pytest.mark.parametrize('probe',list(LAYERS))
def test_every_paper_config_survives_saved_result_reload(probe, tmp_path):
    path=ROOT/'deception_detection/scripts/configs/paper_main_llama'/(probe+'.yaml')
    cfg=ExperimentConfig(**yaml.safe_load(path.read_text()),folder=tmp_path,name=probe,caching_off=True)
    restored=ExperimentConfig.from_path(cfg.folder)
    assert restored.detect_layers==cfg.detect_layers
    assert restored.train_data==cfg.train_data
    assert restored.extra_train_data==cfg.extra_train_data
    assert restored.eval_data==cfg.eval_data
    assert restored.caching_off is True
