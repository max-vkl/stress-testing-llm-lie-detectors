"""Check the curated data/config closure without loading model weights."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[2]
LAYERS=dict(zip(('ours','marks_burger_truth','marks_burger_lie','goldowsky_dill_repe',
    'goldowsky_dill_roleplay','zou_repe','cooney_dyl','cundy_doluschat','macdiarmid_cp','goldowsky_dill_sd'),
    (20,18,20,22,21,34,31,34,32,36),strict=True))
MODEL='meta-llama/Llama-3.3-70B-Instruct'
REVISION='6f6073b423013f6a7d4d9f39144961bfbfbc386b'
DOLUS_SHA256='7cc8b0e938ae59ae7d24c568865614060775c982e74254668c91ea0a4c278396'

def validate(root: Path=ROOT, require_assets: bool=False, require_gpu: bool=False) -> dict:
    mapping=json.loads((root/'dataset_paths.json').read_text())
    assert len(mapping)==108, 'Expected 108 retained rollout registrations'
    provenance=json.loads((root/'data/provenance.json').read_text())
    for name,digest in provenance['data_sha256'].items():
        path=root/name
        assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==digest, f'Data changed: {name}'
    for key,value in mapping.items():
        path=(root/value).resolve()
        assert path.is_relative_to(root.resolve()) and path.is_file(), f'Missing or external rollout: {key}'
    config_dir=root/'deception_detection/scripts/configs/paper_main_llama'
    assert {p.stem for p in config_dir.glob('*.yaml')}==set(LAYERS), 'Unexpected experiment configs'
    inputs=set()
    for name,layer in LAYERS.items():
        cfg=yaml.safe_load((config_dir/(name+'.yaml')).read_text())
        assert cfg['model_name']=='llama-70b-3.3' and cfg['detect_layers']==[layer], name
        assert len(cfg['eval_data'])==33 and len(set(cfg['eval_data']))==33, name
        for dataset in cfg['eval_data']:
            for model in ('llama-70b-3.3','gemma-3-27b-it','gemma-4-31b-it'):
                assert dataset+'__'+model in mapping, f'Missing {dataset} for {model}'
        dataset=cfg['train_data']
        train_model='llama-70b-3.3' if cfg.get('train_on_policy',False) else 'prewritten'
        inputs.add((dataset,train_model))
        if train_model!='prewritten':assert dataset+'__'+train_model in mapping
        elif dataset.startswith('tiu__'):assert (root/'data/tiu'/(dataset.split('__')[1]+'.jsonl')).is_file()
        for dataset in cfg.get('extra_train_data',[]):
            for model in ('llama-70b-3.3','gemma-3-27b-it','gemma-4-31b-it'):
                assert dataset+'__'+model in mapping, f'Missing extra-training rollout: {dataset} for {model}'
            inputs.add((dataset,'llama-70b-3.3'))
        inputs.update((dataset,'llama-70b-3.3') for dataset in cfg['eval_data'])
    if require_assets:
        full=root/'data/dolus_chat/full_dataset.json'
        assert full.is_file() and hashlib.sha256(full.read_bytes()).hexdigest()==DOLUS_SHA256, 'Run setup.sh to prepare full DolusChat'
        from huggingface_hub import snapshot_download
        snapshot=Path(snapshot_download(MODEL,revision=REVISION,local_files_only=True))
        index=json.loads((snapshot/'model.safetensors.index.json').read_text())
        assert all((snapshot/f).is_file() and (snapshot/f).stat().st_size>0 for f in set(index['weight_map'].values())), 'Incomplete model cache'
        from deception_detection.models import ModelName, get_model_and_tokenizer
        from deception_detection.repository import DatasetRepository
        _,tokenizer=get_model_and_tokenizer(ModelName.LLAMA_70B_3_3,omit_model=True)
        repository=DatasetRepository()
        for dataset,model in sorted(inputs):
            loaded=repository.get(dataset,model=model,tokenizer=tokenizer)
            assert len(loaded)>0, f'Empty input dataset: {dataset}'
    if require_gpu:
        import torch
        assert torch.cuda.is_available(), 'CUDA GPU required'
        assert torch.cuda.is_bf16_supported(), 'BF16-capable CUDA GPU required'
    return {'configs':10,'rollouts':len(mapping),'integrity':'passed','assets_checked':require_assets,'gpu_checked':require_gpu}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--require-assets',action='store_true');p.add_argument('--require-gpu',action='store_true')
    a=p.parse_args();print(json.dumps(validate(require_assets=a.require_assets,require_gpu=a.require_gpu)))
