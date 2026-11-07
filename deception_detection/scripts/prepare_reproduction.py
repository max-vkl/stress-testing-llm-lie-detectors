"""Prepare pinned assets once, using existing Hugging Face caches when possible."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from deception_detection.scripts.validate_reproduction import ROOT, MODEL, REVISION, DOLUS_SHA256

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def prepare() -> None:
    from huggingface_hub import HfApi, snapshot_download
    os.environ['HF_HUB_ENABLE_HF_TRANSFER']='0'
    os.environ['HF_HUB_DISABLE_XET']='1'
    try:
        info=HfApi().model_info(MODEL,revision=REVISION,files_metadata=True)
    except Exception as exc:
        raise RuntimeError('Llama access check failed. Accept the model license and set HF_TOKEN or log in to Hugging Face.') from exc
    cache=Path(os.environ.get('HF_HOME',Path.home()/'.cache/huggingface'))
    cache.mkdir(parents=True,exist_ok=True)
    expected={s.rfilename:s.lfs for s in info.siblings if s.rfilename.endswith('.safetensors') and s.lfs}
    try:snapshot=Path(snapshot_download(MODEL,revision=REVISION,local_files_only=True))
    except Exception:snapshot=None
    missing=sum(v.size for k,v in expected.items() if snapshot is None or not (snapshot/k).is_file() or (snapshot/k).stat().st_size!=v.size)
    if shutil.disk_usage(cache).free < missing+2*1024**3:
        raise RuntimeError(f'Insufficient model-cache disk space: need {missing/1024**3:.1f} GiB plus 2 GiB headroom')
    if shutil.disk_usage(ROOT).free < 2*1024**3:raise RuntimeError('At least 2 GiB free space is required for dataset preparation')
    snapshot=Path(snapshot_download(MODEL,revision=REVISION,allow_patterns=['*.json','*.safetensors','*.model','*.jinja'],max_workers=2))
    for name,metadata in sorted(expected.items()):
        path=snapshot/name
        # HF blob filenames are content identities; check existing validated signatures first.
        signature=[str(path.resolve()),path.stat().st_size,path.stat().st_mtime_ns]
        record=ROOT/'.data/model_checksums.json'
        saved=json.loads(record.read_text()) if record.exists() else {}
        if saved.get(name)!={'sha256':metadata.sha256,'signature':signature}:
            print('Verifying model shard:',name,flush=True)
            if sha256(path)!=metadata.sha256:raise RuntimeError(f'Corrupted model shard: {name}; remove that cached blob and rerun setup')
            saved[name]={'sha256':metadata.sha256,'signature':signature}
            record.parent.mkdir(parents=True,exist_ok=True)
            record.write_text(json.dumps(saved,indent=2)+'\n')
    full=ROOT/'data/dolus_chat/full_dataset.json'
    if not full.exists():
        subprocess.run([sys.executable,'-m','deception_detection.scripts.import_dolus_chat','--variant','full_seed0'],cwd=ROOT,check=True)
    if sha256(full)!=DOLUS_SHA256:raise RuntimeError('Full DolusChat integrity check failed')
    print(json.dumps({'model':MODEL,'revision':REVISION,'shards':len(expected),'doluschat_pairs':64888,'state':'ready'}),flush=True)

if __name__=='__main__':prepare()
