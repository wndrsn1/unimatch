#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path
from typing import List, Optional, Tuple

def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def _is_meta(obj):
    return isinstance(obj, dict) and isinstance(obj.get('frames'), list) and obj.get('frames') and all(k in obj['frames'][0] for k in ('left_image_path','right_image_path','disparity_path'))

def _find_meta(root: Path) -> List[Path]:
    out=[]
    for p in root.rglob('*.json'):
        obj=_load_json(p)
        if _is_meta(obj):
            out.append(p)
    return sorted(out)

def _pick(paths: List[Path]) -> Tuple[Optional[Path], Optional[Path]]:
    train=next((p for p in paths if 'train' in p.name.lower() or 'synthetic' in str(p).lower()), None)
    val=next((p for p in paths if ('val' in p.name.lower() or 'test' in p.name.lower() or 'real' in str(p).lower()) and p!=train), None)
    if train is None and paths: train=paths[0]
    if val is None and len(paths)>1: val=paths[1]
    return train,val

def _rebase(src: Path, dst: Path):
    obj=_load_json(src)
    if not _is_meta(obj):
        raise RuntimeError(f'Invalid metadata: {src}')
    frames=[]
    for fr in obj['frames']:
        item=dict(fr)
        for k in ('left_image_path','right_image_path','disparity_path'):
            item[k]=os.path.relpath((src.parent / item[k]).resolve(), dst.parent)
        frames.append(item)
    obj['frames']=frames
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(obj))

def _download(repo_id: str, out_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError('Install dependency first: pip install huggingface_hub') from exc
    local=out_dir/'hf_snapshot'
    snapshot_download(repo_id=repo_id, repo_type='dataset', local_dir=str(local), local_dir_use_symlinks=False, resume_download=True)
    return local

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo_id', default='jacoblin/cloud-stereo')
    ap.add_argument('--output_dir', default='datasets/cloud-stereo')
    ap.add_argument('--source_dir', default=None)
    args=ap.parse_args()

    out=Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    src=Path(args.source_dir).resolve() if args.source_dir else _download(args.repo_id, out)

    metas=_find_meta(src)
    if not metas: raise RuntimeError(f'No valid Cloud-Stereo metadata JSON found under {src}')
    train,val=_pick(metas)
    if train is None: raise RuntimeError('Could not infer training metadata file')
    _rebase(train, out/'train.json'); print(f'Wrote {out/"train.json"} from {train}')
    if val:
        _rebase(val, out/'val.json'); print(f'Wrote {out/"val.json"} from {val}')
    else:
        print('Warning: no val/test metadata file inferred.')

    link=out/'data'
    if link.is_symlink() or link.exists():
        if link.is_dir() and not link.is_symlink(): shutil.rmtree(link)
        else: link.unlink()
    os.symlink(src, link)
    print(f'Created symlink {link} -> {src}')

if __name__=='__main__':
    main()
