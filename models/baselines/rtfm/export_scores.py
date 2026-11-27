"""Place beside your existing RTFM model.py and dataset.py; run from that directory."""
import argparse, os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import dataset as dataset_mod
from dataset import Dataset
from model import Model

def remap(root):
    index={}
    for p in Path(root).rglob('*.npy'): index.setdefault(p.name,[]).append(str(p.resolve()))
    original=dataset_mod.group_videos
    def patched(list_file):
        result=[]
        for base,paths in original(list_file):
            resolved=[]
            for p in paths:
                matches=index.get(str(p).replace('\\','/').split('/')[-1],[])
                if len(matches)!=1: raise ValueError(f'{p}: {len(matches)} matches under {root}')
                resolved.append(matches[0])
            result.append((base,resolved))
        return result
    dataset_mod.group_videos=patched

def score_video(model,x):
    n=x.shape[1]
    for c in range(n,0,-1):
        if n%c: continue
        try:
            return torch.stack([model(inputs=x[:,i:i+c])[6][0,:,0] for i in range(0,n,c)]).mean(0)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    raise RuntimeError('GPU OOM even for one crop; use --device cpu')

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--ckpt',required=True); p.add_argument('--out',required=True); p.add_argument('--feat-root'); p.add_argument('--feature-size',type=int); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); p.add_argument('--gt'); p.add_argument('--strip-suffix',default='_i3d'); a=p.parse_args()
    if a.feat_root: remap(a.feat_root)
    ds=Dataset(None,test_mode=True); loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0)
    if len(ds.names)!=len(loader): raise ValueError('Dataset names mismatch')
    state=torch.load(a.ckpt,map_location='cpu',weights_only=True)
    for k in ('model','state_dict'):
        if k in state: state=state[k]; break
    dim=state['fc1.weight'].shape[1]
    if a.feature_size is not None and a.feature_size!=dim: raise ValueError('Feature size mismatch')
    model=Model(dim,a.batch_size).to(a.device); model.load_state_dict(state,strict=True); model.eval(); out={}
    for k,x in zip(ds.names,loader):
        if x.ndim!=4 or x.shape[-1]!=dim: raise ValueError(f'{k}: invalid shape {x.shape}')
        k=str(k); k=k[:-len(a.strip_suffix)] if a.strip_suffix and k.endswith(a.strip_suffix) else k
        if k in out: raise ValueError(f'Duplicate video {k}')
        out[k]=np.repeat(score_video(model,x.to(a.device).permute(0,2,1,3)).float().cpu().numpy(),16)
    dest=Path(a.out); dest.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(dest,**out)
    if a.gt:
        from sklearn.metrics import roc_auc_score
        y=np.load(a.gt,allow_pickle=False).reshape(-1); s=np.concatenate(list(out.values()))
        if len(y)!=len(s): print('GT length mismatch; use named evaluation')
        else: print('AUC (only valid if original list order matches GT):',roc_auc_score(y,s))
    print(len(out),'videos ->',dest)
if __name__=='__main__': main()
