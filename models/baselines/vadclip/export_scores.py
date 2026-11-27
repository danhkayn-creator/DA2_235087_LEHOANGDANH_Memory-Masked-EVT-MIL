"""Place beside your existing VadCLIP model.py, ucf_option.py and utils/."""
import os,tempfile,re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import models.baselines.vadclip.ucf_option as ucf_option
from models.baselines.vadclip.model import CLIPVAD
from models.baselines.vadclip.utils.dataset import UCFDataset
from models.baselines.vadclip.utils.tools import get_batch_mask,get_prompt_text

def remap_csv(csv_path,root,temp):
    df=pd.read_csv(csv_path); col=df.columns[0]
    if root:
        index={}
        for p in Path(root).rglob('*.npy'): index.setdefault(p.name,[]).append(str(p.resolve()))
        def resolve(x):
            p = Path(str(x).replace("\\", "/"))
            root_path = Path(root).resolve()

            # Ưu tiên đường dẫn tương đối gốc
            if "UCFClipFeatures" in p.parts:
                idx = p.parts.index("UCFClipFeatures")
                relative = Path(*p.parts[idx + 1:])
                candidate = root_path / relative

                if candidate.is_file():
                    return str(candidate)

            # Tìm theo thư mục cha và tên file
            candidate = root_path / p.parent.name / p.name

            if candidate.is_file():
                return str(candidate)

            # Fallback: tìm theo tên file
            matches = list(root_path.rglob(p.name))

            if len(matches) != 1:
                raise ValueError(
                    f"{p}: {len(matches)} matches under {root_path}"
                )

            return str(matches[0])
        df[col]=df[col].map(resolve)
    if not df[col].map(os.path.isfile).all(): raise FileNotFoundError('Missing feature path')
    df.to_csv(temp,index=False); return df[col].tolist()

@torch.inference_mode()
def main():
    p=ucf_option.parser; p.add_argument('--feat-root'); p.add_argument('--score',type=int,choices=[1,2],default=1); p.add_argument('--out',required=True); a=p.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    labels={k:k for k in ['Normal','Abuse','Arrest','Arson','Assault','Burglary','Explosion','Fighting','RoadAccidents','Robbery','Shooting','Shoplifting','Stealing','Vandalism']}
    model=CLIPVAD(a.classes_num,a.embed_dim,a.visual_length,a.visual_width,a.visual_head,a.visual_layers,a.attn_window,a.prompt_prefix,a.prompt_postfix,device)
    checkpoint = torch.load(
    a.model_path,
    map_location="cpu",
    weights_only=False
    )

    state = checkpoint["model_state_dict"]

    print("[INFO] Checkpoint loaded successfully")
    print("[INFO] Epoch:", checkpoint.get("epoch"))
    print("[INFO] Saved AP:", checkpoint.get("ap"))

    # Xử lý DataParallel nếu có
    state = {
        k.removeprefix("module."): v
        for k, v in state.items()
    }
    for k in ('model','state_dict'):
        if k in state: state=state[k]; break
    model.load_state_dict(state,strict=True); model.to(device).eval(); groups={}; prompts=get_prompt_text(labels)
    with tempfile.TemporaryDirectory() as tmp:
        csv=Path(tmp)/'test.csv'; paths=remap_csv(a.test_list,a.feat_root,csv)
        ds=UCFDataset(a.visual_length,str(csv),True,labels); loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0)
        if len(loader)!=len(paths): raise ValueError('Dataset/list mismatch')
        for path,item in zip(paths,loader):
            visual=item[0].squeeze(0); length=int(item[2]); maxlen=a.visual_length
            if length<=0: raise ValueError(f'Empty sequence {path}')
            if visual.ndim==2: visual=visual.unsqueeze(0)
            lengths=torch.tensor([min(maxlen,length-i) for i in range(0,length,maxlen)],dtype=torch.long)
            if visual.ndim!=3 or visual.shape[0]<len(lengths) or visual.shape[1]!=maxlen: raise ValueError(f'{path}: inconsistent padded shape {visual.shape}, lengths {lengths.tolist()}')
            visual=visual[:len(lengths)].to(device); mask=get_batch_mask(lengths,maxlen).to(device)
            _,l1,l2=model(visual,mask,prompts,lengths)
            if a.score==1: s=torch.sigmoid(l1.reshape(-1)[:length])
            else: s=1-l2.reshape(-1,l2.shape[-1])[:length].softmax(-1)[:,0]
            if s.numel()!=length: raise ValueError(f'{path}: score length mismatch')
            k=re.sub(r'__\d+$','',Path(path).stem); groups.setdefault(k,[]).append(s.float().cpu().numpy())
    out={}
    for k,seq in groups.items():
        if len({len(s) for s in seq})!=1: raise ValueError(f'{k}: crop lengths mismatch')
        out[k]=np.repeat(np.mean(seq,axis=0),16).astype(np.float32)
    dest=Path(a.out); dest.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(dest,**out); print(len(out),'videos ->',dest)
if __name__=='__main__': main()
