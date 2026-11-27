"""Convert existing MIL .mat predictions using user's MATLAB mapping; evaluate."""
import argparse,csv,sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'evaluation'))
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evaluation.evaluate_scores import (
    annotations,
    frame_lengths,
    evaluate,
    name
)

def frame_scores_matlab(pred,n,actual):
    if n<=0 or actual<=0: raise ValueError('Invalid frame count')
    det=np.zeros(n*16,np.float32)
    shots=np.floor(np.linspace(1,n,33)+.5).astype(int)
    for i in range(32):
        start=(shots[i]-1)*16; end=max(shots[i],shots[i+1]-1)*16
        det[start:end]=pred[i]
    return np.pad(det,(0,max(0,actual-len(det))),mode='edge')[:actual]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--results_dir',required=True); p.add_argument('--ann_txt',required=True); p.add_argument('--videos',required=True); p.add_argument('--counts_csv',required=True); p.add_argument('--out_dir',required=True); p.add_argument('--threshold',type=float,default=.5); p.add_argument('--temporal-map',action='store_true'); a=p.parse_args()
    ann=annotations(a.ann_txt); lengths=frame_lengths(a.videos,None)
    with open(a.counts_csv) as f: counts={name(r['video']):int(r['num_c3d_feats']) for r in csv.DictReader(f)}
    out={}
    for f in sorted(Path(a.results_dir).glob('*.mat')):
        k=name(f.name); pred=np.asarray(loadmat(f)['predictions']).reshape(-1)
        if k in out: raise ValueError(f'Duplicate {k}')
        if len(pred)!=32: raise ValueError(f'{k}: expected 32 scores')
        out[k]=frame_scores_matlab(pred,counts[k],lengths[k])
    dest=Path(a.out_dir); dest.mkdir(parents=True,exist_ok=True)
    evaluate(out,ann,lengths,dest,a.threshold,temporal=a.temporal_map)
    np.savez_compressed(dest/'video_scores.npz',**out)
    # legacy tie-order dependent AUC, preserved separately for reproducibility
    scores=np.concatenate([out[k] for k in sorted(out)]); y=[]
    for k in sorted(out):
        a1=np.zeros(lengths[k],np.uint8)
        for s,e in ann[k]: a1[s-1:e]=1
        y.append(a1)
    gt=np.concatenate(y); g=gt[np.argsort(-scores,kind='stable')]; rec=np.cumsum(g>0)/gt.sum(); fpr=np.cumsum(g==0)/(gt==0).sum(); trap=getattr(np,'trapezoid',None) or np.trapz
    import json
    (dest/'auc_result.json').write_text(json.dumps({'auc_matlab_style':float(trap(rec,fpr)),'note':'tie-order dependent; sorted video names'},indent=2))
if __name__=='__main__': main()
