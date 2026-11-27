import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve

def main():
    p=argparse.ArgumentParser(); p.add_argument('--eval-dirs',nargs='+',required=True,help='name=path'); p.add_argument('--out-dir',required=True); a=p.parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); rows=[]; reference=None; curves=[]
    for item in a.eval_dirs:
        name,path=item.split('=',1); path=Path(path); m=json.loads((path/'test_metrics.json').read_text())
        with np.load(path/'predictions.npz',allow_pickle=False) as z:
            current=(z['video_names'],z['video_lengths'],z['y_true']); s=z['y_score']; y=current[2]
        if reference is None: reference=current
        elif not all(np.array_equal(a,b) for a,b in zip(reference,current)): raise ValueError(f'{name}: different test coverage or frame labels')
        if rows and rows[0]['threshold']!=m['threshold']: raise ValueError('Threshold mismatch; compare using same fixed threshold or document validation policy separately')
        fields=['accuracy','precision','recall','f1','auc','ap','far_normal_videos','far_all_negative_frames','temporal_map_binary','threshold','n_videos','n_frames']; rows.append({'model':name,**{k:m.get(k) for k in fields}}); curves.append((name,*roc_curve(y,s)[:2],*precision_recall_curve(y,s)[:2]))
    with (out/'model_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for is_roc in [True,False]:
        plt.figure()
        for n,fpr,tpr,pr,rc in curves: plt.plot(fpr if is_roc else rc,tpr if is_roc else pr,label=n)
        plt.xlabel('False positive rate' if is_roc else 'Recall'); plt.ylabel('True positive rate' if is_roc else 'Precision'); plt.legend(); plt.tight_layout(); plt.savefig(out/('combined_roc.png' if is_roc else 'combined_pr.png'),dpi=180); plt.close()
if __name__=='__main__': main()
