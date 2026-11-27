import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(); p.add_argument('--loss',required=True); p.add_argument('--out',required=True); a=p.parse_args(); loss=np.load(a.loss,allow_pickle=False)
if loss.ndim!=1 or not len(loss) or not np.isfinite(loss).all(): raise ValueError(f'Expected numeric 1-D loss history, got {loss.shape}')
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); plt.plot(np.arange(1,len(loss)+1),loss); plt.xlabel('Recorded step'); plt.ylabel('Training loss'); plt.grid(alpha=.2); plt.tight_layout(); plt.savefig(out,dpi=180)
