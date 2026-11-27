"""Optional extraction for NEW raw videos; never overwrite official VadCLIP features."""
import argparse
from pathlib import Path
import json
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='Raw video directory')
    p.add_argument('--output', type=Path, required=True, help='Separate output directory')
    p.add_argument('--stride', type=int, default=16, help='Sample every N decoded frames')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    args = p.parse_args()
    if args.stride < 1 or args.batch_size < 1: p.error('stride and batch-size must be positive')
    if not args.input.is_dir(): p.error('input directory does not exist')
    if args.output.resolve() == args.input.resolve() or args.input.resolve() in args.output.resolve().parents:
        p.error('output must be outside input directory')
    if args.output.resolve().name == 'clip_features': p.error('Use a separate output directory, e.g. custom_clip_features')
    import cv2
    import torch
    import clip
    from PIL import Image
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    model, preprocess = clip.load('ViT-B/16', device=device)
    model.eval().requires_grad_(False)
    videos = sorted(v for v in args.input.rglob('*') if v.is_file() and v.suffix.lower() in {'.mp4','.avi','.mov','.mkv'})
    if not videos: raise ValueError('No videos found')
    targets=[(args.output/v.relative_to(args.input)).with_suffix('.npy') for v in videos]
    if len(set(targets))!=len(targets): raise ValueError('Output name collision')
    failed=[]
    for video,dst in zip(videos,targets):
        if dst.exists():
            print('Skip existing:',dst); continue
        cap=cv2.VideoCapture(str(video)); batch=[]; feats=[]; indices=[]; frame_idx=0
        fps=float(cap.get(cv2.CAP_PROP_FPS))
        def flush():
            if batch:
                with torch.inference_mode():
                    z=model.encode_image(torch.stack(batch).to(device)).float().cpu().numpy()
                feats.append(z); batch.clear()
        try:
            if not cap.isOpened(): raise ValueError('Cannot open video')
            while True:
                ok,frame=cap.read()
                if not ok: break
                if frame_idx % args.stride == 0:
                    batch.append(preprocess(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))))
                    indices.append(frame_idx)
                    if len(batch)>=args.batch_size: flush()
                frame_idx+=1
            flush()
            if not feats: raise ValueError('No decoded frames')
            arr=np.concatenate(feats,axis=0)
            if arr.shape!=(len(indices),512) or not np.isfinite(arr).all(): raise ValueError('Invalid CLIP features')
            dst.parent.mkdir(parents=True,exist_ok=True)
            np.save(dst,arr,allow_pickle=False)
            dst.with_suffix('.json').write_text(json.dumps({'source':video.relative_to(args.input).as_posix(),'fps':fps if np.isfinite(fps) and fps>0 else None,'decoded_frames':frame_idx,'frame_indices':indices,'stride':args.stride,'encoder':'ViT-B/16','feature_shape':list(arr.shape)},indent=2),encoding='utf8')
            print(dst,arr.shape)
        except Exception as e:
            failed.append((str(video),str(e))); print('ERROR:',video,e)
        finally: cap.release()
    if failed: raise RuntimeError(f'{len(failed)} videos failed: {failed[:10]}')

if __name__=='__main__': main()
