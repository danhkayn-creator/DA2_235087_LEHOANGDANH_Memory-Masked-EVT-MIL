
import json
from pathlib import Path

import torch
import torch.utils.data as data
import numpy as np

import options
from config import Config
from model import WSAD
from dataset_loader import UCF_crime
from ucf_test import test


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():

    args = options.parse_args()
    config = Config(args)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "UR-DMU original pipeline requires CUDA"
        )

    device = torch.device("cuda")

    output_dir = ROOT / "results/ur_dmu"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    net = WSAD(
        config.len_feature,
        flag="Test",
        a_nums=60,
        n_nums=60
    ).to(device).eval()

    dataset = UCF_crime(
        root_dir=config.root_dir,
        mode="Test",
        modal=config.modal,
        num_segments=config.num_segments,
        len_feature=config.len_feature
    )

    loader = data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers
    )

    info = {
        "step": [],
        "auc": [],
        "ap": [],
        "ac": []
    }

    with torch.no_grad():

        metrics = test(
            net,
            config,
            None,
            loader,
            info,
            step=0,
            model_file=args.model_file,
            save_path=str(
                output_dir / "video_scores.npz"
            )
        )

    result = {
        "checkpoint": str(args.model_file),
        "n_feature_entries": len(dataset),
        "n_videos": len(dataset) // 10,
        "crops_per_video": 10,
        "original_metrics": metrics,
        "status": (
            "completed"
            if metrics is not None
            else "scores_exported_gt_length_mismatch"
        )
    }

    out = output_dir / "original_metrics.json"

    out.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
