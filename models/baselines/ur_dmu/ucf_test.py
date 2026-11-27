
import os
import re
from pathlib import Path

import torch
import numpy as np

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

DEFAULT_SAVE = str(
    ROOT / "results/ur_dmu/video_scores.npz"
)

NUM_CROPS = 10
FRAMES_PER_SEGMENT = 16


def _vname(x):

    if isinstance(x, (list, tuple)):
        x = x[0]

    b = os.path.basename(str(x))

    return re.sub(
        r'(__\d+)?\.npy$',
        '',
        b
    )


@torch.no_grad()
def test(
    net,
    config,
    wind,
    test_loader,
    test_info,
    step,
    model_file=None,
    save_path=DEFAULT_SAVE
):

    net.eval()
    net.flag = "Test"

    device = next(net.parameters()).device

    # Load checkpoint
    if model_file is not None:

        print(
            ">>> Loading checkpoint:",
            model_file,
            flush=True
        )

        state = torch.load(
            model_file,
            map_location=device
        )

        net.load_state_dict(
            state,
            strict=True
        )

    # Load original ground truth
    gt_path = HERE / "frame_label/gt-ucf.npy"

    if not gt_path.exists():
        raise FileNotFoundError(
            f"Ground truth not found: {gt_path}"
        )

    frame_gt = np.load(gt_path).reshape(-1)

    total_samples = len(test_loader.dataset)

    if total_samples % NUM_CROPS != 0:
        raise ValueError(
            f"Invalid crop count: {total_samples}"
        )

    if test_loader.batch_size != 1:
        raise ValueError(
            "UR-DMU test requires batch_size=1"
        )

    frame_predictions = []

    cls_label = []
    cls_pre = []

    crop_scores = []
    crop_names = []
    crop_labels = []

    per_video = {}

    print("=" * 60)
    print("UR-DMU INFERENCE")
    print("=" * 60)
    print("Device:", device)
    print("Feature entries:", total_samples)
    print("Crops per video:", NUM_CROPS)
    print("Expected videos:", total_samples // NUM_CROPS)
    print("=" * 60)

    for i, (_data, _label, _name) in enumerate(
        test_loader
    ):

        _data = _data.to(device)

        label = int(
            _label.reshape(-1)[0].item()
        )

        video_name = _vname(_name)

        # Forward
        res = net(_data)

        if "frame" not in res:
            raise KeyError(
                "Model output does not contain 'frame'"
            )

        prediction = (
            res["frame"]
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        if not prediction.size:
            raise ValueError(
                f"Empty prediction: {video_name}"
            )

        if not np.isfinite(prediction).all():
            raise ValueError(
                f"Invalid prediction: {video_name}"
            )

        if prediction.min() < 0 or prediction.max() > 1:
            raise ValueError(
                f"Expected scores in [0, 1]: {video_name}"
            )

        crop_scores.append(prediction)
        crop_names.append(video_name)
        crop_labels.append(label)

        # Aggregate 10 crops
        if len(crop_scores) == NUM_CROPS:

            # All crops must belong to the same video
            if len(set(crop_names)) != 1:
                raise ValueError(
                    f"Invalid crop grouping: {crop_names}"
                )

            if len(set(crop_labels)) != 1:
                raise ValueError(
                    f"Inconsistent crop labels: {crop_names}"
                )

            shapes = {
                x.shape
                for x in crop_scores
            }

            if len(shapes) != 1:
                raise ValueError(
                    f"Crop lengths mismatch: {video_name}"
                )

            video_name = crop_names[0]

            if video_name in per_video:
                raise ValueError(
                    f"Duplicate video: {video_name}"
                )

            # Mean prediction across 10 crops
            a_predict = np.mean(
                np.stack(crop_scores, axis=0),
                axis=0
            )

            # Video-level prediction
            cls_label.append(crop_labels[0])

            cls_pre.append(
                int(a_predict.max() > 0.5)
            )

            # Segment scores -> frame scores
            fpre = np.repeat(
                a_predict,
                FRAMES_PER_SEGMENT
            ).astype(np.float32)

            per_video[video_name] = fpre

            frame_predictions.append(fpre)

            # Reset crop buffer
            crop_scores.clear()
            crop_names.clear()
            crop_labels.clear()

            processed = len(per_video)

            if processed % 10 == 0 or processed == 290:
                print(
                    f"[{processed}/290] "
                    f"{video_name}: {len(fpre)} frames",
                    flush=True
                )

    # Validate output
    if crop_scores:
        raise ValueError(
            "Unprocessed crops remain"
        )

    if len(per_video) != 290:
        raise ValueError(
            f"Expected 290 videos, got {len(per_video)}"
        )

    frame_predict = np.concatenate(
        frame_predictions
    )

    print("=" * 60)
    print("Feature entries:", total_samples)
    print("Predicted videos:", len(per_video))
    print("Predicted frames:", len(frame_predict))
    print("Ground truth frames:", len(frame_gt))
    print("=" * 60)

    # Save named scores independently of aggregate GT
    # so unified evaluation can align each video.
    if save_path:

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        np.savez_compressed(
            save_path,
            **per_video
        )

        print(
            "[SUCCESS] Saved:",
            save_path
        )

    # Original UR-DMU evaluation
    if len(frame_predict) != len(frame_gt):

        print(
            "[WARNING] Original ground truth and "
            "predictions have different lengths."
        )

        print(
            "Use evaluation/evaluate_scores.py "
            "for named frame-level alignment."
        )

        return None

    fpr, tpr, _ = roc_curve(
        frame_gt,
        frame_predict
    )

    auc_score = float(
        auc(fpr, tpr)
    )

    precision, recall, _ = precision_recall_curve(
        frame_gt,
        frame_predict
    )

    # Original UR-DMU PR-AUC
    ap_score = float(
        auc(recall, precision)
    )

    # sklearn Average Precision
    average_precision = float(
        average_precision_score(
            frame_gt,
            frame_predict
        )
    )

    accuracy = float(
        np.mean(
            np.asarray(cls_label) ==
            np.asarray(cls_pre)
        )
    )

    test_info["step"].append(step)
    test_info["auc"].append(auc_score)
    test_info["ap"].append(ap_score)
    test_info["ac"].append(accuracy)

    print("=" * 60)
    print("UR-DMU ORIGINAL METRICS")
    print("=" * 60)
    print(f"ROC-AUC        : {auc_score:.6f}")
    print(f"PR-AUC         : {ap_score:.6f}")
    print(f"AP             : {average_precision:.6f}")
    print(f"Video Accuracy : {accuracy:.6f}")
    print("=" * 60)

    return {
        "auc": auc_score,
        "pr_auc": ap_score,
        "average_precision": average_precision,
        "video_accuracy": accuracy
    }
