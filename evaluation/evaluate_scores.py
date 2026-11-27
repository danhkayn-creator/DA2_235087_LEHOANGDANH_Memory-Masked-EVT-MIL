
"""
Evaluate named frame scores against UCF temporal annotations.

Supports:
- UCF-Crime annotation files with 6 or 7 columns.
- Frame-level ROC-AUC, AP, Accuracy, Precision, Recall, F1.
- False Alarm Rate (FAR).
- Binary Temporal AP and mAP.
- Annotation boundary clipping.
- Strict test coverage validation.

No GPU needed.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
}

MAX_FRAME_TOLERANCE = 2


# ============================================================
# VIDEO NAME NORMALIZATION
# ============================================================

def name(s):
    p = Path(str(s))

    if p.suffix.lower() in VIDEO_EXTENSIONS | {".npy", ".mat"}:
        s = p.stem
    else:
        s = p.name

    return re.sub(
        r"(?:__\d+|_i3d|_C)$",
        "",
        s
    )


# ============================================================
# LOAD TEMPORAL ANNOTATIONS
# ============================================================

def annotations(path):
    out = {}

    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1
    ):
        t = line.split()

        if not t:
            continue

        if len(t) not in (6, 7):
            raise ValueError(
                f"Invalid annotation at line {line_number}: {line}"
            )

        k = name(t[0])
        category = t[1]

        # Supports both 6-column and 7-column annotations.
        # Never include the optional total-frame column.
        values = list(map(int, t[2:6]))

        if k in out:
            raise ValueError(
                f"Duplicate annotation: {k}"
            )

        valid_intervals = []

        for start, end in zip(
            values[::2],
            values[1::2]
        ):
            # Normal or unused annotation interval
            if start == -1 and end == -1:
                continue

            if start < 0 or end < 0 or end < start:
                raise ValueError(
                    f"Invalid interval: {k}, "
                    f"({start}, {end})"
                )

            valid_intervals.append(
                (start, end)
            )

        # Validate Normal videos
        if category.lower() == "normal" and valid_intervals:
            raise ValueError(
                f"Normal video contains anomaly annotations: {k}"
            )

        out[k] = valid_intervals

    if not out:
        raise ValueError("Annotation file is empty")

    return out


# ============================================================
# LOAD FRAME LENGTHS
# ============================================================

def frame_lengths(videos, counts):

    # Option 1: Read frame metadata from CSV
    if counts:
        with open(counts, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        out = {
            name(r["video"]): int(r["frames"])
            for r in rows
        }

        if len(out) != len(rows):
            raise ValueError(
                "Duplicate video in frames CSV"
            )

        return out

    # Option 2: Read original video metadata
    import cv2

    root = Path(videos)

    if not root.is_dir():
        raise FileNotFoundError(
            f"Video directory not found: {root}"
        )

    out = {}

    for p in sorted(root.rglob("*")):

        if not p.is_file():
            continue

        if p.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        k = name(p.name)

        if k in out:
            raise ValueError(
                f"Duplicate raw video: {k}"
            )

        cap = cv2.VideoCapture(str(p))

        try:
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()

        if (
            not np.isfinite(n)
            or not np.isfinite(fps)
            or n <= 0
            or fps <= 0
        ):
            raise ValueError(
                f"Invalid video metadata: {p}"
            )

        # Normalize video duration to 30 FPS.
        # Equivalent to MATLAB rounding for positive values.
        out[k] = int(
            np.floor(n / fps * 30 + 0.5)
        )

    return out


# ============================================================
# EXTRACT CONTIGUOUS INTERVALS
# ============================================================

def intervals(mask):

    mask = np.asarray(
        mask,
        dtype=bool
    ).reshape(-1)

    d = np.diff(
        np.r_[False, mask, False].astype(np.int8)
    )

    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)

    return list(zip(starts, ends))


# ============================================================
# TEMPORAL AVERAGE PRECISION
# ============================================================

def temporal_ap(proposals, truth, iou):

    total = sum(
        map(len, truth.values())
    )

    if not total:
        return None

    used = {
        k: set()
        for k in truth
    }

    tp = []

    sorted_proposals = sorted(
        proposals,
        key=lambda r: -r[0]
    )

    for conf, k, s, e in sorted_proposals:

        overlaps = []

        for a, b in truth[k]:

            intersection = max(
                0,
                min(e, b) - max(s, a)
            )

            union = max(e, b) - min(s, a)

            overlap = (
                intersection / union
                if union > 0 else 0.0
            )

            overlaps.append(overlap)

        eligible = [
            j
            for j, value in enumerate(overlaps)
            if value >= iou and j not in used[k]
        ]

        if eligible:

            j = max(
                eligible,
                key=lambda j: overlaps[j]
            )

            used[k].add(j)
            tp.append(1)

        else:
            tp.append(0)

    if not tp:
        return 0.0

    cum = np.cumsum(tp)

    recall = cum / total

    precision = cum / np.arange(
        1,
        len(tp) + 1
    )

    r = np.r_[0, recall, 1]
    p = np.r_[0, precision, 0]

    p = np.maximum.accumulate(
        p[::-1]
    )[::-1]

    return float(
        np.sum(np.diff(r) * p[1:])
    )


# ============================================================
# BUILD FRAME-LEVEL GROUND TRUTH
# ============================================================

def build_ground_truth(
    video_name,
    annotation,
    n_frames,
    base=1,
    tolerance=MAX_FRAME_TOLERANCE,
    adjustments=None
):

    if n_frames <= 0:
        raise ValueError(
            f"Invalid frame count: {video_name}"
        )

    # Every video starts as entirely Normal.
    y = np.zeros(
        n_frames,
        dtype=np.uint8
    )

    # Normal videos have no anomaly intervals.
    # The loop is skipped, leaving all labels equal to 0.
    for start, end in annotation:

        # Convert inclusive annotation endpoints
        # into Python zero-based half-open intervals.
        a = start - base
        b = end + 1 - base

        # Correct minor mismatches at the end of the video.
        if 0 <= a < n_frames and b > n_frames:

            diff = b - n_frames

            if diff <= tolerance:

                print(
                    f"[WARNING] {video_name}: "
                    f"annotation end {b} -> {n_frames} "
                    f"(difference: {diff} frames)"
                )

                if adjustments is not None:
                    adjustments.append({
                        "video": video_name,
                        "original_start": int(start),
                        "original_end": int(end),
                        "converted_start": int(a),
                        "converted_end": int(b),
                        "corrected_end": int(n_frames),
                        "difference": int(diff),
                    })

                b = n_frames

            else:
                raise ValueError(
                    f"Large annotation mismatch: "
                    f"{video_name}, end={b}, "
                    f"frames={n_frames}, diff={diff}"
                )

        # Validate annotation boundaries.
        if not 0 <= a < b <= n_frames:
            raise ValueError(
                f"Annotation outside frame range: "
                f"{video_name}, {(a, b)}, "
                f"n={n_frames}"
            )

        # IMPORTANT: This must stay inside the loop.
        # All anomaly intervals must be included.
        y[a:b] = 1

    return y


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate(
    scores,
    ann,
    lengths,
    out,
    threshold=0.5,
    base=1,
    tail="strict",
    allow_subset=False,
    temporal=False,
    max_tail=15
):

    out = Path(out)
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    if not 0 <= threshold <= 1:
        raise ValueError(
            "Threshold must be in [0, 1]"
        )

    if base not in (0, 1):
        raise ValueError(
            "Annotation base must be 0 or 1"
        )

    if tail not in ("strict", "edge"):
        raise ValueError(
            f"Invalid tail policy: {tail}"
        )

    if max_tail < 0:
        raise ValueError(
            "max_tail must be non-negative"
        )

    # --------------------------------------------------------
    # Test coverage validation
    # --------------------------------------------------------

    missing = set(ann) - set(scores)
    extra = set(scores) - set(ann)

    if (missing or extra) and not allow_subset:
        raise ValueError(
            f"Test coverage mismatch: "
            f"missing={sorted(missing)[:10]}, "
            f"extra={sorted(extra)[:10]}"
        )

    keys = sorted(
        set(ann) & set(scores)
    )

    if not keys:
        raise ValueError(
            "No matching test videos"
        )

    ys = []
    ss = []

    truth = {}
    proposals = []

    normal_fp = 0
    normal_n = 0

    normal_videos = 0
    abnormal_videos = 0

    adjustments = []
    annotation_adjustments = []

    print("=" * 60)
    print("UCF-CRIME BASELINE EVALUATION")
    print("=" * 60)

    print("Expected test videos:", len(ann))
    print("Available score files:", len(scores))
    print("Matched test videos:", len(keys))
    print("Missing videos:", len(missing))
    print("Extra videos:", len(extra))
    print("Threshold:", threshold)
    print("Annotation base:", base)

    print("=" * 60)

    # --------------------------------------------------------
    # Process individual videos
    # --------------------------------------------------------

    for k in keys:

        if k not in lengths:
            raise ValueError(
                f"Missing frame count: {k}"
            )

        n = int(lengths[k])

        s = np.asarray(
            scores[k],
            dtype=np.float32
        ).reshape(-1)

        # Validate scores
        if (
            n <= 0
            or not len(s)
            or not np.isfinite(s).all()
        ):
            raise ValueError(
                f"Invalid score/length: {k}"
            )

        if s.min() < 0 or s.max() > 1:
            raise ValueError(
                f"Expected probability [0,1]: {k}"
            )

        # ----------------------------------------------------
        # Align score length with video length
        # ----------------------------------------------------

        if len(s) != n:

            if (
                tail == "strict"
                or abs(len(s) - n) > max_tail
            ):
                raise ValueError(
                    f"{k}: scores={len(s)}, "
                    f"frames={n}; "
                    f"verify sampling/timebase"
                )

            adjustments.append({
                "video": k,
                "scores": int(len(s)),
                "frames": int(n)
            })

            s = np.pad(
                s,
                (0, max(0, n - len(s))),
                mode="edge"
            )[:n]

        # ----------------------------------------------------
        # Build Ground Truth
        # ----------------------------------------------------

        y = build_ground_truth(
            video_name=k,
            annotation=ann[k],
            n_frames=n,
            base=base,
            tolerance=MAX_FRAME_TOLERANCE,
            adjustments=annotation_adjustments
        )

        # Binary predictions
        pred = s >= threshold

        ys.append(y)
        ss.append(s)

        # ----------------------------------------------------
        # Normal-video False Alarm Rate
        # ----------------------------------------------------

        if not ann[k]:

            normal_videos += 1

            normal_fp += int(
                pred.sum()
            )

            normal_n += n

        else:
            abnormal_videos += 1

        # Use the corrected Ground Truth intervals for
        # temporal evaluation as well.
        truth[k] = intervals(
            y.astype(bool)
        )

        # ----------------------------------------------------
        # Temporal anomaly proposals
        # ----------------------------------------------------

        if temporal:

            for a, b in intervals(pred):

                confidence = float(
                    s[a:b].max()
                )

                proposals.append(
                    (confidence, k, a, b)
                )

    # --------------------------------------------------------
    # Aggregate frame-level results
    # --------------------------------------------------------

    Y = np.concatenate(ys)
    S = np.concatenate(ss)

    P = S >= threshold

    if len(np.unique(Y)) != 2:
        raise ValueError(
            "Both classes required"
        )

    tn, fp, fn, tp = map(
        int,
        confusion_matrix(
            Y,
            P,
            labels=[0, 1]
        ).ravel()
    )

    # --------------------------------------------------------
    # Frame-level metrics
    # --------------------------------------------------------

    m = {

        "accuracy": float(
            accuracy_score(Y, P)
        ),

        "precision": float(
            precision_score(
                Y,
                P,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                Y,
                P,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                Y,
                P,
                zero_division=0
            )
        ),

        "auc": float(
            roc_auc_score(Y, S)
        ),

        "ap": float(
            average_precision_score(Y, S)
        ),

        "far_all_negative_frames": (
            fp / (fp + tn)
            if fp + tn else None
        ),

        "far_normal_videos": (
            normal_fp / normal_n
            if normal_n else None
        ),

        "threshold": float(threshold),

        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,

        "n_videos": len(keys),
        "n_normal_videos": normal_videos,
        "n_abnormal_videos": abnormal_videos,

        "n_frames": int(len(Y)),

        "n_normal_frames": int(
            np.sum(Y == 0)
        ),

        "n_abnormal_frames": int(
            np.sum(Y == 1)
        ),

        "annotation_base": base,
        "annotation_frame_tolerance": MAX_FRAME_TOLERANCE,

        "tail_policy": tail,

        "missing_videos": sorted(missing),
        "extra_videos": sorted(extra),

        "tail_adjustments": adjustments,

        "annotation_adjustments": annotation_adjustments,

        "n_annotation_adjustments": len(
            annotation_adjustments
        ),

        "total_clipped_frames": sum(
            item["difference"]
            for item in annotation_adjustments
        ),
    }

    # --------------------------------------------------------
    # Temporal AP and mAP
    # --------------------------------------------------------

    if temporal:

        ap = {
            f"{i / 10:.1f}": temporal_ap(
                proposals,
                truth,
                i / 10
            )
            for i in range(1, 6)
        }

        m["temporal_ap_iou"] = ap

        m["temporal_map_binary"] = (
            float(np.mean(list(ap.values())))
            if all(
                v is not None
                for v in ap.values()
            )
            else None
        )

        m["n_temporal_proposals"] = len(
            proposals
        )

        m["n_ground_truth_intervals"] = sum(
            map(len, truth.values())
        )

        m["temporal_protocol"] = (
            "Binary proposals from contiguous frames "
            ">= fixed threshold; max score confidence; "
            "interpolated AP; not class-wise mAP"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    # JSON metrics
    (out / "test_metrics.json").write_text(
        json.dumps(
            m,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # Frame predictions
    np.savez_compressed(
        out / "predictions.npz",

        y_true=Y,
        y_score=S,

        video_names=keys,

        video_lengths=[
            len(y)
            for y in ys
        ]
    )

    # Annotation adjustment log
    adjustment_path = (
        out / "annotation_adjustments.csv"
    )

    with open(
        adjustment_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        fieldnames = [
            "video",
            "original_start",
            "original_end",
            "converted_start",
            "converted_end",
            "corrected_end",
            "difference"
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            annotation_adjustments
        )

    # Scalar metrics CSV
    with open(
        out / "test_metrics.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "metric",
            "value"
        ])

        for metric, value in m.items():

            if not isinstance(
                value,
                (dict, list)
            ):
                writer.writerow([
                    metric,
                    value
                ])

        if temporal:

            for iou, value in m[
                "temporal_ap_iou"
            ].items():

                writer.writerow([
                    f"temporal_ap@{iou}",
                    value
                ])

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = np.array([
        [tn, fp],
        [fn, tp]
    ])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Normal",
            "Anomaly"
        ]
    )

    display.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title(
        "Frame-level Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        out / "confusion_matrix.png",
        dpi=180
    )

    plt.close()

    # ========================================================
    # ROC CURVE
    # ========================================================

    fpr, tpr, _ = roc_curve(Y, S)

    plt.figure(figsize=(7, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {m['auc']:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        out / "roc_curve.png",
        dpi=180
    )

    plt.close()

    # ========================================================
    # PRECISION-RECALL CURVE
    # ========================================================

    pr, rc, _ = precision_recall_curve(
        Y,
        S
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        rc,
        pr,
        label=f"AP = {m['ap']:.4f}"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        out / "pr_curve.png",
        dpi=180
    )

    plt.close()

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("EVALUATION COMPLETED")
    print("=" * 60)

    excluded = {
        "tail_adjustments",
        "annotation_adjustments",
        "missing_videos",
        "extra_videos"
    }

    print(
        json.dumps(
            {
                k: v
                for k, v in m.items()
                if k not in excluded
            },
            indent=2
        )
    )

    print("=" * 60)
    print(f"Accuracy : {m['accuracy'] * 100:.2f}%")
    print(f"Precision: {m['precision'] * 100:.2f}%")
    print(f"Recall   : {m['recall'] * 100:.2f}%")
    print(f"F1-score : {m['f1'] * 100:.2f}%")
    print(f"ROC-AUC  : {m['auc'] * 100:.2f}%")
    print(f"AP       : {m['ap'] * 100:.2f}%")

    if m["far_normal_videos"] is not None:
        print(
            f"FAR      : "
            f"{m['far_normal_videos'] * 100:.2f}%"
        )

    if (
        temporal
        and m["temporal_map_binary"] is not None
    ):
        print(
            f"Temp mAP : "
            f"{m['temporal_map_binary'] * 100:.2f}%"
        )

    print("=" * 60)
    print("Results saved to:", out)
    print("=" * 60)

    return m


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    p = argparse.ArgumentParser(
        description="Evaluate UCF-Crime anomaly scores"
    )

    p.add_argument(
        "--scores",
        required=True
    )

    p.add_argument(
        "--ann-txt",
        required=True
    )

    g = p.add_mutually_exclusive_group(
        required=True
    )

    g.add_argument("--videos")
    g.add_argument("--frames-csv")

    p.add_argument(
        "--out-dir",
        required=True
    )

    p.add_argument(
        "--threshold",
        type=float,
        default=0.5
    )

    p.add_argument(
        "--annotation-base",
        type=int,
        choices=[0, 1],
        default=1
    )

    p.add_argument(
        "--tail",
        choices=["strict", "edge"],
        default="strict"
    )

    p.add_argument(
        "--max-tail",
        type=int,
        default=15
    )

    p.add_argument(
        "--allow-subset",
        action="store_true"
    )

    p.add_argument(
        "--temporal-map",
        action="store_true"
    )

    a = p.parse_args()

    if not 0 <= a.threshold <= 1:
        p.error(
            "threshold must be in [0,1]"
        )

    with np.load(
        a.scores,
        allow_pickle=False
    ) as z:

        scores = {
            name(k): z[k]
            for k in z.files
        }

        if len(scores) != len(z.files):
            raise ValueError(
                "Duplicate normalized video keys"
            )

    evaluate(
        scores=scores,

        ann=annotations(
            a.ann_txt
        ),

        lengths=frame_lengths(
            a.videos,
            a.frames_csv
        ),

        out=a.out_dir,

        threshold=a.threshold,
        base=a.annotation_base,
        tail=a.tail,
        allow_subset=a.allow_subset,
        temporal=a.temporal_map,
        max_tail=a.max_tail
    )


if __name__ == "__main__":
    main()
