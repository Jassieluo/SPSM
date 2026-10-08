"""Estimate a fixed signed grayscale spectral prior from clean YOLO-labelled images."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output signed prior .npy path")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--class-id", type=int, default=None, help="Use only one YOLO class; default uses every class")
    parser.add_argument("--max-images", type=int, default=None, help="Optional deterministic limit after sorting")
    return parser.parse_args()


def iter_image_paths(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    return sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def read_yolo_boxes(label_path: Path, width: int, height: int, class_id: int | None) -> list[tuple[int, int, int, int]]:
    """Read all valid normalized YOLO boxes and clip them to the image."""
    boxes: list[tuple[int, int, int, int]] = []
    try:
        lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return boxes
    for line in lines:
        fields = line.split()
        if len(fields) < 5:
            continue
        try:
            cls, xc, yc, bw, bh = map(float, fields[:5])
        except ValueError:
            continue
        if not np.isfinite([cls, xc, yc, bw, bh]).all() or bw <= 0 or bh <= 0:
            continue
        if class_id is not None and int(round(cls)) != class_id:
            continue
        x1 = max(0, min(width - 1, int(np.floor((xc - bw / 2.0) * width))))
        y1 = max(0, min(height - 1, int(np.floor((yc - bh / 2.0) * height))))
        x2 = max(x1 + 1, min(width, int(np.ceil((xc + bw / 2.0) * width))))
        y2 = max(y1 + 1, min(height, int(np.ceil((yc + bh / 2.0) * height))))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def conjugate_partner(field: np.ndarray) -> np.ndarray:
    """Return the conjugate-frequency partner in an even-sized fftshift grid."""
    # For an even HxW grid, fftshifted index (i,j) pairs with (-i mod H,
    # -j mod W), implemented by flip followed by a one-pixel cyclic shift.
    return np.roll(np.roll(np.flip(np.flip(field, axis=0), axis=1), 1, axis=0), 1, axis=1)


def process_one(task: tuple[str, str, int, int, int | None]) -> tuple[np.ndarray, dict] | None:
    image_name, label_name, width, height, class_id = task
    image_path, label_path = Path(image_name), Path(label_name)
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    h0, w0 = bgr.shape[:2]
    boxes = read_yolo_boxes(label_path, w0, h0, class_id)
    if not boxes:
        return None

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    full = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    mask = np.zeros((h0, w0), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = 1
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    target = full * mask

    # Estimator: Re(T * conjugate(U_g)), where U_g is the full-image phase
    # carrier. This matches the runtime's fftshift and real(ifft(...)).
    full_fft = np.fft.fftshift(np.fft.fft2(full))
    target_fft = np.fft.fftshift(np.fft.fft2(target))
    full_mag = np.abs(full_fft)
    carrier = np.ones_like(full_fft, dtype=np.complex128)
    np.divide(full_fft, full_mag, out=carrier, where=full_mag > EPS)
    coefficient = np.real(target_fft * np.conj(carrier))
    coefficient = 0.5 * (coefficient + conjugate_partner(coefficient))
    if not np.isfinite(coefficient).all():
        return None

    stats = {
        "image": image_path.name,
        "boxes": len(boxes),
        "full_energy": float(np.mean(full * full)),
        "target_energy": float(np.mean(target * target)),
    }
    return coefficient.astype(np.float64), stats


def estimate(tasks: Iterable[tuple[str, str, int, int, int | None]], workers: int) -> tuple[np.ndarray, list[dict], int]:
    tasks = list(tasks)
    shape = (tasks[0][3], tasks[0][2])
    total = np.zeros(shape, dtype=np.float64)
    records: list[dict] = []
    skipped = 0
    if workers <= 1:
        results = map(process_one, tasks)
        for result in tqdm(results, total=len(tasks), desc="Estimating SPSM prior"):
            if result is None:
                skipped += 1
                continue
            coeff, stats = result
            total += coeff
            records.append(stats)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = pool.map(process_one, tasks, chunksize=1)
            for result in tqdm(results, total=len(tasks), desc=f"Estimating SPSM prior ({workers} workers)"):
                if result is None:
                    skipped += 1
                    continue
                coeff, stats = result
                total += coeff
                records.append(stats)
    if not records:
        raise RuntimeError("No valid image/label pair was processed")
    return total / len(records), records, skipped


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive")
    if args.width % 2 or args.height % 2:
        raise ValueError("width and height must be even for the conjugate-symmetry projection")

    image_paths = iter_image_paths(args.image_dir)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    tasks = [
        (str(image_path), str(args.label_dir / f"{image_path.stem}.txt"), args.width, args.height, args.class_id)
        for image_path in image_paths
        if (args.label_dir / f"{image_path.stem}.txt").is_file()
    ]
    if not tasks:
        raise RuntimeError(f"No image/label pairs found under {args.image_dir} and {args.label_dir}")

    prior, records, skipped = estimate(tasks, max(1, args.workers))
    if not np.isfinite(prior).all():
        raise FloatingPointError("The estimated prior contains NaN or Inf")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, prior.astype(np.float32))
    metadata = {
        "method": "mean Re(T * conjugate(U_g)) with conjugate-symmetry projection",
        "runtime_convention": "fftshift(fft2(gray/255)); real(ifft2(ifftshift(A*U)))",
        "target_proxy": "union of all valid YOLO bounding boxes; resized grayscale image multiplied by nearest-neighbor mask",
        "input_scale": "uint8 grayscale divided by 255",
        "posthoc_max_normalization": False,
        "width": args.width,
        "height": args.height,
        "class_id": args.class_id,
        "workers": args.workers,
        "candidate_images": len(tasks),
        "processed_images": len(records),
        "skipped_images": skipped,
        "mean_boxes_per_processed_image": float(np.mean([r["boxes"] for r in records])),
        "prior_min": float(prior.min()),
        "prior_max": float(prior.max()),
        "prior_mean": float(prior.mean()),
        "prior_median": float(np.median(prior)),
        "prior_l2": float(np.linalg.norm(prior)),
        "mean_full_energy": float(np.mean([r["full_energy"] for r in records])),
        "mean_target_energy": float(np.mean([r["target_energy"] for r in records])),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"prior": str(args.output), "metadata": str(metadata_path), **metadata}, indent=2))


if __name__ == "__main__":
    main()
