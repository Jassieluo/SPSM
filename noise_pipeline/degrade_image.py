"""Deterministic, auditable synthetic degradation pipeline for remote-sensing images."""

from __future__ import annotations

import argparse
import json
import math
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

try:  # Support both package imports and direct execution from this directory.
    from .add_noisy_PSF import add_atmospheric_blur
    from .add_noisy_perlin_cloud import add_clouds
    from .add_noisy_sensor import add_sensor_noise
    from .add_noisy_stripe import add_stripe_noise
except ImportError:  # pragma: no cover - used by the command-line entry point
    from add_noisy_PSF import add_atmospheric_blur
    from add_noisy_perlin_cloud import add_clouds
    from add_noisy_sensor import add_sensor_noise
    from add_noisy_stripe import add_stripe_noise


# Paper settings: RS-AOD / ShipRSImageNet use full degradation at strength 1.0.
# Table IV uses each isolated stage and the full pipeline at these three strengths.
# Sorted image i uses PAPER_SEED + i (zero-based); worker count does not change it.
PAPER_SEED = 42
PAPER_STRENGTHS = (0.8, 1.0, 1.2)
PAPER_CONDITIONS = ("stripe", "cloud", "psf", "sensor", "full")

DEFAULT_CONFIG: dict[str, Any] = {
    "stripe": {
        "intensity": 40.0,
        "angle_range": [-8.0, 8.0],
        "noise_type": "random",
        "density": 0.20,
        "width_range": [1, 3],  # Upper bound excluded: widths are 1 or 2 pixels.
    },
    "cloud": {"density": 0.5, "scale": 140.0},
    "psf": {"sigma": 1.0, "kernel_size": 13},
    "sensor": {"gaussian_sigma": 15.0, "apply_poisson": True, "photon_scale": 100.0},
    "order": ["stripe", "cloud", "psf", "sensor"],
    "enable_noises": ["stripe", "cloud", "psf", "sensor"],
}
# Other fixed settings are implemented in the corresponding component functions:
# stripe pixel jitter sigma=2; cloud octaves=6, persistence=0.55, sigmoid slope=8,
# cloud smoothing sigma=3 and BGR color=(1.0, 0.96, 0.92); PSF reflection borders.


def paper_config(condition: str = "full", strength: float = 1.0) -> dict[str, Any]:
    """Return the paper's nominal or component-wise degradation configuration.

    strength is alpha: stripe intensity=40*alpha, cloud opacity=0.5*alpha,
    blur sigma=alpha, sensor read sigma=15*alpha and photon scale=100/alpha.
    All remaining parameters stay fixed. Each call returns an independent copy.
    """
    if condition not in PAPER_CONDITIONS:
        raise ValueError(f"Unknown degradation condition: {condition}")
    strength = float(strength)
    if not math.isfinite(strength) or strength <= 0:
        raise ValueError("The degradation strength must be finite and positive")
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if condition in {"stripe", "full"}:
        config["stripe"]["intensity"] *= strength
    if condition in {"cloud", "full"}:
        config["cloud"]["density"] *= strength
    if condition in {"psf", "full"}:
        config["psf"]["sigma"] *= strength
    if condition in {"sensor", "full"}:
        config["sensor"]["gaussian_sigma"] *= strength
        config["sensor"]["photon_scale"] /= strength
    config["enable_noises"] = list(config["order"]) if condition == "full" else [condition]
    return config


def calculate_snr(original: np.ndarray, noisy: np.ndarray) -> float:
    """Return image-level signal-to-noise ratio in dB after all degradations."""
    original_f = original.astype(np.float32)
    noisy_f = noisy.astype(np.float32)
    signal_power = float(np.mean(original_f**2))
    noise_power = float(np.mean((noisy_f - original_f) ** 2))
    if noise_power <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(max(signal_power, 1e-12) / noise_power))


def _copy_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    if config:
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            else:
                merged[key] = value
    return merged


def _sample_parameters(config: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Sample per-image parameters once, so they can be saved in the manifest."""
    stripe = dict(config["stripe"])
    low, high = stripe.pop("angle_range", [-8.0, 8.0])
    stripe["angle"] = float(rng.uniform(float(low), float(high)))
    stripe["width_range"] = tuple(int(x) for x in stripe.get("width_range", [1, 3]))
    return {
        "stripe": stripe,
        "cloud": dict(config["cloud"]),
        "psf": dict(config["psf"]),
        "sensor": dict(config["sensor"]),
    }


def degrade_image(
    img: np.ndarray,
    stripe_params: dict[str, Any] | None = None,
    cloud_params: dict[str, Any] | None = None,
    psf_params: dict[str, Any] | None = None,
    sensor_params: dict[str, Any] | None = None,
    order: list[str] | None = None,
    enable_noises: list[str] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Apply the documented degradation sequence with one per-image RNG."""
    rng = np.random.default_rng(seed)
    order = list(order or DEFAULT_CONFIG["order"])
    enabled = set(enable_noises or DEFAULT_CONFIG["enable_noises"])
    if "all" in enabled:
        enabled = set(DEFAULT_CONFIG["enable_noises"])

    known_steps = set(DEFAULT_CONFIG["order"])
    unknown_steps = (set(order) | set(enable_noises or [])) - known_steps - {"all"}
    if unknown_steps:
        raise ValueError(f"Unknown degradation step(s): {sorted(unknown_steps)}")
    if enabled - set(order):
        raise ValueError(f"Enabled degradation steps missing from order: {sorted(enabled - set(order))}")

    for step in order:
        if step not in enabled:
            continue
        if step == "stripe" and stripe_params:
            img = add_stripe_noise(img, rng=rng, **stripe_params)
        elif step == "cloud" and cloud_params:
            img = add_clouds(img, rng=rng, **cloud_params)
        elif step == "psf" and psf_params:
            img = add_atmospheric_blur(img, **psf_params)
        elif step == "sensor" and sensor_params:
            img = add_sensor_noise(img, rng=rng, **sensor_params)
    return img


def _process_one(task: tuple[str, str, dict[str, Any], list[str], list[str], int]) -> dict[str, Any]:
    src_name, dst_name, config, order, enabled, seed = task
    random.seed(seed)
    np.random.seed(seed)
    src = Path(src_name)
    dst = Path(dst_name)
    image = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if image is None:
        return {"file": src.name, "ok": False, "error": "imread", "seed": seed}

    params = _sample_parameters(config, np.random.default_rng(seed))
    noisy = degrade_image(
        image,
        stripe_params=params["stripe"],
        cloud_params=params["cloud"],
        psf_params=params["psf"],
        sensor_params=params["sensor"],
        order=order,
        enable_noises=enabled,
        seed=seed,
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), noisy):
        return {"file": src.name, "ok": False, "error": "imwrite", "seed": seed}
    return {
        "file": src.name,
        "ok": True,
        "seed": seed,
        "parameters": params,
        "snr_db": calculate_snr(image, noisy),
    }


def batch_degrade_images(
    src_dir: str | Path,
    dst_dir: str | Path,
    config: dict[str, Any] | None = None,
    seed: int = PAPER_SEED,
    max_workers: int | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a sorted, reproducible image set and write an audit manifest."""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    cfg = _copy_config(config)
    files = sorted(
        p
        for p in (src_dir.iterdir() if src_dir.exists() else [])
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )
    if not files:
        raise FileNotFoundError(f"No supported images found in {src_dir}")

    tasks = [
        (str(src), str(dst_dir / src.name), cfg, cfg["order"], cfg["enable_noises"], seed + idx)
        for idx, src in enumerate(files)
    ]
    worker_count = max_workers or min(16, max(1, len(tasks)))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        rows = list(tqdm(executor.map(_process_one, tasks), total=len(tasks), desc="Generating degradation"))

    snr_values = [row["snr_db"] for row in rows if row.get("ok") and np.isfinite(row["snr_db"])]
    manifest = {
        "source_dir": str(src_dir),
        "output_dir": str(dst_dir),
        "seed": seed,
        "num_images": len(files),
        "num_success": sum(bool(row.get("ok")) for row in rows),
        "config": cfg,
        "snr_db": {
            "min": float(min(snr_values)) if snr_values else None,
            "max": float(max(snr_values)) if snr_values else None,
            "mean": float(np.mean(snr_values)) if snr_values else None,
        },
        "files": rows,
    }
    manifest_path = Path(manifest_path) if manifest_path else dst_dir.parent / f"{dst_dir.name}_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src_dir", type=Path)
    parser.add_argument("dst_dir", type=Path)
    parser.add_argument("--seed", type=int, default=PAPER_SEED)
    parser.add_argument("--condition", choices=PAPER_CONDITIONS, default="full")
    parser.add_argument("--strength", type=float, choices=PAPER_STRENGTHS, default=1.0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    result = batch_degrade_images(
        args.src_dir,
        args.dst_dir,
        config=paper_config(args.condition, args.strength),
        seed=args.seed,
        max_workers=args.workers,
        manifest_path=args.manifest,
    )
    print(json.dumps({"num_images": result["num_images"], "snr_db": result["snr_db"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
