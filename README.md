# SPSM

- `spectral_prior/generate_prior.py`: signed spectral-prior estimation.
- `spsm/spsm.py`: SPSM with grayscale reconstruction and a trainable Conv-BN-SiLU projection.
- `noise_pipeline/`: synthetic degradation pipeline. `degrade_image.py` contains the paper's nominal configuration and component-wise strength settings.
- [datasets.md](datasets.md): dataset download links.

Seeds used in the paper: `0` for training; `42` for noise generation on RS-AOD, ShipRSImageNet, and Table IV. Images are sorted by filename, and image `i` uses seed `42 + i` (zero-based). Full-data prior estimation is deterministic and does not use random sampling.
