# Wilson Loop Datasets

Gauge configurations are not stored in the repository (too large). Download or generate them as described below.

## Available Ensembles

### `test_small` — for development and testing
- Lattice: 8^3 x 16
- beta: 5.8
- Action: Wilson gauge
- Configs: 100
- Purpose: fast validation, unit tests

### `benchmark` — for official scoring
- Lattice: 20^3 x 40
- beta: 6.0
- Action: Wilson gauge
- Configs: 4000
- Purpose: benchmark scoring and leaderboard

## Generating Configs

Use the provided script:

```bash
python tasks/wilson_loop/scripts/generate_configs.py \
    --latt 8,8,8,16 --beta 5.8 --n_configs 100 \
    --output tasks/wilson_loop/dataset/
```

## File Format

Configs are stored as numpy `.npy` files with shape `(Nd, Lx, Ly, Lz, Lt, Nc, Nc)` complex128, where Nd=4 (x,y,z,t directions), Nc=3 (SU(3) colors).
