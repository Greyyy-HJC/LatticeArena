# gsfit_2pt Dataset

`gsfit_2pt` uses deterministic synthetic pion two-point samples.

The synthetic dataset logic lives in `dataset/synthetic.py`, and a saved archive
can be generated with:

```bash
python tasks/gsfit_2pt/scripts/generate_fake_data.py
```

By default this writes:

- `tasks/gsfit_2pt/dataset/fake_data.npz`

Each case stores:

- correlated bootstrap/jackknife-like samples
- known true energies
- known amplitudes

You can customize the generated archive:

```bash
python tasks/gsfit_2pt/scripts/generate_fake_data.py \
  --output tasks/gsfit_2pt/dataset/fake_data_small.npz \
  --num-samples 12 \
  --noise-multiplier 0.5 \
  --lt 48
```

Use a saved archive in the benchmark:

```bash
python tasks/gsfit_2pt/benchmark/run.py \
  --submission plain \
  --dataset-file tasks/gsfit_2pt/dataset/fake_data.npz
```
