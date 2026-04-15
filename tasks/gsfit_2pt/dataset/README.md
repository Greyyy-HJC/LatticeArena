# gsfit_2pt Dataset

The v1 `gsfit_2pt` benchmark uses deterministic synthetic pion 2pt samples.
You can generate a concrete fake dataset archive directly inside this folder:

```bash
python tasks/gsfit_2pt/scripts/generate_fake_data.py
```

By default this writes:

- `tasks/gsfit_2pt/dataset/fake_data.npz`

The archive contains multiple pion-like correlator cases, each with:

- bootstrap/jackknife-like samples
- known true energies
- known amplitudes

You can also customize the generated fake data:

```bash
python tasks/gsfit_2pt/scripts/generate_fake_data.py \
  --output tasks/gsfit_2pt/dataset/fake_data_small.npz \
  --num-samples 12 \
  --noise-multiplier 0.5 \
  --lt 48
```

To benchmark against a saved fake dataset instead of regenerating in memory:

```bash
python tasks/gsfit_2pt/benchmark/run.py --operator plain \
  --dataset-file tasks/gsfit_2pt/dataset/fake_data.npz
```

Future extension:

- drop real example correlator samples into this directory
- add a second benchmark mode that compares against a trusted reference fit
- keep the public submission interface unchanged
