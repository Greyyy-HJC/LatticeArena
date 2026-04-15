# Task: Pion 2pt (boosted)

优化目标：设计更好的 **pion interpolating operator**，在 **有动量的 pion**（重点示例 `|p| ~ 1 GeV`）上得到：

- 更高信噪比（SNR）
- 更小激发态污染（更早出现稳定平台）

## Physics target

测量的关联函数为：

\[
C_\pi(\mathbf{p}, t) = \langle O_\pi(\mathbf{p}, t_0+t) O_\pi^\dagger(\mathbf{p}, t_0) \rangle,
\quad
O_\pi(\mathbf{p}, t) = \sum_{\mathbf{x}} e^{i\mathbf{p}\cdot\mathbf{x}} \bar d(\mathbf{x},t)\,\Gamma\,u(\mathbf{x},t).
\]

你需要优化 `O_\pi` 的构造（例如 smearing profile、动量注入策略、Dirac 结构、位移/导数结构），从而改善 boosted pion 的有效质量平台和统计精度。

## What you implement

实现 `tasks/pion_2pt/interface.py` 中的 `PionInterpolatingOperator`：

- `setup(...)`：每个规范场一次性的预处理
- `build(...)`：对给定动量（GeV）和源时刻返回 source/sink profile 与 Dirac 结构

## Baseline

`operators/plain.py` 提供了一个简单 baseline：

- Gaussian 空间 profile
- plane-wave 相位注入动量
- `Gamma = gamma_5`

## Suggested benchmark metrics

- **SNR at fixed t**: `mean(C)/std(C)`
- **Plateau onset**: 有效质量进入平台的最早 `t_min`
- **Excited-state contamination**: 2-state fit 中 excited-state 振幅
- **Dispersion consistency**: `E(p)` 与色散关系的一致性

## Quick check

```bash
pytest tasks/pion_2pt/tests/
python tasks/pion_2pt/benchmark/run.py --operator plain
```
