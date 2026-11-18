# Executable S16T16 pion projection pilot

The scientific definition is in [the task specification](../../doc/pion_projection.md).
This directory implements a **local research pilot**, including fixed measurement,
cache verification, baseline search, a frozen agent submission, and held-out
diagnostic evaluation. It is not a secure multi-user competition service.

The first run is recorded in [the results report](../../doc/pion_projection_results.md).
The reference checks failed; the software consequently emits `score: null`.
Raw losses and uncertainty intervals are still useful diagnostics.

> **2026-11-18 audit and contract fixes.** The reference model identity is now
> frozen explicitly (`reference.py`: schema `pion-reference-v1`, kind plus the
> hashes of `statistics.py` and the selected implementation). `develop` writes
> it into `submission.json`; `evaluate` and `candidate` must use that frozen
> kind and refuse on a hash mismatch. Submissions predating the field can only
> be replayed read-only with `LATTICEARENA_LEGACY_REFERENCE=1`. The reflection
> gate no longer hard-codes `T=16`: it builds `C(t)-C(T-t)` for any even
> temporal extent. The earlier two-state / T=32 / calibrated-p-value studies
> remain diagnostics in `doc/pion_projection_results.md`; the absolute score on
> this ensemble is still not trustworthy (reference-model systematics are as
> large as the effect). No new GPU measurement was run in this audit.
> A read-only audit of the archived runs (shapes, hashes, reference scan,
> covariance spectra, corrected T=32 reflection) is in
> `results/pion_projection/score_diagnostics_v2/`; it shows the 5-point
> reference covariance is rank deficient (15 dimensions vs 12 blocks) and that
> the archived T=32 reflection z of 274.27 was a hard-coded-T artefact
> (corrected 2.40), while the T=32 fit and window-shift failures remain.

## Reproduce the run

From the repository root, after setting up [the environment](../../doc/environment.md):

```bash
.venv/bin/python -m pytest -q tests/test_pion_projection.py
bash scripts/run_pion.sh --output results/pion_projection/reproduction measure --split development
bash scripts/run_pion.sh --output results/pion_projection/reproduction develop
bash scripts/run_pion.sh --output results/pion_projection/reproduction measure --split validation
bash scripts/run_pion.sh --output results/pion_projection/reproduction evaluate
bash scripts/run_pion.sh --output results/pion_projection/reproduction plot
```

`--output` precedes the subcommand. Use a fresh output directory: `develop` refuses
to overwrite a frozen submission or reopen development after validation data
exist. Reproduction does not make these previously inspected configurations a
new hidden test set. `plot` uses recorded results and needs no new inversion.

Small result arrays, provenance, frozen proposals, and figures are stored under
`results/`. The approximately 6.4 GiB of point propagators for 45 configurations
live under ignored `.cache/pion_projection/propagators/`; no gauge files are copied
into the repository. Cache loading checks the gauge/software provenance, array
checksum, and, in the fixed measurement runner, all 12 explicit Dirac residuals.

## Fixed v1 protocol

`protocol.json` is the numerical source of truth and its hash accompanies data
and submissions. The protocol was fixed before the first development measurements.

| Setting | v1 value |
| --- | --- |
| Development / gap / validation IDs | 0–29 / 30–34 / 35–49 |
| Valence action | Wilson, kappa 0.14, periodic space, antiperiodic time |
| Source / residual / iteration limit | Origin, 12 spin-color columns, 1e-10, 10000 |
| Diffusion step | `H = (I + 0.2 * spatial_hopping) / 2.2` |
| Measured kernels | `H^n`, n = 0, 2, 4, 8, 16, 32, 64 |
| Score / amplitude windows | 2–5 / 6–8, folded per configuration |
| Common reference channels | n = 0, 8, 16; separate amplitudes, common mass |
| Reference windows | 4–8, 5–8 (nominal), 6–8 |
| Covariance | 25% diagonal shrinkage, eigenvalue floor 1e-10 times largest eigenvalue |
| Mass search | Bounded scalar minimization, 0.1 < am < 3.0 |
| Blocks / bootstrap | Contiguous blocks of 5; 1000 bootstrap draws |
| Score uncertainty | 200 paired outer draws, each with 128 inner draws |
| Seeds | 20261109 development; +1 validation; +2 nested intervals |
| Amplitude gate | Absolute amplitude / bootstrap SE at least 3 |
| Reference gates | Relative mass SE <= 5%; approximate fit p >= .01; paired window shifts <= 2.5 SE; no mass-bound hits |
| Reflection gate | Maximum absolute forward/backward block-mean z <= 4 over basis channels and t = 1–7 |
| Per-configuration budget | 300 seconds, checked after completion; CuPy pool limit 10 GiB |

The covariance is computed from block means. Reference fitting fixes its metric
from development; each candidate's late-amplitude metric is also fixed from its
development data before validation. Bootstrap samples vary means, reference mass,
and fitted amplitudes using these fixed metrics. Thus the intervals are conditional
on the declared fitting metrics, not a marginalization over all covariance
estimators. Shrinkage chi-square p-values are diagnostic approximations, not exact
finite-sample probabilities. The paired mass-window check uses the uncertainty of
the *difference* on common bootstrap draws.

The five-configuration block length was conservatively fixed; SEs for lengths
1, 2, 3, and 5 are reported on development. Trajectory spacing is not known and
decorrelation has not been established. Only six development blocks and three
validation blocks remain. Nested intervals include the statistical-variance term
in each outer loss; no replica is discarded. Replicas with repeated identical
blocks remain evaluable because fit metrics are fixed. These intervals are
necessarily coarse with three validation blocks.

The reference failure is a gate on scores, not a reason to search for more
favorable windows after looking at validation. Candidate failures produce an
explicit error or invalid record; incomplete measurement checkpoints cannot be
loaded as complete splits.

## Agent search and actual operator

The baseline search evaluates all seven diffusion powers on development and
selects the lowest loss. `H^(2k)` is the sink bilinear obtained by smearing both
quark legs with `H^k`; its contraction equals `||H^k S||^2`. QUDA evaluates these
norms incrementally, avoiding redundant smearing and inversions.

For the agent attempt, let `A_n` be the fitted development amplitude of basis n.
Optimize the actual linear kernel

\[
K=\sum_n \frac{u_n}{A_n}H^n,\qquad \sum_n u_n=1,\quad \sum_n |u_n|\le 8.
\]

The search minimizes development loss plus `0.001 * sum(u_n^2)` using SLSQP,
at most 500 iterations per start. The eight deterministic starts are the seven
basis vectors and their uniform average. The regularizer is a search choice and
is **not** added to the reported task loss. All starts, hyperparameters, and
weights are frozen before validation. An overall normalization has no effect on
the task score.

`submission.json` records raw coefficients, normalized coefficients, development
amplitudes, selected baseline, fitting metrics, protocol/data/source hashes, and
reference diagnostics. Applying the same coefficients to configuration-level
correlators is exact by linearity; it is not a time-dependent alteration of the
estimator. The generic kernel route below independently applies the actual
operator to propagators.

## Submit an arbitrary spatial kernel

The implemented v1 channel requires a **real Hermitian, spin-scalar** observable.
Signed kernels and negative correlators are permitted. The broader complex
non-Hermitian scoring extension in the scientific specification is not implemented
here and is rejected, rather than silently discarding a physical imaginary part.

A Python module exposes `build_kernel(spatial_links, parameters)` returning an
object with `apply(field)`. Spatial links are CuPy arrays `(3,z,y,x,3,3)`, ordered
x, y, z; a field is `(z,y,x,3,nrhs)`. The returned field has the same shape.
The evaluator provides only the current time slice's spatial links, freezes
parameters, and owns propagator loading, contraction, folding, and scoring.
The same constructor is used at every time and configuration.

The generic runner takes this JSON definition (use actual module/parameters):

```json
{
  "builder": "tasks.pion_projection.kernels:build_kernel",
  "parameters": {
    "alpha": 0.2,
    "powers": [0, 8, 32],
    "weights": [-0.2, 0.5, 0.7]
  }
}
```

```bash
bash scripts/run_pion.sh candidate --definition my_candidate.json --split development
bash scripts/run_pion.sh candidate --definition my_candidate.json --split validation
```

These commands use the complete cached propagator manifests from the main run.
Development application freezes the definition, builder hash, correlators, and
late-fit metric under `results/.../candidates/<hash>/`; validation verifies that
freeze and evaluates against the same common reference and local baseline,
including nested uncertainty estimates. User modules run as trusted local code:
kernel covariance, linearity, translation/cubic symmetry, and absence of hidden
data access remain required and subject to review/testing. This is not a Python
sandbox or an automatic proof of those properties. Time limits are cooperative
checks; the CuPy limit does not cap QUDA's separately allocated memory.

The real submitted operator for this run is in
[`agent_candidate.json`](../../results/pion_projection/s16t16_v1/agent/llm_candidate.json).
For the diffusion family, a quick independent operator check uses the first
configuration of each split and compares the generic contraction against the
previously measured basis combination:

```bash
bash scripts/run_pion.sh candidate --definition results/pion_projection/s16t16_v1/agent/llm_candidate.json --split development --verify-only
bash scripts/run_pion.sh candidate --definition results/pion_projection/s16t16_v1/agent/llm_candidate.json --split validation --verify-only
```

The generic matrix-free Python route is slower than the QUDA diffusion fast path;
these checks do not retrain the proposal. Reusing the now-public validation set
for subsequent proposals must be disclosed and cannot support fresh held-out claims.

## 2026-11-18 审计状态

| 项 | 状态 |
| --- | --- |
| reference spec（kind + 源码哈希）冻结 | 已实现（`reference.py`，schema `pion-reference-v1`） |
| `evaluate`/`candidate` 校验冻结 spec | 已实现；旧 submission 需 `LATTICEARENA_LEGACY_REFERENCE=1` 只读重放 |
| 反射检查任意偶数 T | 已实现；零 SE 时返回 0 而不是 NaN |
| 合成 develop→evaluate 端到端 | 通过（CPU，见本轮提交说明） |
| legacy submission 只读重放 | 已用 `LATTICEARENA_LEGACY_REFERENCE=1` 验证：无 flag 时拒绝，有 flag 时精确复现归档 risk（local 0.31267、tuned 0.01256、SLSQP 0.00770） |
| 归档只读诊断（inventory/参考扫描/协方差谱/T32 反射修正） | 已生成 `results/pion_projection/score_diagnostics_v2/` |
| 协方差 shrinkage/block 长度预声明扫描 | 已生成 `stability_scan.json`：网格内没有组合能让名义参考通过（max p=0.0072） |
| 两态拟合参数报告 | 已返回并保存 $m_1$、gap、幅度、normal-matrix 条件数、边界标记 |
| block 级 p 值校准 | 已尝试，**不可用**：12 blocks / 15 维导致零分布循环偏窄（p 退化为 $1/(N+1)$），详见诊断 README |
| 模型平均（AIC/BIC） | 已生成 `model_average.json`：权重退化为两态（$\approx1$），但两态过拟合，排序随模型翻转 |
| 两态 profile likelihood | 1σ 区间 $[1.025,1.065]$，宽度 0.04 ≈ 4× bootstrap SE——存在 sloppy 方向 |
| 新数据 / 新 blind validation / anchor | **部分完成**：200 组态 S16T16C（120 dev / 60 val）已测量；参考仍失败、GEVP 平台仍限 $t\le5$；C2 完整算符集与 C3 anchor 仍 NOT_RUN |
| C2 GEVP | 合成验证通过；真实数据 2×2（$E_0=1.067\pm0.010$）与补测 $H^{24}$ 后的 3×3（dev 1.026±0.017 / val 0.997±0.018）**都无平台** → Gram GEVP（2–3 族算符）在 $t=3$–$5$ 给出 $m_0\approx1.02$–$1.05$，偏向两态参考（1.057）而非单 cosh（1.086）；$t=6$ 噪声大，仍不足以认证 |
| C3 能量目标 | 合成数据验证通过（`energy_target.py`，认证/拒绝/无差别三种场景）；真实数据 NOT_RUN（需要独立 anchor） |
| S1 评分证书模块 | 已实现 `scoring.py` + `score_spec.example.json` + 合成测试（真实数据仍无法认证） |
| C3 anchor 校准 | 新增 `anchor_calibration.py` + `results/pion_projection/anchor_calibration_v1/`：block-draw 能量 + 独立共享 anchor；naïve 每-dataset 证书在 null 下假认证 27–50%，独立 dataset 均值/SE 才用于物理推断；Nboot 50→800 使 naïve 区间宽度 0.032→0.0099 |
| C2 错误激发态 | 纯激发态世界：7/8 gate 通过且质量 1.50（独立 guard 7/7 标出）；single-cosh gate 本身无法区分 E1/E0 |
| C2 J/K 族 | `run_family`（J=2、K=2、alpha 分摊）：null 0/7 被认证、better 1/7（功效 0.14）；族错误控制但功效受 N 限制 |
| C2 证书端到端校准 | `certificate_calibration_v1/`：blocks 6/12/24 扫描；no_improvement 假认证 3/8（6 blocks）、0/7、0/8；known_improvement 功效 0.75–0.875；失配全 abstain |
| A1 证书复用 | `evaluate` 在 `<output>/joint/` 存在且 manifest 与 run 身份一致时复用 joint 的证书（`certification_source='joint'`）；不一致则 abstain。测试覆盖 |
| C2 错误激发态 | 纯激发态世界：7/8 gate 通过且质量 1.50（独立 guard 7/7 标出）；single-cosh gate 本身无法区分 E1/E0 |
| C2 J/K 族 | `run_family`（J=2、K=2、alpha 分摊）：null 0/7 被认证、better 1/7（功效 0.14）；族错误控制但功效受 N 限制 |
| C2 证书端到端校准 | 新增 `certificate_calibration.py` + `results/pion_projection/certificate_calibration_v1/`：每 dataset 走 dev/val→gate→joint→certificate；no_improvement 假认证 0.067（1/15）、known_improvement 功效 1.0、misspecified 全 abstain；blocks/JK 扫描与错误激发态场景 NOT_RUN |
| C1 参考 gate 校准 | 新增 `reference_gate.py` + `reference_gate_calibration_v1/`：dof=8（旧脚本 10）、每 dataset null 校准、N_dataset=100/N_null=50、cfg/time 噪声分开；正确模型假拒绝 0.02/0.03、失配功效 0.66/0.59 |
| B2 GEVP 失败计数 | `summarise` 用固定平台时间集合：任一时间缺失 → 整 replica 失败；输出 local/replica 失败与有效时间；cosh/rank 失败记入 reasons；`gram_stats_v2/` 重跑（forward 1.0629±0.0075，0 失败）；反例测试 `partial_plateaus` |
| B1 CV 冻结度量 | `nested_cv_v2/`：每个 training fold 计算 `late_metric` 并传给 held-out risk；fold_plan 保存 train/test IDs、late hash、protocol hash；5 折 local/SLSQP=480.6±146.8（折间散布，不是独立 SE）；明确定名为 outer cross-fit |
| P1.4 全链 nested CV | 新增 `nested_cv.py` + `results/pion_projection/nested_cv_v1/`：5 折、每折重拟合参考/标度/协方差/SLSQP；训练折 $m_0=1.074$–$1.078$，local/SLSQP=209–750（平均 427±89）；学习核重训 NOT_RUN（GPU） |
| P1.1/P1.2 双层校准 | 新增 `calibration_full.py` + `results/pion_projection/score_calibration_v1/`：完整估计器 24 blocks 假阳性 0.025，fixed-metric 0.095（反保守），6/12 blocks 时 0.235/0.085；两态真值拒绝率约 0.53；$(k+1)/(N+1)$ 分辨率约 0.0025 |
| P1.3 120-dev 参考诊断 | 新增 `diagnostics_200.py` + `results/pion_projection/score_diagnostics_v3/`：协方差谱（15/12/9 维，rank 满、条件数 $10^6$–$10^8$）、block×shrinkage 全失败、$m_0$ 随窗口从 1.44 掉到 1.08、两态 profile 半宽/SE=2.7；见 `score_diagnostics_v3/README.md` |
| P0.2/P0.3 GEVP 统计 | 新增 `gevp_stats.py`：去重 → block bootstrap → replica 平均矩阵 → 秩截断 GEVP → forward/cosh 有效能量；已知谱 100 次模拟覆盖率 **0.95**；真实 Gram 数据 forward $E_0=1.0629\pm0.0075$（探索性）；见 [`doc/gevp_stats.md`](../../doc/gevp_stats.md) |
| P0.1 GEVP 矩阵身份 | 新增 [`doc/gram_operator_identity.md`](../../doc/gram_operator_identity.md) + `gram_identity_check.json`：$M$（sink–sink Gram）与标准 source–sink $C$ 差约 30%，$C$ 的 anti-Hermitian 部分也是 30% → 旧 Gram GEVP 降级为探索性；multisource 才是标准矩阵，但源端核需对齐 |
| §6.1 能量估计实验 | 新增 `energy_experiment.py` + `results/pion_projection/energy_target_v1/`：单 cosh 偏差 0.0065/覆盖率 0、两态无偏但 SE 0.039、基态消失偏差 0.33/覆盖率 0；$N_{\rm boot}$ 从 50→800 使 naive SE 缩小 4 倍而独立数据集 SE 不随其变化；真实 anchor NOT_RUN |
| F3 状态持久化 | `submission.json`/`validation_results.json` 现在带 `run_state`（operator/numerical/reference/statistical/comparison；evaluate 永久记录 validation 观察）；见 [`doc/run_state.md`](../../doc/run_state.md) |
| F3 状态/账本切片 | 新增 `core/state.py`（子状态、合法迁移、score 可空、validation 观察永久）与 `core/ledger.py`（分阶段/环境账本，拒绝混合设备）；`nested_cv` 的 `fold_plan.json` 增加 bindings；见 [`doc/run_state.md`](../../doc/run_state.md) |
| F1 验收（一致性） | `consistency.py` + `results/pion_projection/consistency_v1/`：证书与 comparator 的 4 个 lower bound 逐位一致；原样副本不拒、改数据值/改协议 hash 均 fail-closed |
| F1 接口切片 4 | 新增 `core/reference_provider.py`（pion `single`/`twostate`、wilson `identity`）；`agents/loop.py` 已改走该入口 |
| F1 接口切片 5 | 新增 `core/search.py`（Budget + search + SearchResult）；`agents/loop.py` 的每轮候选已改走该路径，非法/异常候选带原因记录 |
| F1 接口切片 3–4 | 新增 `core/measurement.py`（bundle + 能力声明 + 拒绝）与 `core/comparator.py`（共享 draw 校验、paired ratio、log 下界）；`scoring.py` 已改为调用 comparator；pion bundle 声明 linear_response=True、cross_matrix=False |
| F1 接口切片 1–2 | 新增 `core/specs.py` + `tasks/pion_projection/spec.py`（TaskSpec/RunSpec、OperatorContract）；见 [`doc/interfaces.md`](../../doc/interfaces.md) |
| F2 LLM/搜索隔离 | 统一 `core/candidates.validate_weights`（sum=1、L1≤8、finite、维数）；`agents/loop.py` 记录拒绝原因到 `agent/rejections.json`；`--no-validation` 下搜索不打开 validation（拦截测试证明）；mock 轨迹见 `results/pion_projection/agent_mock_v1/` 与 [`doc/agent_isolation.md`](../../doc/agent_isolation.md) |
| §7 新测量先决条件 | 新增 [`doc/pion_new_measurement_plan.md`](../../doc/pion_new_measurement_plan.md) + `measurement_plan.example.json`（schema `pion-measurement-plan-v1`，校验器 `measurement_plan.py`）；身份绑定、γ5/自收缩、source–sink 矩阵、盲性、power、预算、停止规则；执行 ⛔ |
| A2 gate artifact | `develop` 写每模型 gate artifact（model/window/prior/reference hash/dev hash/candidates/metric/calibration）；joint 校验 model/dev/protocol/window，缺失即 `gate_artifact_missing` 并 abstain；历史 run 不被追认（`joint_identity_v3/`） |
| A2 draw 身份 | `joint_draws.npz` 保存 inner 索引；`joint_manifest.json` 记录输入/依赖/spec；证书要求 `draw_ids` + hash，输出目录拒绝覆盖；产物 `joint_identity_v2/` |
| F0 RunSpec 强制 | `bundle_for` 校验 split 行数与 RunSpec 范围一致，manifest 带 `run_spec` digest |
| A1 candidate 身份 | `candidate` 现在调用 `verify_run_identity`（protocol/dev hash 失败即拒绝）；结果写 `certification=abstain`（candidate 只做诊断，证书由 joint 发）与 `physical_score=None` |
| A1 evaluate 认证块 | 身份检查抽到 `identity.verify_run_identity`；evaluate 写 `certification`/`physical_score`/`conditional_scores`；无 joint draws 时明确 abstain；产物 `results/pion_projection/certification_v1/` |
| A1/A2 联合认证身份 | `joint.run` 现在核对冻结 reference kind、models 精确匹配、protocol/dev hash、report kind/version，拒绝覆盖旧输出；区分 `bootstrap_mean_risks` 与 `point_risks`；新产物 `results/pion_projection/joint_identity_v1/` |
| P0.6 联合 resampling | 新增 `joint.py` 与 `python -m tasks.pion_projection.joint`：同一 draw 重拟合参考、同一组 validation 重采样、所有模型/候选共享 draw；产物 `results/pion_projection/joint_20261118/`，因参考 gate 失败而 abstain |
| P0.4 评分契约 | 新增 [`doc/score_contract.md`](../../doc/score_contract.md)：estimand、证书统计量、独立单位、有效范围、abstain 条件 |
| P0.5 schema 强制 | `scoring.py` 现在精确校验模型/候选集合、一维等长 draw、原始正值、invalid 比例、近零 baseline、improvement 门槛；反例测试见 `tests/test_pion_projection.py` |
| $m_0$ 敏感度扫描 | 分数随 $m_0$ 变化约 7 倍、排序在 1.06 附近翻转（`score_diagnostics_v2/m0_scan.json`） |
| 固定 $m_0$ 的端到端评分 | 单 cosh 下 SLSQP 更好（1.63）、两态打平、GEVP 质量下 tuned 更好（0.70）；见 `fixed_m0_scores.json` |

## S1 评分证书模块（`scoring.py`）

`todo.md` 推荐的“保守改善证书”已经落成一个纯函数：

```python
from tasks.pion_projection.scoring import improvement_certificate
out = improvement_certificate({model: {candidate: risk_draws}}, spec)
```

- `spec` 走 `pion-score-v1` schema（示例：`score_spec.example.json`），必须显式给出
  `estimand`、`baseline_id`、`alpha`、`comparison_family`、`risk_floor`；
- 每个模型 $m$、候选 $c$ 用同一组 bootstrap draw 计算
  $D_{cm}=\log R_{b,m}-\log R_{c,m}$；
- 同时下界用 Bonferroni 水平 $\alpha/(J\cdot K)$ 的分位数，再对所有模型取最小：

$$
L_c=\min_m \operatorname{quantile}_{\alpha/(JK)}(D_{cm}),
\qquad \text{improvement}=\exp(L_c);
$$

- 只有 $L_c>0$ 才算 certified；无效 draw 比例超过 `invalid_tolerance` 直接 **abstain**；
- 不把原始 risk 比值直接平均（分母接近零会爆），也不对每个候选单独抽参考质量。

**当前真实数据仍不能认证**：单 cosh / 两态两个参考都失配（见上面的模型平均与 profile），
这个模块只保证“给定可信参考 draws 时，证书的定义是可执行的”。合成测试覆盖：真实改善被认证、
变差被拒绝、NaN draw 触发 abstain、schema 校验。
