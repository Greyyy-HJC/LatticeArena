# 静态夸克势：空间 Wilson 线核优化（pilot）

第二个任务。目标：在固定的纯规范系综上，设计一个**空间链路核**（对每个时间片的 3 个空间方向做平滑），用它构造空间 Wilson 线，改善静态夸克势 `V(R)` 的**精度**和**平台稳定性**。

## 固定 vs 可改

| 角色 | 内容 |
| --- | --- |
| 固定（出题人） | 组态与划分；时间方向 link 不参与优化；`R = 1,2,3`；`T = 4..8`；对全部空间点、3 个方向、16 个时间原点取平均；三个拟合窗口 `[4,5,6] / [4,5,6,7] / [5,6,7,8]`；bootstrap 与种子 |
| 可改（agent） | `build_kernel(spatial_links, parameters)`：输入 `(3, t, z, y, x, 3, 3)` 的空间 link，返回同形状的模糊化空间 link。必须规范协变、时间局域、同一规则用于所有时间片与组态、不使用时间方向 link |

参考实现是带投影的 APE 平滑：

$$
F = \mathrm{Proj}\big[(1-\alpha)U + \tfrac{\alpha}{4}\textstyle\sum_{\text{4 staples}}\big]
$$

`Proj` 是**双边极分解**投影：

$$
Q = M\,(M^\dagger M)^{-1/2},\qquad P = Q\,e^{-i\arg(\det Q)/3}
$$

它满足 $P(gMh)=gP(M)h$（独立的左右 $SU(3)$ 变换），这是 APE staple 需要的协变性。
三个空间方向每个 link 只有 4 个 staple，所以归一化是 $\alpha/4$（旧版误用 $\alpha/6$）。
旧版逐列 Gram–Schmidt 只保持左酉；2026-11-18 的 CPU 反例显示规范变换后最大元素差 0.44，已修复。

> **2026-11-18 修复说明**：旧 APE 核不满足规范协变，所有旧 APE 结果（`APE(0.5,5)` 等）
> 只能作为历史诊断；identity 不受影响。数值 $\alpha$ 的口径也变了，新旧关系为
> $\alpha_{\rm new}=2\alpha_{\text{old}}/(3-\alpha_{\text{old}})$（保持 staple 与 identity 的比例）。
> 修复后需要重新测量才能给出物理结论（本轮 GPU 重测未运行）。

## 打分

对每个 `R`：

```
V(R) = -d log W(R,T) / dT      （名义窗口 T = 4,5,6,7 上的线性拟合）
risk(R) = (SE_boot(V) / V_ref)² + (spread_windows(V) / V_ref)²
risk = mean_R risk(R)
score = 100 · (risk_identity + 1e-12) / (risk + 1e-12)
```

- `V_ref` 是**开发集 identity**（不平滑）的 `V(R)`，作为固定尺度；不随候选变化。
- 第一项奖励精度（bootstrap 误差小），第二项奖励平台稳定（三个窗口给出的 V 一致）。
- identity 恒等于 100 分。
- gate：identity 的 `V(R) > 0` 且相对平台散布 ≤ 5%；不过就不给分。

## 命令

```bash
# 需要 GPU（cupy）和组态；CPU 上只能跑测试
python -m tasks.wilson_loop --output results/wilson_loop/s16t16_v1 measure --split development
python -m tasks.wilson_loop --output results/wilson_loop/s16t16_v1 develop
python -m tasks.wilson_loop --output results/wilson_loop/s16t16_v1 measure --split validation
python -m tasks.wilson_loop --output results/wilson_loop/s16t16_v1 evaluate
python -m tasks.wilson_loop --output results/wilson_loop/s16t16_v1 plot
```

`measure` 在 GPU 上约 1.5 s/组态（60 组态约 1.5 分钟）；CPU 上会慢约 10 倍。7 个单元测试完全不需要 GPU：

```bash
python -m pytest -q tests/test_wilson_loop.py
```

## 这次 pilot 的结果（16⁴、β=6.0、100 组态，开发 0–59 / 留出 70–99）

参考通过（identity 的 V(1,2,3) = 0.410 / 0.597 / 0.703，平台散布 0.0004 / 0.0025 / 0.0058）。

| 核 | 开发 risk | 留出 risk | 留出 score |
| --- | ---: | ---: | ---: |
| identity | 7.82e-5 | 5.60e-5 | 100.0 |
| APE α=0.5, N=5 | 1.95e-5 | 5.81e-5 | 95.7 |
| APE α=0.5, N=10 | 1.74e-5 | 5.42e-5 | **102.7** |
| APE α=0.7, N=10 | 1.65e-5 | 5.51e-5 | 101.5 |

**诚实结论**：开发集上 APE 好 4.5 倍，但**没有转移到留出集**——留出集只有 6 个块，±3% 的差异不显著。这个 pilot 目前**没有**证明空间 smearing 能稳定改善静态势；它证明的是这条测量/打分链路能跑通并给出可比较的数字。

## 限制（v0）

- 只支持协议里列的 4 个核；任意 `build_kernel` 提交（`candidate` 子命令）还没实现——那需要重新测量，成本较高。
- `V(R)` 用 T=4..8 的单指数拟合，忽略周期回绕项 `exp(-V(16-T))`（在 1e-4–1e-6 量级）。
- 组态自相关没有测；5 组态一块是保守选择。
- 16⁴、淬火、β=6.0，`V(R)` 是格点单位下的裸势，不能直接和物理量比较。

## 文件

| 文件 | 作用 |
| --- | --- |
| `protocol.json` | 组态路径、划分、R/T、窗口、核列表、种子、阈值 |
| `common.py` | protocol、split 索引、NERSC 读取、npz 读取与校验 |
| `kernels.py` | `build_kernel`（APE + 重幺正化） |
| `measure.py` | Wilson 圈测量（支持 cupy 加速） |
| `statistics.py` | V(R) 拟合、bootstrap、risk |
| `experiment.py` | `develop` / `evaluate` |
| `plots.py` | `V(R)` 与 risk 图 |
| `__main__.py` | CLI |

## 2026-11-18 审计状态

| 项 | 状态 |
| --- | --- |
| P0 双边极分解投影 + det 修正 | 已实现；`project(gMg')=g·project(M)·g'`、APE 多 alpha/steps 协变测试 |
| F4 HMC 质量 | 新增 `hmc_quality.py` + `results/wilson_loop/hmc_quality_v1/`：manifest 审计（seed 20261109、首轨迹 401、间隔 5、$\tau_{\rm int}\approx0.512$；ΔH/接受率/热化声明缺失=NOT_RUN）；toy 积分器可逆性 $10^{-10}$、$\Delta H$ 二阶（每减半步长 ÷4）→ 通过；生产 ΔH/接受率仍 NOT_RUN |
| F3 状态/账本切片 | `run_state` 已写入 wilson `develop` 的 submission、`evaluate` 继承并记录 `validation_observed`；`measure` 的 provenance 带 `ledger`（setup/kernel、设备/精度/并行度）；见 [`doc/run_state.md`](../../doc/run_state.md) |
| F1 验收（一致性） | pion 的跨入口一致性 + 缓存失效反例见 `results/pion_projection/consistency_v1/`；wilson 的 generation/kernel-hash 拒绝已有测试 |
| F1 接口切片 4 | `core/reference_provider.py` 提供 wilson 的 identity `v_ref`；pion 的 fit/masses 也走同一入口 |
| F1 接口切片 5 | `core/search.py`（Budget/search/SearchResult）；wilson 当前无 optimizer/LLM 搜索，适配器为 N/A，但共享校验器可用 |
| F1 接口切片 3–4 | `core/measurement.py` + `core/comparator.py`；wilson bundle 声明三项能力全为 False（归档 run 没有 cross/response，禁止推断） |
| F1 接口切片 1–2 | 新增 `core/specs.py` + 两个 task 的 spec adapter（TaskSpec/RunSpec、OperatorContract/CandidateSpec）；见 [`doc/interfaces.md`](../../doc/interfaces.md)，测试 `tests/test_specs.py` |
| F2 搜索集成 | `agents/loop.py` 真正走 `core.search.search`；NaN/Inf evaluator 结果被拒绝；每轮写 `agent/search_trace.json`（history/rejections/best/预算） |
| F3 状态一致性 | evaluate 报告写 `score_status`（conditional_diagnostic/unscored_reference_failed）；端到端测试断言 physical_score=None、run_state.score=None、comparison=draft/failed、conditional_scores 数值与标签一致 |
| F3 状态证据绑定 | `RunState.set(validated)` 必须带 evidence hash；`physical_ready()`/`score_or_null()` 只在五状态全部 validated 且有 hash 时给分；develop 的 `run_state` 与 `conditional_scores`/`physical_score` 分开 |
| F2 LLM/搜索隔离 | 共享 `core/candidates.validate_weights`；mock 轨迹 + 拦截测试见 [`doc/agent_isolation.md`](../../doc/agent_isolation.md) |
| F0 可信 bundle | 两个 `bundle_for` 改用 `load_data(run, split, protocol)`（IDs/hash/shape/finite），manifest 带 `ids`/`inputs`/`code` hash；篡改 protocol hash 的测试拒绝 |
| F0 证据绑定修复 | `audit_run` 的每条证据现在带 code/inputs hash；两个 kernel hash 缺失时判 NOT_RUN（不再 None==None 通过）；pion 行数按 RunSpec 校验（不用 len(data) 自证）；`covariance_residual` 文档明确只是固定 U 的交换关系，不是完整 gauge 协变 |
| F0 证据 schema | 新增 `core/evidence.py`、`core/operator_checks.py`、`scripts/audit_run.py`、`scripts/summarize_evidence.py`；wilson v3 与 pion 200 各产出一份同 schema 证据（`results/*/evidence/`）；见 [`doc/evidence_framework.md`](../../doc/evidence_framework.md) |
| P3 两态谱反例 | 新增 `spectral_physical.py` + `results/wilson_loop/spectral_physical_v1/`：$W=c_0e^{-E_0T}+c_1e^{-E_1T}$ 扫描 $c_1$；spread gate 全部通过而 bias/SE 到 32.5，`window_z` 在 $c_1\ge0.2$ 才超 3；有限 $N_t$ 谱写入文档 |
| P3 物理质量门控 | 新增 `spectral_quality.py` + `doc/wilson_spectral_quality.md`：合成“平但错”反例（偏差 170σ、risk $2.7\times10^{-7}$、gate 通过）；v3/v4 窗口 $V$ 图；`reference_nominal_window` 改为显式 spec |
| P2 invalid 接入 | `risk` 对非有限/非正 W fail-closed 并有反例测试；`evaluate` 逐核记录 `invalid_reason`；`paired`/`paired_audit` 剔除无效 draw 并计数 |
| F0 RunSpec 强制 | 两个 bundle 用 `task_spec_for` 校验 split 行数与 RunSpec 范围一致，并把 `run_spec` digest 写入 manifest |
| P2 invalid 接入 | `statistics.risk` 现在对非有限/非正 W fail-closed；`evaluate` 逐核记录 `invalid_reason`；`paired` 剔除无效 draw 并记录 `invalid_draws`，比例超 5% 即拒绝 |
| P2 点值/jackknife/coverage | `paired_audit_v2/`：归档点值（v2 0.967–1.055）与诊断重算分列；jackknife 改为 leave-one-block-out；非退化 coverage（已知增益/零效应各 0.983，60 次模拟） |
| P2 无效样本计数 | 新增 `invalid_audit.py` + `results/wilson_loop/invalid_audit_v1/`：预声明 mean/bootstrap 非正、NaN、零分母规则；v2/v3/v4 计数全为 0（规则未触发）；合成数据单测覆盖计数路径 |
| P2 配对统计审计 | 新增 `paired_audit.py` + `results/wilson_loop/paired_audit_v1/`：v2 点值 1.07–1.18 vs 嵌套中位数 3.0–3.4（唯一 block 分布、jackknife 0.49–4.53、相关性 0.14–0.30 解释）；`probability_better` 改名 `exceedance_fraction` |
| P1 投影失败定位 | `_inv_sqrt_hermitian` 的失败信息带 batch indices；measure 在异常外层附 `cfg <id>`；测试 `indices [1]` |
| P1 恢复事务 | 测量改为 npz+provenance+commit marker 三文件事务；resume 互验 generation/文件 hash/records 逐行 loops hash/gauge hash；部分文件、依赖变更、gauge 变更均拒绝；见 [`doc/run_state.md`](../../doc/run_state.md) §7 |
| P1 恢复故障注入 | `check_resume` 现在校验 provenance 记录数与 id 顺序；故障注入表见 [`doc/run_state.md`](../../doc/run_state.md) §7（generation/前缀/记录不一致均拒绝，残留 tmp 忽略） |
| P1 可信加载/冻结 | 新增 `identity.py`：generation 写入 NPZ/provenance、resume 只接同 generation 前缀、`read_nersc` 校验头部几何、`trust_status` 只看身份；`paired.main` 改用 `load_data`；v3 归档被判 archived_diagnostic（证据 `results/wilson_loop/audit_20261118_identity/`） |
| P0 measure 入口修复 | 修掉 `start` 被 `perf_counter()` 覆盖、`spec['ids'][:start+...]` 用浮点切片的 bug；新增端到端 mock reader 测试（空目录/limit/已完成后 resume 不重复测量）；行数逐次核对 |
| P0 投影数值边界 | 已修：逐矩阵 Newton–Schulz 尺度 + 收敛检查；混尺度 batch 与单矩阵一致（1e-16），rank-2 受控拒绝；证据 `results/wilson_loop/audit_20261118_projection/` |
| P0 staple 归一化 | 已从 `/6` 改为 `/4`；`alpha` 数值口径变化（旧 α ↔ 新 $2\alpha/(3-\alpha)$） |
| 旧 APE 结果 | **失效**（旧投影不协变），保留为历史诊断；identity 不受影响 |
| P1 provenance | 部分完成：NPZ/loop 校验、每配置 gauge sha256、kernel hash、计时、git commit/dirty、原子保存；`evaluate` 现在校验 development SHA、kernel hash 与 kernel 参数；仍缺硬件/QUDA 版本 |
| P2 配对区间 | 已写好预注册的 `paired.py`（outer 500 × inner 500，统计量 risk(identity)/risk(candidate)，报告中位数/95% 区间/P(优于 identity)）；在 v1 归档上做了 pipeline demo，标记 `legacy_non_covariant`；等 v2 重测数据重跑 |
| v2 协议（alpha/4 + 极分解） | 已使用（`protocol_v2.json`），`WILSON_PROTOCOL` 选择 |
| v2 配对分析 | 已完成（`paired_s16t16_v2`）；点估计 3× 但 95% 区间跨 1，未证明改善 |
| v3 扩大系综重测 | **已完成**：80 留出组态 / 16 blocks，三个配对区间均不含 1，APE 显著优于 identity |
| v4 加入 $R=4$ | **已完成**：$V(4)$ 的 SE 从 0.0233 降到 0.005–0.007，分数 663–808（尺度与 v3 不可比） |
| 修复后的 GPU 重测 | **已完成**（`s16t16_v2`，60+30 组态，2026-11-18） |

## 重新测量准备：v2 协议

修复后的 APE 需要重测，`protocol_v2.json` 把新口径固定下来：

- staple 归一化 `alpha/4`（三个空间方向 4 个 staple），双边极分解投影；
- 新 $lpha$ 与旧数值的关系 $\alpha_{v2}=2\alpha_{v1}/(3-\alpha_{v1})$（保持 staple/identity 比例）；
- 版本 `s16t16-wilson-v2`，`supersedes` 明确 v1 的旧 APE 结果不能再用；
- 用 `WILSON_PROTOCOL=tasks/wilson_loop/protocol_v2.json` 选择，默认协议保持 v1 不变。

生成器 `pilot/hmc_generate.py` 已从 kunshan 取回仓库，重生产时可以核对输入身份。
修复后的 GPU 重测已完成（`s16t16_v2` 与 `s16t16_v3`，见上）；更远的“更大 $R$ / 更多体积”仍是 NOT_RUN。

## v2 重测结果（2026-11-18，GPU 已运行）

用 v2 协议（$\alpha/4$ + 双边极分解投影）在 kunshan 上重测了 60 个开发组态和 30 个留出组态
（4 个核，约 6.5 s/组态）：

| 核 | 开发 risk | 留出 risk | 留出 score |
| --- | ---: | ---: | ---: |
| identity | $7.82\times10^{-5}$ | $5.553\times10^{-5}$ | 100 |
| APE(α=0.4, 5 步) | $2.11\times10^{-5}$ | $5.745\times10^{-5}$ | 96.7 |
| APE(α=0.4, 10 步) | $1.96\times10^{-5}$ | $5.378\times10^{-5}$ | 103.3 |
| APE(α=0.6087, 10 步) | $1.90\times10^{-5}$ | $5.262\times10^{-5}$ | **105.5** |

参考 gate 通过（identity 的 $V(R)=0.41029/0.59723/0.70335$，与 v1 相同）；kernel 现在是规范协变的。

**但改善仍未被证明**：预注册配对分析（500 outer × 500 inner，统计量 risk(identity)/risk(candidate)）：

| 核 | 中位数 | 95% 区间 | $P(\text{优于 identity})$ |
| --- | ---: | ---: | ---: |
| APE(0.4, 5) | 3.01 | [0.64, 28.1] | 0.842 |
| APE(0.4, 10) | 3.42 | [0.69, 23.5] | 0.874 |
| APE(0.6087, 10) | 3.41 | [0.72, 21.0] | 0.894 |

区间全部跨 1（6 个留出 block 下重复块退化、分布强偏斜），所以只能说“点估计偏好 APE，但未达到
预注册的置信水平”。

**开发/留出差距很大**：开发集上 APE 的 risk 比 identity 低约 4 倍，留出集上只换来 3–5% 的分数。
这说明开发集上的改善有相当一部分来自对 12 个 block 的过拟合，不能外推。

产物：`results/wilson_loop/s16t16_v2/`（测量、冻结与验证）和
`results/wilson_loop/paired_s16t16_v2/`（配对分析）。

> 过程中修的两个执行问题：真实 NERSC 头的 `FLOATING_POINT` 是 `IEEE64LITTLE`（不是 `1`/`true`）；
> 原子保存的临时文件必须以 `.npz` 结尾，否则 numpy 会再追加一个 `.npz`。

## v3：扩大系综后 APE 显著优于 identity（2026-11-18）

v2 的 6 个留出 block 太窄，配对区间跨 1。按 todo 的建议“扩留出集到 60+”，新生成了一套
200 组态的系综（`data/S16T16B`，接受率 97.7%），开发 0–99、留出 120–199（80 组态 = **16 blocks**）：

| 核 | 留出 risk | 留出 score | 配对中位数 | 95% 区间 | $P(\text{优于 identity})$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity | $5.625\times10^{-5}$ | 100 | — | — | — |
| APE(0.4, 5) | $2.136\times10^{-5}$ | **263.3** | 3.08 | [1.32, 5.93] | 0.986 |
| APE(0.4, 10) | $2.416\times10^{-5}$ | 232.8 | 2.94 | [1.25, 5.82] | 0.982 |
| APE(0.6087, 10) | $2.535\times10^{-5}$ | 221.9 | 2.84 | [1.17, 6.34] | 0.982 |

**三个配对区间全部不含 1**，这是这个 task 第一次在预注册分析下证明“协变 APE 优于 identity”。
完整报告（势表、评分、配对、限制、复现）见 [`doc/wilson_loop_v3.md`](../../doc/wilson_loop_v3.md)。
参考 gate 通过（identity 的 $V(R)=0.41042/0.59893/0.70849$，新系综与 v1 在噪声内一致）。

产物：`results/wilson_loop/s16t16_v3/`、`results/wilson_loop/paired_s16t16_v3/`；
协议 `tasks/wilson_loop/protocol_v3.json`。

**这也解释了 v2 的“未证明”**：不是 APE 无效，而是 6 个留出 block 的功效不够。v2 与 v3 的
点估计（约 3 倍改善）一致，v3 只是把区间收窄到了 1 以上。
