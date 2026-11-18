# results/

这些是**真实跑出来的**结果文件，默认被 `.gitignore` 忽略，这里只强制加入小体积的、可离线复算的产物。

## `pion_projection/s16t16_v1/`

- **数据来源**：2026-11-13 在 kunshan DCU 上，用本仓库代码 + PyQUDA HMC 现产的 100 个 `16^4`、β=6.0 纯规范组态；开发 0–59、gap 60–69、留出 70–99。
- **协议**：本次运行用的是 60/10/30 的协议，和仓库里 `tasks/pion_projection/protocol.json`（30/5/15、jinchen 路径）**不同**。该运行的协议快照存在本目录的 `protocol.json`，哈希为 `d03b1e6f…`；`development.npz` / `validation.npz` 里也带着同一个哈希。
- **结论**：参考态没过 gate（名义窗口 p=1.7e-8，窗口漂移 21.6σ / 7.8σ），所以 `score_status = unscored_reference_failed`，所有 `score` 为 `null`。这是设计行为，不是 bug。
- **前三名（诊断口径）**：`agent_signed_mixture` 0.00770（相对 local 40.6）、`tuned_gaussian` 0.01256（24.9）、`local` 0.31267（1.0）。区间与解读见 `REPORT.md`。

### 文件

| 文件 | 内容 |
| --- | --- |
| `development.npz` / `validation.npz` | `(60,7,16)` / `(30,7,16)` 的关联函数 + ids + protocol_hash |
| `development_results.json` | 7 个基线的开发集 risk，以及 SLSQP 混合的结果 |
| `submission.json` | 冻结提交：权重、哈希、参考诊断 |
| `validation_results.json` | 留出结果（score 全 null）+ 嵌套区间 |
| `REPORT.md` | 代码生成的前三名表格 + LLM 解读 + LLM 作答（post-hoc） |
| `pilot_results.png` | 4 面板诊断图 |
| `protocol.json` | 本次运行实际使用的协议（60/10/30） |
| `agent/` | LLM 每轮提案（`round{r}_{i}.json`）、最佳（`best.json`）、交卷定义（`llm_candidate.json`） |
| `candidates/*/development_verification.json` | 用 `candidate --verify-only` 在 1 个组态上核对通用算子与基矢组合一致（最大相对差 1.9e-15） |

### 离线复算

有 npz 就能跑分析，不需要 GPU：

```bash
python -m tasks.pion_projection --output results/pion_projection/s16t16_v1 report   # 需要 API key
python -m tasks.pion_projection --output results/pion_projection/s16t16_v1 agent --rounds 1
```

注意：`develop` / `evaluate` 默认用 `tasks/pion_projection/protocol.json`；跑本目录的 60/10/30 协议时用
`LATTICEARENA_PROTOCOL=results/pion_projection/s16t16_v1/protocol.json`，不一致时会直接报错——这是冻结机制。

## 2026-11-18 新增

### `pion_projection/s16t16_gevp/`

为 C2 补测 $H^{24}$ 的 run（协议 `tasks/pion_projection/protocol_gevp.json`，幂次
$\{0,2,4,8,16,24,32,64\}$，同一批 60/30 组态）。参考仍失败（分数 `null`）；3×3 Hankel GEVP
$E_0=1.026\pm0.017$（dev）/ $0.997\pm0.018$（val），无平台。详见 `doc/pion_gevp_design.md`。

### `pion_projection/s16t16_twostate/` 与 `s16t16_t32/`

参考模型的诊断 run：两态参考（gate 通过但过拟合，profile 1σ 宽度 0.04）、T=32（参考仍失败，
归档反射 z=274.27 是硬编码 T 的假警报，修正后 2.40）。只读，不参与正式评分。

### `pion_projection/score_diagnostics_v2/`

只读归档审计：inventory、参考扫描、协方差谱（15 维 / 12 blocks、4 个零特征值）、
block/shrinkage 扫描、block 级校准（循环性，不可用）、模型平均（权重退化）、两态 profile、
GEVP 合成验证、能量目标合成验证。

### `wilson_loop/s16t16_v3/` 与 `paired_s16t16_v3/`

修复后的协变 APE 在 200 组态系综（`data/S16T16B`）上的重测：开发 0–99、留出 120–199
（80 组态 / 16 blocks）。留出分数 identity 100、APE(0.4,5) **263.3**、APE(0.4,10) 232.8、
APE(0.6087,10) 221.9；配对区间 [1.32, 5.93] 等均不含 1。详见 `doc/wilson_loop_v3.md`。

### `optimize_gaussian/v9/`

用 link 中心 A1+ 特征重训的学习核：测试 risk 0.009236（相对 $G^8$ 2.12×），候选 risk 0.007721
与 SLSQP 0.007703 打平（0.998 [0.78, 1.18]）。含 `manifest.json`（protocol/W/θ/IDs/依赖 hash）。
详见 `tasks/optimize_gaussian/README.md`。

> 协议选择：跑某个历史 run 时用 `LATTICEARENA_PROTOCOL=<run>/protocol.json`（pion）或
> `WILSON_PROTOCOL=<...>`（wilson）；仓库默认协议与历史 run 不同时，直接跑会按冻结机制报错。
