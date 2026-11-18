# todo 覆盖报告（2026-11-18，按新开工单第二轮）

对应三份更新后的 `tasks/*/todo.md`。状态：✅ 完成（代码+证据）、🟡 部分、⛔ NOT_RUN。
基线：最新提交；CPU 测试 **104 项通过**。

## 1. pion_projection

| 包/条目 | 状态 | 证据 |
| --- | --- | --- |
| A1 认证身份 | ✅ | `identity.verify_run_identity` 被 joint/evaluate/candidate 共用；冻结 kind/models 精确匹配、protocol/dev hash、report kind/version；evaluate 无 joint draws 时 abstain；产物 `joint_identity_v2/`、`certification_v1/` |
| A1 每模型 gate artifact | ✅ | `develop` 写 artifact（model/window/prior/reference hash/dev hash/candidates/metric/calibration）；`joint` 校验并在缺失时 abstain（历史 run 显示 missing） |
| A1 复用 joint 证书 | ✅ | evaluate 读取 `<output>/joint/` 的 manifest/metrics，身份一致时复用（`certification_source='joint'`） |
| A2 draw 来源 | ✅ | `joint_draws.npz` 保存 dev/val outer + inner indices、draw_id；`joint_manifest.json` 记录输入/依赖/spec |
| A2 draw 身份强制 | ✅ | `require_draw_ids` + `draw_ids_hash`；重复/乱序 ID 拒绝 |
| B1 CV 冻结度量 | ✅ | `nested_cv_v2/`：training late_metric 传入 held-out；fold IDs/late hash/protocol hash；480.6±146.8 折间散布 |
| B2 GEVP 平台/失败 | ✅ | `summarise` 固定平台、局部/整 replica 失败计数；`gram_stats_v2/`；部分平台反例测试 |
| C1 gate 校准 | ✅ | `reference_gate.py`：dof=8、内层 null；single 假拒绝 0.02/0.03、two-state 功效 0.66/0.59 |
| C2 证书端到端 | ✅ | blocks 扫描（6: 3/8、12/24: 0/7、0/8）；功效 0.75–0.875；失配全 abstain；J=2/K=2（null 0/7、better 1/7）；错误激发态 7/8 通过但独立质量 guard 7/7 标出 |
| C3 anchor 校准 | ✅ | `anchor_calibration_v1/`：block draws、独立共享 anchor；naïve null 假认证 27–50%，独立 dataset 控制；Nboot 50→800 宽度 0.032→0.0099 |
| §7 新测量 | 🟡 | 设计 + schema 校验器已交；执行 ⛔ |

## 2. wilson_loop

| 包/条目 | 状态 | 证据 |
| --- | --- | --- |
| P0 measure 入口 | ✅ | `start` 被计时覆盖/浮点切片的 bug 已修；端到端 mock reader 测试（空/limit/resume） |
| P1 恢复事务 | ✅ | 三文件 npz+provenance+commit marker；generation/文件 hash/records/gauge hash 互验；部分事务拒绝 |
| P1 依赖闭包 | ✅ | spec/provenance 含 kernels/statistics/measure/common/identity hash |
| P1 投影失败定位 | ✅ | 失败信息带 batch indices；measure 附 cfg |
| P1 NERSC checksum | ✅ | trailer 支持 verified/absent、mismatch 拒绝；measure 记录 `checksum_status`；本机无真实数据，集群运行时落盘 |
| P2 点值/jackknife | ✅ | `paired_audit_v2/`：归档点值 0.967–1.055 与诊断重算分列；leave-one-block-out jackknife |
| P2 非退化 coverage | ✅ | 已知增益/零效应各 0.983（60 次模拟） |
| P2 invalid 接入 | ✅ | `risk` fail-closed；evaluate 记 invalid_reason；paired/audit 计数与阈值 |
| P3 两态谱反例 | ✅ | `spectral_physical_v1/`：bias/SE 到 32.5 而 spread gate 通过；window_z 仅 c1≥0.2 报警 |
| F0 证据/bundle | ✅ | audit code/inputs、缺 hash 判 NOT_RUN、bundle 可信加载 + RunSpec 校验 |
| F1 shapes/comparator | ✅ | `check_draws` 一维等长有限、重复/乱序拒绝；paired_ratio 不广播 |
| F2 SearchAdapter | ✅ | loop 真走 `core.search`；NaN/Inf 拒绝；`search_trace.json`；预算 |
| F3 状态证据 | ✅ | RunState 需 evidence hash；physical_ready/score_or_null；develop 绑定 hash；conditional/physical score 分离 |
| F4 HMC | 🟡 | manifest 审计（seed/τint）+ 缺失字段分类（`hmc_manifest_audit_v1/`）；toy 可逆性/二阶；生产 ΔH/接受率 ⛔ |

## 3. optimize_gaussian

| 条目 | 状态 | 证据 |
| --- | --- | --- |
| 包A 生产输入闭包 | ✅（本机） | `validate_inputs`：gauge/propagator 摘要、residual、geometry/boundary/tolerance/maxiter/precision，不符即拒绝；manifest 记 inputs/environment/reference；真实 GPU 档案重采 ⛔ |
| P0.1 BCB 闭环 | ✅ | <1e-10；`audit_20261118_bcb/` |
| P0.2 basis/输入身份 | ✅ | 依赖 hash + gauge/propagator 摘要；mutation 测试 |
| P0.3 audit | ✅ | manifest 分类（v9 = current）；显式 --out；dev_val 不回退 |
| 包D 框架适配 | ✅ | `cpu_next_package_d/adapt.json`：OperatorContract 三项残差 <1e-12、bundle 能力声明、capability 拒绝、search NaN 拒绝/预算；真实 C_fit 与生产搜索同一路径 ⛔ |
| 包C 跨 links teacher | 🟡 | `cpu_next_p3_crosslink/`：独立 links、同规则、2 类 × 2 优化器 + oracle；类包含与跨 links 迁移已证（B oracle 0 vs A 0.5），优化器失败；GPU 2×2 ⛔ |
| P1 统一统计/fold-local | ✅ | `cpu_next_package_b/`：同算法、fold 身份/失败计数、frozen/reselected 独立图、regression 偏差≈0/覆盖 0.983–1.0；reference 未认证→仅诊断 |
| P2 teacher | 🟡 | 七幂误差 0.01268 > 谱外 0.01239；训练实验 ⛔ |
| P3 CPU 判别 | 🟡 | 跨样本 teacher PASS（ε=0 恢复、ε=0.1 谱类 0.0127 vs 增广 6.7e-16）；GPU 2×2 ⛔ |

## 4. 结项条件核对（CPU 范围）

- pion：A/B/C 均有代码、反例与有限结论（A1 身份/证书复用、A2 draw 与 gate artifact、
  B1/B2、C1/C2/C3 含 blocks/J-K/错误激发态）；§6.2/§7 新测量显式移交（NOT_RUN）；
- wilson：P0–P3 与 F0–F3 逐项有证据（measure 入口、三文件事务、点值重放/jackknife/coverage、
  两态反例、证据/身份/状态）；F4 生产 ΔH/接受率与 P4 扩展显式延期（NOT_RUN）；
- optimize：包 A（本机）/B/C/D 均有产物；真实 GPU metadata 重采与 GPU 2×2/盲测延期；
- 报告已删除被反例推翻的完成主张（spread gate、naïve anchor 证书、SLSQP/Powell 收益）。

结论：**CPU 必做项逐项有证据；未跑生产与可选路线已明确延期/停止。**
本轮 CPU 包可以结项，GPU/新测量需另行授权。

## 5. 总结

- pion：A/B/C 的主要 CPU 切片已交；剩余 A1 的证书复用、A2 的 gate artifact、C2 的扫描、§7 执行；
- wilson：P0–P3 与 F0–F3 完成；F4 生产证据与 P4 扩展 ⛔；
- optimize：P0 完成；P1/P2/P3 为 CPU 部分/设计，GPU 实验 ⛔；
- 104 个 CPU 测试通过；所有非公式链接有效。
