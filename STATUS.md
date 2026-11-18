# 项目状态（2026-11-18，新开工单第二轮后）

三个任务的最终状态。细节见 [`doc/final_report.md`](doc/final_report.md)、
[`doc/todo_coverage.md`](doc/todo_coverage.md) 和各自的 task README。

## 一句话

| task | 状态 | 关键数字 |
| --- | --- | --- |
| `wilson_loop` | **修复后显著改善；P0–P3/F0–F3 完成** | v3 263/233/222（区间不含 1）；measure 入口 bug 修复；三文件恢复事务；两态反例 bias/SE 32.5 而 gate 通过；invalid 规则 fail-closed |
| `optimize_gaussian` | **与强基线未检出差异；P0 完成、P1/P2/P3 CPU 部分** | v9 2.12×；0.998 [0.78,1.18]；BCB <1e-10；七幂 teacher 盲区 0.01268；GPU 2×2 ⛔ |
| `pion_projection` | **无可信分数；A/B/C 主要 CPU 切片完成** | 参考仍失败（χ²=90.6/8）；gate 假拒绝 0.02/0.03、失配功效 0.66/0.59；证书假认证 1/15、失配全 abstain；naïve anchor 证书 null 下 27–50% |

## 主要新增（本轮）

### pion

- A1/A2：`identity.verify_run_identity` 被 joint/evaluate/candidate 共用；inner indices + `require_draw_ids`；
  重复/乱序 draw ID 拒绝；evaluate 无 joint draws 时 abstain；
- B1：nested CV 的 training late_metric 冻结，fold_plan 带 train/test IDs 与 hash；
- B2：GEVP 固定平台、局部/整 replica 失败计数；
- C1：dof=8 + 内层 null 的 gate 校准（0.02/0.03 vs 0.66/0.59）；
- C2：端到端证书校准（假认证 1/15、功效 15/15、失配全 abstain）；
- C3：anchor 校准（block draws、独立共享 anchor、naïve 证书反保守）。

### wilson

- P0：measure 入口的 `start`/浮点切片 bug 修复 + 端到端测试；
- P1：三文件恢复事务 + 依赖闭包 + 投影失败 batch 索引；
- P2：归档点值重放、leave-one-block-out jackknife、非退化 coverage 0.983、invalid fail-closed；
- P3：两态谱反例（spread gate 盲点、window_z 敏感性）。

### optimize

- 包 A 切片：`train.validate_inputs`（gauge/propagator 摘要、residual、metadata 不符即拒绝）；
  `run_manifest` 记录 inputs 状态、environment=unknown、reference 摘要；
- 包 B 切片：`amplitude_source`（train/test）让 point 与 replica 同算法；fold IDs/weights hash/nfev 已存；
  synthetic regression 偏差 −1.9e-6/−1.0e-5、覆盖 1.000/0.983；
- 包 C：`p3_crosslink` 独立 train/test links、同一 $Q[U]$ 规则；oracle 显示嵌套类跨 links 误差 0、
  谱类 0.06→0.5；SLSQP/Powell 不收敛（nfev 1.3e4–9.4e4）；
- 包 D：`adapt.py` 真实跑通 OperatorContract/MeasurementBundle/SearchAdapter，能力拒绝不静默回退。

### 公共框架

- F0：证据 code/inputs 绑定、缺 hash 判 NOT_RUN、bundle 可信加载 + RunSpec；
- F1：check_draws 一维等长有限、重复/乱序拒绝；
- F2：loop 真走 SearchAdapter，NaN 拒绝，search_trace；
- F3：RunState 需 evidence hash，physical/conditional score 分离。

## 剩余（都需要新资源/授权）

- pion：A1 证书复用、A2 每模型 gate artifact、C2 blocks/JK 扫描、§6.2/§7 执行；
- wilson：NERSC checksum 支持范围、F4 生产 ΔH/接受率、P4 扩展；
- optimize：GPU class×optimizer 2×2、消融、新盲测、power simulation。

## 结项状态（CPU 范围）

- 三份 todo 的 CPU 必做项均已有代码、反例/诊断与有限结论；
- 未跑项（pion §6.2/§7、wilson F4 生产 ΔH/接受率、optimize 真实 GPU metadata/2×2/盲测）
  已逐项标 NOT_RUN 并写明原因；
- 结论：**CPU 范围可结项**；GPU/新测量需另行授权。

## 健康检查

- 104 个 CPU 测试通过；
- 所有非公式链接有效；
- README / 总报告 / 结果 JSON 数字一致。
