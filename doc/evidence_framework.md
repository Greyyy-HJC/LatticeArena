# 公共证据框架（F0 切片）

本文件描述已实现的公共层：`core/evidence.py`、`core/operator_checks.py`、
`scripts/audit_run.py`、`scripts/summarize_evidence.py`。**F1–F5 的接口/状态/成本账本仍未实现**。

## 1. schema

每条证据记录：

| 字段 | 含义 |
| --- | --- |
| `test_id` | 稳定 id（如 `wilson.shape_finite`） |
| `task` | 所属 task |
| `claim` | 这条检查声称什么 |
| `command` | 复现命令 |
| `seed` / `geometry` / `dtype` | 运行条件 |
| `code` / `inputs` | 代码与输入 hash |
| `observed` / `threshold` | 观测值与判据 |
| `status` | `PASS` / `FAIL` / `NOT_RUN` / `INCONCLUSIVE` |
| `scope` | `synthetic_CPU` / `archived_data` / `live_GPU` |

`NOT_RUN` 是合法状态：缺失的检查不会被当成通过。`validate` 会拒绝缺字段、非法状态和非法 scope。

## 2. 已有产物

```bash
python scripts/audit_run.py --task wilson_loop --run results/wilson_loop/s16t16_v3 \
    --protocol tasks/wilson_loop/protocol_v3.json --out results/wilson_loop/evidence/s16t16_v3.evidence.json
python scripts/audit_run.py --task pion_projection --run results/pion_projection/s16t16_gevp200 \
    --out results/pion_projection/evidence/s16t16_gevp200.evidence.json
python scripts/summarize_evidence.py results/wilson_loop/evidence/*.json results/pion_projection/evidence/*.json
```

当前结果：wilson 3 条、pion 4 条，全部 `PASS`、scope `archived_data`。它们只覆盖
“文件/形状/hash 一致”，不是物理证据——物理检查仍由各 task 的测试与诊断报告承担。

## 3. 可复用检查与反例 fixture

`core/operator_checks.py` 提供：

- `linearity_residual(op, f, g)`；
- `hermiticity_residual(bilinear, f, g)`；
- `covariance_residual(op, transform, f)`。

反例 fixture（只在测试里用，不改生产核）：

- `bad_gram_schmidt_projector`：右乘 Gram–Schmidt，**右协变失败**（旧 bug 的形态）；
- `bad_nonhermitian_kernel`：前向 hop，非 Hermitian；
- `good_hermitian_kernel`：对称 hop，Hermitian。

测试 `test_operator_checks_and_counterexamples` 验证：合法算子通过、反例被检出。

## 4. 与 task 的分工

- 公共层只负责 schema、可复用检查与汇总脚本；
- 每个 task 自己决定保存哪些物理检查（gauge、Hermitian、O_h、平移、时间局域），
  不强迫所有 task 用同一组约束；
- 测试总数不能替代逐项验收；`PASS` 的 scope 必须写清楚（合成/归档/真实 GPU）。

## 5. 未做（明确 NOT_RUN）

- F1 五类接口（TaskSpec/RunSpec、OperatorContract、MeasurementProvider、ReferenceProvider/Metric、SearchAdapter）；
- F2 mock LLM 隔离与候选校验（当前 `agents/loop.py` 仍只查维数/非零）；
- F3 状态机、fold-local 统计与成本账本；
- F4 HMC 质量证据（force/可逆性/ΔH）；
- F5 集成关闭条件。
