# 公共接口（F1，第 1–2 类切片）

已实现：`core/specs.py`（校验）+ 两个 task 的 spec adapter。**第 3–5 类接口（MeasurementProvider、
ReferenceProvider/Metric、SearchAdapter）仍未抽取**，任务仍直接调用现有实现。

## 1. TaskSpec / RunSpec

字段：`name / observable / geometry / boundary / source / splits / reference / metric / budget / dependencies`。

校验：

- geometry 必须是四个正整数；
- splits 必须声明 `development` 与 `validation`；
- dependencies 是 name→hash 映射。

`run_spec(task, protocol_hash, data)` 绑定一次运行的协议与数据 hash，不做注册表。

适配器：

- `tasks/pion_projection/spec.py`：`task_spec_for(protocol, reference_kind, protocol_path)`、
  `operator_contract_for()`；
- `tasks/wilson_loop/spec.py`：同样的两个函数。

两个 adapter 的 geometry、splits、kernels/statistics hash 都直接来自各自的协议与源码，测试保证与协议一致。

## 2. OperatorContract / CandidateSpec

字段：`name / axes / dtype / transformation / support / parameters / dependencies / constraints`。

允许的 axes（当前实现用到四种）：

- `(mu,t,z,y,x,3,3)`：wilson 的 link；
- `(3,z,y,x,3,3)`：pion 的空间 link；
- `(z,y,x,3,nrhs)`：pion 的费米子场；
- `(t,z,y,x,4,4,3,3)`：传播子（lexico）。

`transformation ∈ {gauge_covariant, gauge_invariant, spin_scalar}`，
`support ∈ {spacetime_local, time_slice, point}`。
`candidate_spec(builder, parameters, contract, dependencies)` 只接受已声明的 contract 名。

## 3. 两个 task 的 contract

| task | name | axes | transformation | support |
| --- | --- | --- | --- | --- |
| pion | `pion-kernel-v1` | `(z,y,x,3,nrhs)` | spin_scalar | time_slice |
| wilson | `wilson-kernel-v1` | `(mu,t,z,y,x,3,3)` | gauge_covariant | spacetime_local |

## 4. Measurement bundle（第 3 类，已完成最小切片）

`core/measurement.bundle(task, arrays, manifest, capabilities)` + `require()`：

- 能力声明：`linear_response`、`cross_matrix`、`new_source`，缺哪个就拒绝哪个
  （`UnsupportedCapability`），绝不从单源/对角缓存推断缺失响应；
- 校验数组非空、有限，manifest 是映射，能力必须是 bool。

两个 task 的 adapter：

| task | arrays | linear_response | cross_matrix | new_source |
| --- | --- | --- | --- | --- |
| pion | development/validation 关联函数 | True | False | False |
| wilson | development/validation 圈 | False | False | False |

## 5. Comparator（第 4 类，已完成最小切片）

`core/comparator.py`：

- `check_draws(risks, draw_ids)`：所有数组必须等长，禁止广播；可返回 draw_ids 的 hash；
- `paired_ratio(baseline, candidate)`：中位数、中心 95% 区间、exceedance fraction；
- `log_lower_bound(baseline, candidate, level, floor)`：pion 证书用的对数下界。

`tasks/pion_projection/scoring.py` 已改为调用这两个函数（真实复用，而不是复制逻辑）。

## 6. SearchAdapter（第 5 类，已完成最小切片）

`core/search.py`：

- `Budget(max_evaluations)`：显式预算，超限即停并标记 `budget_exhausted`；
- `search(proposals, evaluate, validate, budget)`：候选依次经过校验 → 评估 → 记录；
  非法候选与评估异常都写入 `rejected`（带原因），不静默丢弃；
- `SearchResult.best()`：按 risk 取最优。

`agents/loop.py` 已改为走这条路径（LLM 的每个 round 变成 `search` 调用），
`validate_weights` 是统一校验器；SLSQP 仍由自己的约束生成单一候选，但共享同一校验器。

## 7. ReferenceProvider（第 4 类，已完成最小切片）

`core/reference_provider.py`：

- `make_provider(task, kind, **params)`：声明每个 task 支持的 kind
  （pion: `single`/`twostate`；wilson: `identity`），未知 kind 直接拒绝；
- `provide(provider, data, index=...)`：
  - pion 返回 `{'fit', 'masses'}`（重拟合参考 + bootstrap 质量）；
  - wilson 返回 `{'v_ref'}`（identity 的窗口势）。

`agents/loop.py` 已改为走该入口（不再直接 import `make_reference`），两个 task 的测试覆盖
fit/masses 形状与 v_ref 长度。

## 8. 未做（NOT_RUN）
- 跨 task 的缓存失效反例与同 spec 统计一致性报告；
- ReferenceProvider/Metric/Comparator（共享 draw ID 与 baseline 身份）；
- SearchAdapter（optimizer 与 LLM 的统一候选、预算与 evaluator）；
- 跨 task 的缓存失效反例与统计一致性报告。

## 5. 测试

`tests/test_specs.py`：

- TaskSpec 缺字段/几何/splits 报错；
- OperatorContract 的 axes/transformation 白名单；
- CandidateSpec 的 builder 格式与 contract 名；
- 两个 adapter 与各自协议的 geometry/splits/hash 一致。
