# pion 新测量的先决条件（§7 设计）

状态：**NOT_RUN**（需要新 GPU 生产与授权）。可执行模板：
[`tasks/pion_projection/measurement_plan.example.json`](../tasks/pion_projection/measurement_plan.example.json)，
校验器 [`measurement_plan.py`](../tasks/pion_projection/measurement_plan.py)（schema `pion-measurement-plan-v1`）。

## 1. 测量设计

- **系综**：新的 S16T16D（或复用 S16T16C，但必须声明“已观察”）；$\beta=6.0$、$16^4$；
- **source**：point + $H^4$ + $H^{16}$（与 sink 核一致，用同一个 $H^n$，不用近似 Wuppertal）；
- **sink**：8 个幂次 $\{0,2,4,8,16,24,32,64\}$，构成完整 source–sink 矩阵；
- **source 平均**：同一 configuration 内先对 source 平均，再按 configuration block 做 bootstrap；
  **sources 不是独立 configuration**，不能当独立样本增大统计量。

## 2. 身份绑定（每条都要写进 provenance）

| 字段 | 内容 |
| --- | --- |
| gauge | 每个组态文件的 sha256 |
| propagator | 每个组态传播子的 sha256 |
| QUDA | 版本 + 构建 hash |
| device / precision | DCU 型号/驱动、complex128 |
| trajectory IDs | 每个保存组态的 trajectory 序号 |
| generation | sha256(ids + protocol hash + kernel hash) |

现有 `measure.py` 已记录 gauge sha256、kernel sha；新 run 需补 propagator/QUDA/device/trajectory。

## 3. 检查（生产前）

1. **γ5 / self-contraction**：小体积用 fast path 与 generic kernel 对比同一 sink 通道；
2. **residual**：每个 source 的 12 RHS 最大相对残差 ≤ 容差；
3. **gauge header geometry**：`read_nersc(expected_geometry)`；
4. **source–sink 矩阵**：必须实际测量（或严格恒等式），不能按名称猜；
5. **C anti-Hermitian**：$C-C^\dagger$ 范数 $<5\%$，否则拒绝该矩阵（见 `gram_operator_identity.md`）。

## 4. 统计与盲性

- 独立单位：configuration block（长度 5），bootstrap 1000；
- **power simulation**：按 block 方差与目标效应预先模拟，决定新 validation 的组态数；
- 冻结：候选、reference spec、score spec、主终点、分析器、比较族、算力与 validation IDs；
- **盲性由访问/冻结记录判断**，不由日期或机器相同与否；旧 validation 已公开探索，换目录不恢复盲性。

## 5. 资源与成本

| 项 | 估计 |
| --- | --- |
| 反演 | 每 source 每配置 1 次（3 sources） |
| sink | 每配置 3×8 次 kernel apply |
| 90 组态（60 dev + 30 val） | 3 sources × 90 = 270 次反演 |
| 计时 | 用 `core.ledger` 记录 setup/solve/kernel/io/analysis |
| 真实墙钟 | 需 pilot 测量；不得用未定义的“risk×秒” |

S16T16C 的实际参考：8 个 powers、单 source、60 组态约 20 分钟；3 sources 约 3×。

## 6. 停止规则

1. residual/γ5 检查失败 → 停止，不进生产；
2. 新 development 上参考 gate 失败 → 无分数；
3. power simulation 显示目标效应不可达 → 停止；
4. 预算超限 → 归档负结果。

## 7. 与公共框架的适配

- `MeasurementBundle`：声明 `linear_response`/`cross_matrix`/`new_source` 能力；
- `ReferenceProvider`：pion 的 `single`/`twostate`，共享 draw IDs；
- `SearchAdapter`：候选校验 + 预算；
- 统一由公共层生成 same-draw 结果，避免三个 task 各建一套 resampling。

## 8. 当前状态

| 项 | 状态 |
| --- | --- |
| schema + 模板 + 校验器 | ✅（`measurement_plan_v1`） |
| 新系综/S16T16D | ⛔ |
| source–sink 矩阵测量 | ⛔ |
| γ5/self-contraction 小体积检查 | ⛔ |
| power simulation 与冻结 | ⛔ |
