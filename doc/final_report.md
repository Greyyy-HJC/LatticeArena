# LatticeArena 阶段性总结（2026-11-18）

三个任务在 2026-11-18 做了算子修复、契约冻结和 GPU 重测。这一页是索引和结论；
细节都在各自的 task README 与 `results/` 里。

## 0. 一句话

- **`wilson_loop`**：修复核的协变性、扩大系综后，协变 APE **显著优于 identity**（留出分数 263/233/222，
  三个配对区间都不含 1）；
- **`optimize_gaussian`**：修复特征后学习核比高斯基座好 **2.12 倍**，候选 risk 与 7 参数 SLSQP **未检出差异**
  （0.998 [0.78,1.18]）；
- **`pion_projection`**：参考模型仍不可识别，正式 `score` 保持 `null`；GEVP（C2）与能量目标（C3）
  的合成验证已完成，真实数据需要新测量。

## 1. 结果总表

| task | 修复前 | 修复后 | 统计结论 |
| --- | --- | --- | --- |
| wilson_loop | 旧 APE 核不协变，结果作废 | v3：identity 100、APE(0.4,5) **263.3**、APE(0.4,10) 232.8、APE(0.6087,10) 221.9；v4（+$R=4$）663–808（尺度不同） | 配对中位数 2.8–3.1（v3）/ 6.6–9.5（v4），区间均不含 1 |
| optimize_gaussian | v7：1.52×，候选 0.009597，输给 SLSQP（0.803 [0.636,0.944]） | v9：**2.12×**，候选 **0.007721** vs SLSQP 0.007703 | 配对 **0.998 [0.78,1.18]**，未检出差异 |
| pion_projection | 参考 gate 未过（χ²=52/8，窗口偏移 21.6σ/7.8σ） | 诊断更完整，参考仍不可识别；C2/C3 合成验证通过 | 无正式分数（设计如此） |

## 2. wilson_loop：静态势

### 修了什么

- 逐列 Gram–Schmidt 只保左酉 → 换成**双边极分解** $Q=M(M^\dagger M)^{-1/2}$，det→1；
  满足 $P(gMh)=gP(M)h$；
- 3 个空间方向只有 4 个 staple → 归一化改为 $\alpha/4$；新旧 $\alpha$ 关系 $2\alpha/(3-\alpha)$；
- NERSC 读取校验 `DATATYPE`/`FLOATING_POINT`（真实值是 `IEEE64LITTLE`）/payload 长度；
- 原子保存、每配置 gauge sha256、kernel/git hash、evaluate 校验 development SHA 与 kernel 参数。

### 三个阶段

| 版本 | 留出 | 结论 |
| --- | --- | --- |
| v1 | 30 / 6 blocks | 核不协变，作废 |
| v2 | 30 / 6 blocks | 点估计 3×，区间跨 1 → 未证明 |
| **v3** | **80 / 16 blocks**（200 组态系综） | **区间不含 1 → 显著** |

### 物理读数

$V(1)$ 几乎不变，$V(3)$ 的误差减半（0.0073 → 0.0034），中心值 0.715 → 0.706。涂抹改善的是
大 $R$ 的信噪比。

详见 [`doc/wilson_loop_v3.md`](wilson_loop_v3.md)。

## 3. optimize_gaussian：学习型 smear

### 修了什么

- `plaquette` 路径闭合（原先第二/三段沿错方向平移，不是规范不变的）；
- 特征改为 **link 中心**平均（每条 link 两侧两个 plaquette）再做对称组合；
- 24 个 proper rotation + parity 的协变测试**全部通过**（A1+ 认证）；
- 训练缓存 identity、run manifest（protocol/W/θ/IDs/缓存 key/依赖 hash）。

### 结果

| 版本 | 特征 | 测试 risk | 相对 $G^8$ | 候选 risk |
| --- | --- | ---: | ---: | ---: |
| v7 | 旧的前向锚点 | 0.012870 | 1.52× | 0.009597 |
| **v9** | **link 中心 A1+** | **0.009236** | **2.12×** | **0.007721** |

与 SLSQP 的配对比值 $0.998\ [0.78,1.18]$，统计上未检出差异（区间跨 1，不是等效性证明）。学到的系数从“几乎纯均匀”变成有实质
空间调制（$|v_h|\approx0.13$）。

详见 [`tasks/optimize_gaussian/README.md`](../tasks/optimize_gaussian/README.md)。

## 4. pion_projection：为什么没有可信分数

按诊断链顺序，每一步都排除了一个“本来可以救它”的可能。

### 4.1 参考模型本身失配

单 cosh + 共质量在名义窗口 $[5,6,7,8]$ 上 $\chi^2=52/\mathrm{dof}=8$，窗口偏移 21.6σ/7.8σ；
T=32 上仍失败（$\chi^2=120/11$，偏移 20.3σ/16.3σ）。T=32 归档里的反射 z=274.27 是硬编码
T=16 索引造成的假警报，按真实 $C(t)-C(T-t)$ 应为 2.40——但 χ² 和窗口偏移仍然失败。

### 4.2 统计量不足以识别它

名义窗口是 3 通道 × 5 时间 = 15 维，而只有 12 个 block：样本协方差秩 $\le11$，4 个零特征值。
预声明的 block/shrinkage 扫描（$\{2,3,4,5,6\}\times\{0,0.1,0.25,0.5\}$）里没有任何组合能让
名义参考通过（最大 p=0.0072）。

### 4.3 校准 p 值、加长 T、两态、模型平均都不能救

- 模拟校准 p 值：校准本身正确（合成误判率 0%），但名义参考校准后 p≈0 → 模型偏差；
- 两态：gate 通过，但 8 参数对 12 个点、$\chi^2=0.06$，是过拟合；profile 1σ 宽度 0.04 ≈ 4×
  bootstrap SE（sloppy 方向）；
- 模型平均：AIC/BIC 权重退化为两态（$\approx1$），候选排序随参考翻转。

### 4.4 参考质量的直接影响

固定候选权重、扫描 $m_0\in[1.00,1.10]$：分数在 $m_0\approx1.06$–$1.07$ 处峰值约 25,000，
两端掉到 1,000–5,000（约 7 倍）；tuned/SLSQP 的排序在 $m_0\approx1.06$ 翻转。

固定 $m_0$ 的端到端配对评分（200×128）：

| 参考 | $m_0$ | tuned score | SLSQP score | SLSQP/tuned | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| 单 cosh | 1.0862 | 2504 | 4098 | 1.629 [1.398, 2.192] | SLSQP 更好 |
| 两态 | 1.0570 | 18214 | 18119 | 1.089 [0.659, 2.288] | 未检出差异 |
| GEVP | 1.0350 | 6305 | 4425 | 0.701 [0.679, 0.731] | tuned 更好 |

**“谁更好”本身依赖参考质量**：单 cosh 是唯一给出 SLSQP 更好的选择。

> 200 组态 campaign（120 dev / 60 val）后参考**仍然失败**：$m_0=1.0771\pm0.0034$、$\chi^2=90.6/8$；
> 而且 $m_0$ 随统计量漂移（12 blocks 时 1.0862，24 blocks 时 1.0771）。GEVP 平台仍只到 $t=5$。

### 4.5 GEVP 给出的独立证据

- 2×2/3×3 Hankel（powers-of-$H$）没有平台；
- 两族/三族 Gram GEVP（$H^n$、APE-$H^n$、$\alpha=0.4$ 的 $H^n$）在 $t=3$–$5$ 给出
  $m_0\approx1.02$–$1.05$（两族 $1.050\pm0.014$ / $1.036\pm0.017$；三族 $1.035\pm0.010$ /
  $1.023\pm0.024$）；$t=6$ 仍被噪声主导；
- 这些值与两态参考 1.057 在约 $1\sigma$ 内一致，比单 cosh 1.086 低约 2–3σ：**GEVP 证据偏向两态**。

### 4.6 结论

pion 没有可信分数，不是因为“统计量差一点点”，而是因为**参考质量本身在 1.02–1.09 之间不确定，
而分数和排序对它的依赖大于候选之间的差异**。要修，只有三条路：

1. 把 $m_0$ 认证到 $\le1\%$（更多 $t\ge6$ 的统计量，或更好的算符基）；
2. 把 $m_0$ 的系统误差传播进分数（`scoring.py` 的证书框架，对模型集合取同时下界）；
3. 换一个参考更稳的可观测量（能量误差目标，C3 的设计已就绪）。

已建立的工具：`reference.py`（参考 spec 冻结）、`scoring.py`（S1 证书）、`gevp.py` + `gram_gevp.py`
（C2）、`energy_target.py`（C3 合成验证）、`m0_scan.py` / `evaluate_fixed_m0.py`（参考敏感性）。

## 5. 负结果清单

| 尝试 | 结果 |
| --- | --- |
| 只拟合形状（pion/学习核） | 目标可被 game，训练损失到 $10^{-24}$，留出不泛化 |
| 严格非负约束（学习核） | 增益被抹掉（0.956×） |
| 逐方向独立系数 | 破坏立方对称，是过拟合 |
| 加长 $T$ 到 32（pion） | 参考仍失败；反射失败是假警报 |
| 模型平均（pion） | 权重退化，不能救参考 |
| 学习核超过 SLSQP | v9 只做到未检出差异，没有超越 |

## 6. 仍然 NOT_RUN

- pion：真实数据的 C2 多 source 测量（约 4× 反演）、C3 独立 anchor、新盲测；
- wilson：更大 $R$、更多体积/$\beta$、与动力学 QCD 的连接；
- optimize_gaussian：更强的函数类（群平均、多层）或更多数据以图超越 SLSQP。

## 7. 文档与复现索引

| 内容 | 位置 |
| --- | --- |
| 跨任务审计 | [`doc/audit_2026-11-18.md`](audit_2026-11-18.md) |
| wilson v3 结果 | [`doc/wilson_loop_v3.md`](wilson_loop_v3.md) |
| pion GEVP/能量目标设计 | [`doc/pion_gevp_design.md`](pion_gevp_design.md) |
| pion 评分诊断 | [`results/pion_projection/score_diagnostics_v2/`](../results/pion_projection/score_diagnostics_v2/) |
| 学习核记录 | [`tasks/optimize_gaussian/README.md`](../tasks/optimize_gaussian/README.md) |
| todo 覆盖报告 | [`doc/todo_coverage.md`](todo_coverage.md) |
| 测试 | `PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider tests/`（104 项） |

## 10. 2026-11-18 晚：按新开工单的审计与实现

这一轮按更新后的三份 `tasks/*/todo.md` 重做，新增结果如下（证据路径都在括号里）。

### pion_projection

- **GEVP 矩阵身份**：sink–sink Gram 与标准 source–sink 矩阵差约 30%，且后者本身有 30% 的
  anti-Hermitian 部分；旧 Gram GEVP 降级为探索性（`doc/gram_operator_identity.md`）。
- **GEVP 统计**：去重 → block bootstrap → replica 平均矩阵 → 秩截断 → 平台；已知谱 100 个独立
  数据集的 95% 覆盖率为 0.95（`gevp_stats.py`、`doc/gevp_stats.md`）。
- **评分契约**：`score_contract.md` + `scoring.py` 强制模型/候选集合、等长 draw、原始正值、
  invalid 比例、近零 baseline、improvement 门槛；反例（未批准模型、广播、负 risk）全部拒绝。
- **联合 resampling**：`joint.py` 在同一 draw 内重拟合参考并评估所有模型/候选；参考 gate 失败 →
  强制 abstain（`results/pion_projection/joint_20261118/`）。
- **双层校准**：完整估计器 24 blocks 假阳性 0.025，fixed-metric 0.095（反保守），6/12 blocks 时
  0.235/0.085；两态真值拒绝率约 0.53（`score_calibration_v1/`）。
- **120-dev 诊断**：秩满但条件数 $10^6$–$10^8$；$m_0$ 随窗口从 1.44 掉到 1.08；两态 profile
  半宽/SE=2.7（`score_diagnostics_v3/`）。
- **nested CV**：5 折、每折重拟合参考/标度/协方差/SLSQP；local/SLSQP 209–750（平均 427±89）
  （`nested_cv_v1/`）。
- **能量估计实验**：单 cosh 偏差 0.0065、覆盖率 0；两态无偏但 SE 0.039；基态消失偏差 0.33、
  覆盖率 0；$N_{\rm boot}$ 增加只缩小 naive SE（`energy_target_v1/`）。

### wilson_loop

- **投影数值边界**：逐矩阵 Newton–Schulz 尺度 + 收敛检查；混尺度 batch 与单矩阵一致（1e-16），
  rank-2 受控拒绝（`audit_20261118_projection/`）。
- **可信加载**：generation、resume 前缀、gauge 头部几何、`trust_status`；v3 归档判
  archived_diagnostic（`audit_20261118_identity/`）。
- **配对审计**：v2 点值 1.07–1.18 vs 嵌套中位数 3.0–3.4；6 blocks 的唯一 block 分布、jackknife
  0.49–4.53、相关性 0.14–0.30 解释了差异（`paired_audit_v1/`）。
- **谱质量**：合成“平但错”反例偏差 170σ、risk $2.7\times10^{-7}$、gate 通过；v3/v4 窗口
  $V$ 图（`spectral_quality_v1/`、`doc/wilson_spectral_quality.md`）。
- **HMC**：manifest 审计（seed 20261109、τ_int=0.512），但 ΔH/接受率/热化声明缺失；toy SU(3)
  积分器可逆性 $10^{-10}$、$\Delta H$ 二阶（`hmc_quality_v1/`）。

### 公共框架

- F0 证据 schema、F1 五类接口与跨入口一致性（4 个下界差值 0；改数据/改协议均 fail-closed）、
  F2 候选校验与搜索/validation 隔离、F3 状态机与成本账本、F4 审计（`doc/interfaces.md`、
  `doc/agent_isolation.md`、`doc/run_state.md`、`doc/evidence_framework.md`）。

### optimize_gaussian

- 完整 BCB 闭环验证通过（非立方格点、多 theta、$f\ne g$，残差 <1e-10；
  `audit_20261118_bcb/`）。
- P0.2（basis/输入身份闭包）与 P0.3（audit 分类/措辞）已做；README 的“打平/追平”已改为
  “未检出差异”。
- 包 A 切片：`train.validate_inputs`（gauge/propagator 摘要、residual、metadata 不符即拒绝）；
  manifest 记录 inputs/environment/reference 状态。
- 包 B 切片：point/replica 同幅度算法；synthetic regression 偏差 −1.9e-6/−1.0e-5、
  q=1 覆盖 1.000/0.983。
- 包 C：独立 train/test links、同一 $Q[U]$ 规则；oracle 显示嵌套类跨 links 误差 0、
  谱类 0.06→0.5；SLSQP/Powell 不收敛（nfev 1.3e4–9.4e4）。
- 包 D：OperatorContract/MeasurementBundle/SearchAdapter 的 CPU 端到端适配；能力拒绝抛错。

### 仍未做（NOT_RUN）

pion 的 §6.2/§7 执行（真实 anchor、共同目标预测、新测量先决条件）；wilson 的 P4 研究扩展、
生产 HMC ΔH/接受率、多 rank；optimize 的 GPU class×optimizer 2×2、消融、新盲测、真实 W 训练。

## 11. 2026-11-18 深夜：新工单的 CPU 闭环

- pion：`reference_gate_calibration_v1/`（dof=8，假拒绝 0.02/0.03、失配功效 0.66/0.59）、
  `certificate_calibration_v1/`（假认证 3/8 @6 blocks、0/7、0/8；J=2/K=2 族 null 0/7）、
  `anchor_calibration_v1/`（naïve null 27–50%，独立 dataset 控制）、`joint_identity_v2/v3/`、
  `certification_v1/`、`nested_cv_v2/`、`gram_stats_v2/`；
- wilson：measure 入口 bug 修复（`test_measure_entry_end_to_end`）、三文件恢复事务、
  投影失败 batch 索引、NERSC checksum trailer、`paired_audit_v2/`（点值重放、leave-one-block-out、
  非退化 coverage 0.983）、`spectral_physical_v1/`（两态反例 bias/SE 32.5）；
- optimize：包 A/B/C/D（`cpu_next_package_b/`、`cpu_next_p3_crosslink/`、`cpu_next_package_d/`）。

三份 todo 的逐项状态见 [`doc/todo_coverage.md`](todo_coverage.md)；当前 104 个 CPU 测试通过。
