# pion_projection：第二轮验收后的剩余工作

核查日期：2026-11-18；基线：`21a96cf`。**保留本文件：PARTIAL_NOT_FINAL。**
本轮实际复跑 **80 项 CPU 测试，全部通过（5.03 s）**；另做了不落盘的接口反例和归档检查。没有重新进行 GPU/MPI 测量。
当前阻碍结项的是下列具体实现/统计缺口；无需重复已修旧 bug，也不要求研究必须产生正收益。
本轮只更新三份 todo，历史结果保留。公共实现由一名集成负责人维护，接口与跨任务验收见 [Wilson 开工单](../wilson_loop/todo.md)，学习研究见 [Gaussian 开工单](../optimize_gaussian/todo.md)。

## 1. 本轮完成了什么：从活跃待办关闭的部分

| 上轮项目 | 本轮核查到的交付 | 关闭范围 |
| --- | --- | --- |
| Gram 测量对象 | [算符推导](../../doc/gram_operator_identity.md)、`gram_identity_check.py` 与 toy 对照 | 已明确 `<K_i S,K_j S>` 的 sink Gram 结构及其与标准 source–sink 矩阵的区别；关闭“补推导/反例”，不认证独立 anchor |
| GEVP 估计顺序 | [gevp_stats.py](gevp_stats.py)、[结果说明](../../results/pion_projection/gram_stats/README.md) | 去重复、replica 矩阵均值后解 GEVP、rank 截断与 forward/cosh 对照已做；旧逐配置能量平均不再是新路径 |
| GEVP 已知谱例子 | 100 个独立合成 dataset 的覆盖率记录为 0.95 | 关闭这一固定生成族/rank 的 benchmark；不是全部噪声/近简并/选择策略的校准 |
| 条件评分数组契约 | [scoring.py](scoring.py)、[score_contract](../../doc/score_contract.md) | 模型/候选名称、一维等长、原始正值、baseline阈值、invalid 后撤销认证已有检查 |
| 联合抽样 | [joint.py](joint.py)、[joint 产物](../../results/pion_projection/joint_20261118/README.md) | 共享 dev/val 外层索引、replica 内重拟合参考，200×64 已运行；当前失败参考能触发 abstain |
| 200 配置诊断 | [score_diagnostics_v3](../../results/pion_projection/score_diagnostics_v3/README.md) | 120 dev 的谱/窗口/profile 图已交；不能再说只审过旧60 dev |
| 折内训练 | [nested_cv.py](nested_cv.py)、[fold_plan](../../results/pion_projection/nested_cv_v1/fold_plan.json) | 每 outer training fold 重新拟合参考/通道尺度、选幂次、训练 SLSQP；不再复用完整60配置的参考 |
| 能量估计器 | [energy_experiment.py](energy_experiment.py)、[已有实验](../../results/pion_projection/energy_target_v1/README.md) | 已从合成 correlator 拟合能量并统计 bias/variance/MSE/coverage；anchor 证书部分另见 C |
| 新测量准备 | [计划](../../doc/pion_new_measurement_plan.md)、`measurement_plan.py` | 设计与顶层 schema 草案已交；不是生产输入隔离已经执行 |
| 参考 factory/反射 | 上轮修复与回归仍在 | 无须再修原 T32 索引、旧参考类别选择和旧 NaN/floor 问题 |

本 task 仍研究固定 point source 下的零动量 pseudoscalar sink；几何 `[x,y,z,t]`，空间 links `(3,z,y,x,3,3)`，field `(z,y,x,color,nrhs)`。
κ=0.14，空间周期、费米子时间反周期，meson 周期 cosh；质量为 am。核对 quark field 线性，signed overlap 不必非负。
`K=Σu_n H^n/A_n` 与把同一组合再次作用到两腿不同；新 source/cross 测量须显式写 Wick/指标约定。

## 2. 当前图文诊断及可信边界

![新 GEVP 诊断](../../results/pion_projection/gram_stats/gevp_bootstrap.png)

左图比较原始/去重复的 metric 谱，右图给出 t0=2 的 forward 与 cosh 估计；对应数值见 [gevp_bootstrap.json](../../results/pion_projection/gram_stats/gevp_bootstrap.json)。
它支持“去重复与有限 T 处理已有实际实现”，能量随时间仍有明显趋势。下述部分失败统计和子空间选择尚未完成，因此这些读数不是已认证的独立基态质量。

200 配置 campaign 的开发集有24个长度5的 blocks；[原运行结果](../../results/pion_projection/s16t16_gevp200/validation_results.json) 仍给出名义窗口 `[5,6,7,8]` 的 `am=1.077112714±0.003357843`、`χ²=90.641168/8`，参考未通过。
4–8/5–8/6–8 的样本 covariance 已分别满秩15/12/9；raw 条件数约 `3.55e8/5.54e7/8.41e6`。它们不是“12 blocks 秩亏”的旧数据，也不能由满秩或高条件数单独证明参考成立/不成立。

[能量实验 JSON](../../results/pion_projection/energy_target_v1/energy_experiment.json) 的80次独立数据模拟：single-cosh bias≈0.006458、coverage=0；two-state bias≈−0.001288、coverage=0.9875；弱基态 overlap 例子 bias≈0.33145、coverage=0。
这是有价值的失配 benchmark。`ground_state_missing` 实际仍给基态 **0.02 倍** overlap，应描述为弱基态/近乎被激发态支配，不能将字段名当严格零 overlap。
旧报告的“427±89”来自重叠训练集的五折结果；这是折间描述，尚不是独立重复实验的标准误或可发布的物理增益。

## 3. 必做包 A：冻结认证与联合 draws 真正连接（优先 CPU）

### A1. 认证标志必须对应所用模型、数据和分析

**本轮实际反例。** 保持 `results/pion_projection/s16t16_twostate/` 原件不变，调用 `joint.run(models=('single',), outer=4, inner=8)`，将所有输出函数 mock 为不写文件。
旧两态报告 `am=1.057、χ²=0.0583/4、reference_passed=True` 被直接沿用；单态新分析返回 `status=certified`，两个候选均 certified。
这是 gate 绑定缺口，不是单态模型得到重新认证。

[当前 joint](joint.py) 只读 `validation_results.json` 的一个布尔值；`accepted_models` 临时取自 CLI。
[当前 scoring](scoring.py) 明确只负责 conditional summary，这个设计可保留；认证责任必须由上层强制履行。

- [ ] 输入已冻结 `score_spec` 和**每个模型**的 gate artifact，包含 model/window/prior、reference源码、开发数据、候选、metric 与协议摘要。
- [ ] joint 在读取/重采样前核对 submission、dev/val IDs/完整性/hash、依赖、gate 的身份；拒绝借用旧模型/旧 run 的通过状态。
- [ ] 参考接受集合只在 development 确定；空集合明确 abstain，不能用 CLI 随意添加有利参考。
- [ ] 同一认证路径覆盖 `experiment.evaluate`、generic candidate、离线 joint；历史探索允许另行输出 diagnostic_only。
- [ ] 每模型核对精确候选集合，包括第二及以后模型，拒绝多余/缺失项；spec 中计数、阈值和空 draw 要验证类型与 finite。

**拟议产物：**新目录 `results/pion_projection/audit_<new_id>/joint_identity_audit.json` 和负例文字表。
**验收：**上述单态借两态 gate、旧报告、改数据/候选/metric/窗口均被拒绝；合法同一身份可重放；不能靠硬编码“所有情况都abstain”通过。

### A2. draws 需要可验证的来源和配对语义

现有 NPZ 保存 dev/val outer indices，但没有保存 inner indices；报告没有完整输入与冻结spec摘要。
`draw_ids_hash` 只有在调用者传了 draw_ids 时才校验，省略它即可绕开；单一标签向量也不能证明某候选数组未被打乱。

- [ ] 在需要证书的路径强制 draw identity；保存内外索引，或完整 seed、RNG/算法版本与可重建输入身份。
- [ ] 每个 model/candidate 数组绑定相同 draw axis 和生成来源；缺失/重复/重排/长度不符明确拒绝或按冻结规则重排。
- [ ] 与公共 Comparator 统一严格一维/等shape规则。当前公共 `check_draws` 只看 size 的问题由框架负责人修复，本 task 提供回归。
- [ ] 明确参考、幅度、covariance 哪些在外层重估、哪些冻结；输出失败draw及原因，不能丢弃后冒充全流程。
- [ ] `point_risks` 当前保存 outer risks 的均值，应更名为 bootstrap mean，并另存真正原样本点值。
- [ ] joint 默认写原 run、可能覆盖文件：新增显式新输出目录与存在性保护；正式 evaluate 的持久化也要原子且可重放。

**拟议产物：**`joint_manifest.json`、`joint_draws.npz`、配对数据流图/文字、乱序与省略ID测试表。
**验收：**同 seed/输入/算法复现全部 draws；单候选乱序不能静默获得同一证书；认证失败顶层与候选状态一致。

近期可保留的指标为 `D_cm=log(R_bm+δ²)−log(R_cm+δ²)`，每模型/候选构造预声明的同时单侧下界，再取模型间最小值。
它只说明冻结可接受模型集合下的条件风险改善；Bonferroni 不能弥补 bootstrap 覆盖失效。δ、实际收益阈值、比较族在验证前确定。
没有可靠参考就 abstain；不能通过改分数显示尺度或删掉不利模型恢复物理排名。

## 4. 必做包 B：统一估计器与失败语义（CPU）

### B1. CV 的训练状态必须真的冻结

当前 `nested_cv.py` 的 test risk 没有传 training 的 late_inverse，因此在 outer-test blocks 上重估幅度 metric。
这偏离既定的开发冻结度量契约；拟合验证幅度可以是预先定义的估计器组成，但不能顺便重新选择其 metric 而仍称完全冻结。

- [ ] 在每个 training fold 按候选计算 late_metric，显式传给 held-out risk；点值与 bootstrap 使用同一套已声明的幅度/协方差估计规则。
- [ ] 保存 fold 的 train/test configuration/block IDs、reference/尺度/metric、候选weights、预算/收敛、代码/protocol hash；消除当前 `protocol_hash=None`。
- [ ] 本包先交付**固定分析策略的 outer cross-fit**；若宣称 nested model/window/λ/seed selection，必须另有真实 inner folds。固定选择不是错误，但名称和结论要对应。
- [ ] 图和报告区分 fold间散布、bootstrap估计误差、独立重复实验误差，不对重叠fold简单除 sqrt(K) 当独立SE。

**拟议产物：**`fold_metrics.json`、数据依赖图、修正前后 fold risk/失败率图。
**验收：**修改 test 内容不影响 training reference/scales/metric/候选；同一输入经主入口和CV的冻结评测得到一致统计量。参考失败的fold如实保留。
学习核完整重训由 Gaussian task 接入，不能先声称全部方法的nested CV完成。

### B2. GEVP 固定平台和失败计数

新实现的核心估计顺序已修，不重做旧修复；剩余问题在 `summarise` 的 `nanmean`。
它允许不同 replica 使用不同有效时间点，只有整列全NaN才记failure。
本轮小数组反例：3 replicas中有2个各缺一个平台时间，仍报告 `replicas=3, failures=0`。
cosh求根失败产生的NaN也未统一纳入bootstrap失败列表。

- [ ] 固定平台时间集合；任一所需time/mode无效即按预声明整replica策略处理。分别统计局部失败、整replica失败、有效时间/模式数。
- [ ] rank 指定也须检查所选 metric eigenvalues 的正性/可识别性；拒绝将负值开方或补正当合法模式。
- [ ] 选择 development 固定子空间，或明确定义每次外层重新选子空间的自适应估计器并校准；补 mode matching，不能在replica之间混合不同态。
- [ ] 对 noise/gap/t0/rank/cutoff 做有限预声明网格；加入近简并、相关噪声、cosh求根失败和结构重复算符。

**拟议产物：**`gevp_failure_map.png/json`、每replica有效时间图、覆盖率及MC区间文字表。
**验收：**部分失败反例被正确计数，不靠nanmean换平台；重复identity不改变可识别结果；固定真实样本量增加Nboot只收敛抽样误差。
原0.95保留为一条已完成benchmark；其余网格可失败/INCONCLUSIVE，失败边界必须能复现。

## 5. 必做包 C：真正校准评分与能量差（CPU，不依赖真实 anchor）

### C1. 先修参考 gate benchmark 的定义

[calibration_full.py](calibration_full.py) 已为每个独立dataset重估模型，解决了上轮复用原 samples 的问题。
但它以 `chi2.sf(cost,dof)` 直接计算 p；没有每dataset的内层null模拟。400是外层dataset数，不能当内层校准p的分辨率。
当前 nominal 有3通道×4时间=12维，拟合3个幅度+1个质量，应与 `ReferenceFit` 的 dof=8 对齐；脚本 `dim−2` 给出10。
脚本还直接对 raw 分块，没有复用正式 fold 流程。因此已有 full=0.025/fixed=0.095 仅是当前脚本输出，不能作正确全流程校准证据。

- [ ] 使用与正式分析相同的 fold、尺度、covariance、参数计数、窗口/模型规则和实际gate。
- [ ] 明确两个不同实验：渐近χ² gate 的独立dataset频率评估；每dataset内null模拟校准p后的外层频率评估。
- [ ] 分开记录 `N_dataset`、`N_null`、`N_boot`；`(k+1)/(N+1)` 用于哪一个估计量写清楚。二项比例可带MC区间，但不自动等于bootstrap证书覆盖率。
- [ ] 真实模型正确时报告错误拒绝率；失配模型下报告检测功效，不能都叫false positive。
- [ ] temporal correlation 与 configuration/Markov correlation 分开生成；当前 `rho` 沿Euclidean time，不是配置自相关。

**拟议产物：**新 `reference_gate_calibration.json/png` 与新旧算法差异表。原图保留并标明旧定义。
**验收：**参数计数与主代码一致；已知可控情形解释名义水平；失败样本、不同blocks和MC误差完整呈现。

### C2. 校准真正的 candidate 改善证书

- [ ] 每个独立dataset执行训练参考/选择、候选生成或预声明冻结、joint、gate、certificate整条链，而不是直接给人工risk数组。
- [ ] 比较无改善、已知改善、失配、错误激发态参考；采用6/12/24及功效所需blocks，明确J候选/K模型的同时检验族。
- [ ] 保存 reference接受率、abstain率、无条件假认证率、有效条件下覆盖/功效及失败原因；不能只统计被接受的样本。
- [ ] 同一模拟随机数配对候选；不让候选访问测试真值。模型集合不足以识别E0时正确结果是拒绝认证。

**拟议产物：**`certificate_calibration.json`、假认证率/功效/abstain随blocks与失配的图，带MC区间。
**验收：**达到预声明统计容差或明确记录失败范围；不宣称联合区间普遍有效。此项 CPU 即可，不得标为只能等待新GPU。

### C3. 修正 anchor 敏感性实验的伪重复

`energy_experiment.anchor_sensitivity` 把一次dataset得到的 `e_b/e_c/anchor` 各复制200次，再交给 `lower_bound`。
由此 delta标准差为零，所谓certified_fraction实际退化为点差为正比例，没有传播拟合和anchor误差。
已有 `energy_bias_variance.png` 左侧的估计器benchmark可保留；右侧anchor“证书”和对应文字须更正。

- [ ] 从同一个候选/基线dataset的共同block draws传播能量差；独立anchor另行抽样，共用一个anchor draw比较所有候选。
- [ ] bootstrap estimator分布的物理误差不能再除 sqrt(Nboot)；独立dataset均值的SE是另一个估计量。分开测MC误差与物理抽样误差。
- [ ] 对anchor sigma、bias、共享配置相关性扫描；加入真正零改善与弱/零基态overlap反例。
- [ ] 用外层独立dataset校准区间，而非仅绘制点差正负。补MC区间、所有失败与有效样本单位。

**拟议产物：**`anchor_calibration.json/png`、估计量语义表和重复常数数组拒绝/退化处理测试。
**验收：**固定物理数据增加Nboot不会令证书任意变精确；无改善时假认证率受控；anchor偏差影响清楚记录。

若anchor独立无偏且指向同一E0，`E[(Ehat_c−Ehat_*)²]=MSE(Ehat_c)+Var(Ehat_*)`，可比较
`Δ_c=[(Ehat_b−Ehat_*)²−(Ehat_c−Ehat_*)²]/τ_E²`，避免近零风险比。
共同anchor方差只在相应期望差中抵消；共享配置、anchor偏差、模型选择误差都需传播。
这是一条待校准的物理主指标路线；真实anchor缺失时保持NOT_RUN，不将toy胜出转成正式QCD排名。

## 6. 新测量与可选扩展：保留计划，另行执行

现有 [measurement_plan.example.json](measurement_plan.example.json) 是草案；validator只验字段存在。
在生产前还须验证dev/val/anchor的ID唯一与隔离、非占位hash、geometry/source范围、冻结记录、质量/停止规则，并由实际measurement入口读取和强制校验。
输出计划验证文字表，证明重叠IDs、占位hash、无来源anchor会拒绝；通过schema不等于GPU生产完成。
新测量依旧需12 RHS残差、gauge/prop/source摘要、实际QUDA/设备/精度、HMC与配置自相关证据，先做小体积fast/generic对照。
多source在配置内聚合、配置外分块；新增cross/source矩阵需真实测量或严格恒等式。

以下为**可选后续研究，不阻止本轮CPU包以负结果结项**：

| 方向 | 最小下一判别与图文产物 |
| --- | --- |
| 独立基态anchor | 完整source–sink矩阵或长时高统计分析；给识别性/窗口/系统误差图与隔离证据 |
| 共同目标预测 | 固定同一target与训练metric，比较cross-fitted预测残差；缩幅/方差膨胀反例，不冒称ground-state质量 |
| 质量约束成本 | 固定bias/coverage/SE门槛，再给总成本和增量sink成本曲线；达不到质量时unresolved |
| 多动量/多系综/后端 | 重审little group、相位/复共轭、物理smear半径及跨后端残差；一次只扩一个轴 |

## 7. 下一 agent 的有限交付与结项规则

**顺序：A认证与draw身份 → B估计器/CV与GEVP失败 → C完整校准。**
先修正确性，再重跑相关诊断；不先生成更多gauge field或扩大候选网络。
每包可独立提交，但共享schema由框架负责人协调，避免重复实现。

所有新结果写 `results/pion_projection/audit_<new_id>/` 等独立目录；文中的新文件名都是**拟议产物**。
至少交付：数值JSON/NPZ、图或带数值的文字表、3–6句解释（观察/支持/限制/下一判别），以及命令、输入/代码hash、seed、样本单位、阈值、失败计数和PASS/FAIL/NOT_RUN/INCONCLUSIVE。
图必须注明条件分析/真实数据/合成数据、区间含义，不能只贴PNG或“80 tests通过”。

同步更正 [覆盖报告](../../doc/todo_coverage.md)、[STATUS](../../STATUS.md)、[总报告](../../doc/final_report.md) 与相关诊断README：
本轮已完成的具体切片照实保留，A/B/C未过验收不记为P0/P1全部完成；不要把outer cross-fit写成全链nested selection，或把渐近p的外层实验写成完整评分校准。
只在A/B/C约定范围有代码、反例、诊断和有限结论，且新测量明确独立移交后关闭本todo；正收益不是结项条件。
