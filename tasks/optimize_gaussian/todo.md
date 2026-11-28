# optimize_gaussian: trainable-W smear 对照研究 handoff

## 2026-11-25 我自行选定的 D04/D06/D11 输入

用户已要求我自行给出研究输入。下面冻结为可实现的**候选 protocol defaults**，覆盖前文要求研究者选择 D04、D06、D11 的段落。它们是方法选择，不是数据结果、物理授权或 GPU 批准；缺失的 receipt 和资源门仍必须保持打开。

### D04：物理 reference 与风险目标

- **Observable/reference：**S16T16、quenched Wilson、`kappa=0.14`、零动量最低 valence isovector pseudoscalar 的 `a_t m_{pi,0}`；用单独 E-chain 的 candidate-blind 多通道联合谱拟合，通道为 PP `n=0,8,16` 与局域 A4P，共享两态能量、通道振幅独立，PP 偶折、A4P 奇折，主拟合 `t=3..8`，窗口/模型检查 `t_min=2,3,4` 与二态/三态。
- **优化量：**沿用 Pion fixed-protocol 风险，单位 dimensionless：`q_theta(t)=C_theta(t)/(A_theta f_0(t))`，`R_theta=mean_{t=2..5}[Var_boot(q_theta(t))+|E_boot q_theta(t)-1|^2]`；振幅在 `t=6..8` 拟合，`f_0` 使用独立 anchor 的 `a_t m_{pi,0}`。所有 arms 使用相同时间窗、相关协方差、重采样块和 reference draws。D04 通过门为相对质量误差≤5%、`p>=0.01`、PP/A4P 与窗口/模型位移≤2.5 paired SE、无边界命中；失败则没有 D04 risk，也不以 operator-level synthetic loss 代替。
- **数据来源：**参考定义可由现有 S16T16 协议确定，但数值只来自独立 E-chain；当前 E correlator/receipt 不存在于可审阅树，现有单-cosh pilot 门失败，所以 D04 measurement gate 仍未通过。

### D06：minimum meaningful effect 与推断

- 两个已选比较保持不变：trainable-W vs same-budget SLSQP；trainable-W vs fixed degree-4 Bernstein/Chebyshev 强谱基线。
- 定义 `rho_j=R_comparator,j/R_trainableW`，`rho>1` 有利于 trainable-W。**选定 `r_star=1/0.90=1.111111...`**，即至少 10% 的相对风险下降才算有意义；该阈值是考虑学习训练/复杂度成本的事前判据，不从 synthetic `10%` 结果推断。
- 比较族 `F=2`，总单侧 `alpha=0.05`，Bonferroni 每个对比 `alpha=0.025`。使用相同配置 IDs 的 paired block bootstrap（10,000 draws，length-5 moving blocks；length 2/10 只作预先规定的稳定性诊断），分别给风险比单侧下界；报告三个 block length interval 的 hull `[min(lower),max(upper)]`，只有 hull 的下界也高于 `1.111111...` 才可通过。任一风险在全样本或重采样中为零/无效时，`log(rho)` 不可定义，标 `ABSTAIN_RATIO_UNDEFINED`；不临时加 epsilon 或改绝对差门槛。

### D11：固定强谱 baseline 合同

- 空间平移 `T_i` 使用 `(z,y,x,3,nrhs)` 字段与三个周期方向、等权 unitary 3×3 links。定义负协变拉普拉斯 `H=sum_i[T_i+T_i^dagger-2I]`，故 `H=H^dagger` 且 `spec(H) subset [-12,0]`；`y=I+H/12 in [0,1]`。
- 强谱核固定为 `P4(y)=sum_{i=0}^4 b_i*C(4,i)*y^i*(1-y)^(4-i)`，`0<=b0<=b1<=b2<=b3<=b4=1`。这给 `0<=P4<=1`、对 `y` 单调不减、`P4(1)=1` 的低模归一化，因而在上述 Hermitian H 下 PSD。拟合只优化 `b0..b3`，无事后 clipping/projection；超域、非 unitary links、错误 shape/boundary、非有限值或不可行系数均 fail closed。
- 外层 Gaussian sandwich 固定为 `G^2 P4(H) G^2`，`G=I+0.10*sum_i(T_i+T_i^dagger)`，`m=2`、不做每步归一化。其谱位于 `[0.4,1.6]`；raw 零模总因子为 `(1+6*0.10)^(2m)=1.6^4=6.5536`。Pion 风险对共同非零整体振幅不变；报告仍须记录该因子和数值范围。trainable-W 与 SLSQP arms 使用同一固定 sandwich，避免把外层预处理差异混入比较。
- 强谱基线就是这个固定 degree-4 单调 PSD family；不得在看到 confirmation 后改次数、窗口、约束或 Gaussian 参数。矩阵自由/显式小格矩阵一致性可在 CPU 验证；生产 links、D04 receipt、同预算计时、真实 W 训练和 GPU 资源批准仍未提供。

文字流程：`独立 E reference -> 固定 Pion risk -> fit IDs 上分别训练 W、同预算 SLSQP、degree-4 P4 -> 共同 G^2 sandwich -> 锁定参数与候选 -> 新的 paired confirmation IDs -> 两个 Bonferroni risk-ratio 下界都过 1.1111 才算整体成功`。

**交接规则：**本文件下方 2026-11-24 的“等待研究者冻结 D04/D06/D11”是历史 handoff，现由本节 protocol defaults 覆盖；不要再次向用户问目标/reference、`r_star` 或算符形式。合同、检查器和小格算符测试在[PR #6 head `6afedb178f325c15f62b4786c4ed2f075ebcedd8`](https://github.com/Greyyy-HJC/LatticeArena/tree/6afedb178f325c15f62b4786c4ed2f075ebcedd8/tasks/optimize_gaussian)及[对应 CPU 测试](https://github.com/Greyyy-HJC/LatticeArena/blob/6afedb178f325c15f62b4786c4ed2f075ebcedd8/tests/test_optimize_chebyshev_baseline.py)。证据字段与成本金额仍为空门；后续 agent 不要重复 synthetic teacher/meta-init 网格，先补 E reference、共同 split/seed/cost receipts 与独立审阅，再按权限运行三个 arm。

## 2026-11-25 当前 Optimize 接手点

上文 D04/D06/D11 的目标、风险方向/效应门槛、degree-4 算符与 Gaussian sandwich 已由本轮选定，machine contract 已写入 PR #6。历史 Astra handoff 仍用于记录其时状态，不再构成“等待用户选择”的理由。真实 E reference receipt、数据 split/hash、实测 cost ledger、independent review 与 GPU/QUDA 授权仍未取得，故保持 `NOT_READY/UNCERTIFIED`、`physical_score=null`，不训练真实 W、不打开 sealed confirmation。

## 2026-11-25 自选输入后的 machine contract 与比较规程

- PR #6 的[机器合同](https://github.com/Greyyy-HJC/LatticeArena/blob/6afedb178f325c15f62b4786c4ed2f075ebcedd8/tasks/optimize_gaussian/research_contract_v1.json)与[检查器](https://github.com/Greyyy-HJC/LatticeArena/blob/6afedb178f325c15f62b4786c4ed2f075ebcedd8/tasks/optimize_gaussian/contract.py)已记录 D04 的独立 candidate-blind 联合谱 reference/risk 目标、D06 的风险比方向和 `r_star=1.111111...`，以及 D11 的 degree-4 positive monotone spectral family、`H` 区间、Chebyshev 转换和共同 `G^2 P4(H) G^2` sandwich。未把任何 synthetic 指标写入物理 reference。
- 比较族保留两项：trainable-W vs 同预算 SLSQP；trainable-W vs 固定强 degree-4 spectral baseline。计划 `F=2`、单侧总 alpha=.05、每项 .025，10,000 paired block draws、length-5 primary、length-2/10 稳健性 hull；只有两个 ratio hull lower bounds 都超过 `1.111111...` 才算整体通过。风险无效/为零则 `ABSTAIN_RATIO_UNDEFINED`。
- 调用预算采用已有 CPU operator 对照的已执行上限 160 次 objective-oracle invocation/arm；这是调用上限，不是 walltime 相等的证明。梯度/函数 evaluator 的实现差异、setup、training、fit、solve、inference、I/O 和失败恢复仍须分项计时；`amortization_task_count=1`，不预先给训练摊销优惠。真实成本金额、hardware、fit/eval/split/seed hashes 均为空门。
- 文字流程：`独立 E-chain reference -> 冻结同一 D04 risk -> 在共同 fit IDs 上训练 W/SLSQP/degree-4 spectral -> 160-call cap 与完整成本账本 -> 冻结候选 -> 新 paired confirmation IDs -> 两项 Bonferroni 风险比下界都过 r_star；缺 reference、receipt 或授权即 NOT_READY`。
- Optimize/Wilson 任务专项回归 `tests/test_optimize* tests/test_wilson*` **97 passed**，其中 Optimize contract 7 项、Wilson contract 14 项；同时包含选定 degree-4 operator 与有限温 Wilson 代数测试。这只验证 CPU 软件/公式，不证明 D04 reference、真实风险或 GPU 性能。

**下一位 Optimize agent：**从独立 E-chain fit 开始补真实 reference 和哈希，然后在授权的未封存数据上用三个 arm 执行共同 objective/precision 和 split。记录各自 training/setup/solve/inference/I/O/failure-recovery/total walltime；报风险比 paired interval hull 与 planned/valid/failed/abstained 分母及图/文字诊断。未有数据和资源授权时停止在 `NOT_READY/UNCERTIFIED`，不要重跑已有 synthetic 网格或解封 confirmation。

## 决策状态与问题

- 用户要求同时比较学习型 smear 对同预算 SLSQP 与强谱基线；确认族为 `F=2`。
- 保留“学习型 smear 是否有机会优于 SLSQP”作为研究问题。当前维持 `NOT_READY/UNCERTIFIED`、`physical_score=null` 和数据封存；不读取 gap、validation、shared holdout，也不运行生产 GPU/QUDA。
- trainable-W 是算符参数化；SLSQP/L-BFGS-B 是求解器。确认分析须分别回答“可学习 W 是否增加有限预算下的搜索能力”和“是否优于已固定强谱算符”，不可将二者混成一个结论。

## 已完成的 CPU 诊断

主产物是 [2026-11-23 source-closed replay](https://github.com/DianjunZhao/LatticeArena/blob/6f1aa442d4f8b1121899fb6515bcd5b51501b161/tasks/optimize_gaussian/trainable_w_operator_cpu_replay_20261123_v1/README.md)，包含协议、独立合成输入、逐调用原始 trace、代码快照和图。旧的 [`trainable_w_operator_cpu_20261121_final`](https://github.com/DianjunZhao/LatticeArena/blob/6f1aa442d4f8b1121899fb6515bcd5b51501b161/tasks/optimize_gaussian/trainable_w_operator_cpu_20261121_final/README.md) 保留为历史结果：其 `trainable_w.py` 哈希与同目录源不一致且漏掉导入的 SU(3) fixture helper，因此不能作为闭合 provenance 的主收据。新 replay 修复 source hash 缺口，但不增加独立样本。

| 确认维度 | CPU 诊断内容 | 直接观察到的结果 | 可支持的结论 |
|---|---|---|---|
| 同预算局部搜索 | 三个独立合成 panel；每 panel 3 fit + 3 heldout SU(3)-like links；固定 W 与 joint trainable-W；SLSQP/L-BFGS-B；每次 40/160 次 objective+gradient 调用上限 | 36/36 planned solver runs 有返回，4 次触预算/非成功，0 样本排除。cap=160 的 joint-W SLSQP heldout loss 分别约 `0.00614590, 0.00600509, 0.00601227`；固定 W width-2 SLSQP 分别约 `0.00614590, 0.00600511, 0.00601227` | 这个构造目标上 joint W 的增益至多约 `2.5e-8` 量级；没有稳定、实用的学习 W 优势。三个 panel 不足以给总体区间。 |
| 强谱算符 | 同一固定热核目标上的 degree-1/degree-4 Chebyshev 直接最小二乘；不调 heldout | degree-4 heldout loss 为 `2.40e-7, 3.34e-8, 1.11e-7`，比一跳学习参数化约低四至五个数量级 | 该 fixture 的目标按定义就是 `exp(tau H)` 谱函数，因此强谱基占优是结构性负诊断；不能外推为所有物理 smearing 都由谱基胜出。 |
| 数值正确性 | 压缩二次 loss 对照 dense operator loss；W/v/c 梯度对有限差分；在独立 heldout links 复查 | 最大梯度差约 `1.3e-11`，压缩/dense loss 差低于 `2e-16` | 验证此 CPU 实现和 fixture 的代数一致性，不验证 GPU、物理风险或实现耗时优势。 |

![Trainable-W 与固定谱算符对照](https://github.com/DianjunZhao/LatticeArena/blob/6f1aa442d4f8b1121899fb6515bcd5b51501b161/tasks/optimize_gaussian/trainable_w_operator_cpu_replay_20261123_v1/operator_comparison.png)

文字数据流：`protocol + 固定 seed -> 3 个独立合成 link panel -> 每 panel 切分 3 fit / 3 heldout -> 同一初始化/参数边界/调用上限拟合 -> 仅 fit 选择 -> heldout + KKT/梯度残差作诊断`。优化秒数不含算符/目标构造和后检；因此它是 call-budget 对照，不是 walltime 或训练摊销基准。

### 扩展 CPU 压力测试（已完成，见 companion PR #6）

[Trainable-W stress report](https://github.com/DianjunZhao/LatticeArena/blob/6f1aa442d4f8b1121899fb6515bcd5b51501b161/results/optimize_gaussian/trainable_w_stress_20261123_v4/README.md) 扩展到 36 个合成 panel：三种 link generator、谱热核/非谱局域环境两类目标、0/2% 相对 Frobenius 算符噪声、每格 3 个 replicate。每个 panel 使用相同 fit/heldout links、初始化和边界，对照 joint-W SLSQP、joint-W L-BFGS-B、fixed-W width-2 SLSQP，以及 **同为 7 个拟合标量**的 fixed-W width-6 SLSQP；预算为 32/128 次 objective+gradient call。谱目标另与固定 degree-4 Chebyshev 强基线比较，局域目标另报匹配其构造的 degree-2 基线。后者只是容量诊断，不代替选定的强谱基线。

planned/valid/failed/abstained panel = `36/36/0/0`；solver fit = `288/288` returned，其中 93 次达到调用上限或 solver non-success，均保留，未排除样本。三次重复/格只支持描述性 stress test。width-6 等参数量对照中，六个局域环境 cell 的 joint-W/fixed-W heldout loss ratio 中位数均不小于 1（`log10` 比值中位数均为非负），谱目标则接近持平；强 Chebyshev 基线在谱目标下明显较低。图中单点、cell 中位线、budget 曲线、conditioning 和基线误差均明确标为合成/描述性结果。这不证明固定 W 普遍更优，也不否证学习 W 在真实目标上的可能性；它只没有显示稳定的等参数量优势。

复核材料包括 paired loss 图、32/128-call 收敛图、feature conditioning/基线图、逐 panel 输入与 solver trace、源码快照、哈希清单。输出目录内部 `artifact_hashes.json` 已核对，脚本哈希与报告 source manifest 相符。没有 production 或 sealed data/GPU 输入。

### 跨任务 pooled initialization + 同预算 SLSQP（CPU，companion PR #6）

[新增报告](https://github.com/DianjunZhao/LatticeArena/blob/7d05b0e779eba91a175bb333e91751ee5187eee3/results/optimize_gaussian/trainable_w_meta_init_20261123_v5/README.md) 补查了 per-panel stress 未覆盖的可复用初始化机制：12 个训练 panel 的 fit links 只训练一个 joint-W width-2 共享初值；另 12 个独立 eval panel 对比 pooled learned init、随机冷启动和固定非零初值，然后三组均跑同一个 joint-W SLSQP，调用上限 `8/32/128`。Heldout links 不参与 prior 训练或停止/选择。每个 eval panel 同时拟合 degree-4 Chebyshev（指定谱基线）和固定 degree-2 local-polynomial（该合成教师族的 target-matched control）。

planned/valid/failed/abstained 训练 panel=`12/12/0/0`，评估 panel=`12/12/0/0`，在线 solver run=`108/108/0/0`；其中 36 次达到预算、36 次 solver non-success，两个诊断分别保留，未删样本。pooled SLSQP 使用 15 次跨 24 个训练 link cases 的 optimizer objective/gradient 调用；成本折成每任务调用数是 180，再加一次初值诊断等于 192 panel-equivalent evaluations。以每个 panel 冷启动初始 fit loss 的共同 10%/1% 作为合成收敛目标，pooled init 平均分别用 3.58/4.25 次，cold SLSQP 用 4.58/6.92 次；若未来任务真来自同一分布，按已观测调用节省严格覆盖训练成本的最小 K 分别为 193/73。它们是 12-panel 的算术描述，不是置信下界、物理 `r_star` 或墙钟收益证明。

最大预算下 heldout loss 中位数：pooled init `1.44e-10`、cold SLSQP `1.96e-10`、固定非零初值 `9.13e-10`、degree-4 Chebyshev `7.30e-8`，target-matched local degree-2 为 `3.17e-33`。由于本诊断的教师本来就是局域二次多项式，Chebyshev 在此并非匹配的 oracle family，故报告同时提供了 target-matched local control；后者几乎精确恢复教师。结果只支持“在这组窄合成任务里 warm-start 略省 SLSQP calls”，不证明学习型 smear 胜过物理谱算符，也未比较生产 walltime。

图、逐运行原始 trace、训练/评估输入、两种 baseline、分母表、协议、源码快照与哈希在 companion PR #6。仍缺独立物理 reference、事前 `r_star`、固定的 D11 谱算符/约束、可比 production total-walltime 和授权 GPU 确认，因此状态不变。

### Task-specific contract inspector（结构诊断，companion PR #6）

[合约检查器](https://github.com/DianjunZhao/LatticeArena/blob/031fa1498d5101d7501eca531aefb4837a7866cc/tasks/optimize_gaussian/contract.py) 将已选的两条对照写入 versioned contract：trainable-W 对同预算 SLSQP，以及 trainable-W 对固定 degree-4 Chebyshev 强谱基线；[text diagnostic](https://github.com/DianjunZhao/LatticeArena/blob/031fa1498d5101d7501eca531aefb4837a7866cc/results/optimize_gaussian/contract_diagnostic_20261124_v1/README.md) 列出 D04/D06/D11 与公平预算的缺项。五个 fixture 的 contract-check 分母 planned/valid/failed/abstained=`5/1/4/0`：一个全 `SYNTHETIC_ONLY` 完整形状只到 `CONTRACT_COMPLETE_REQUIRES_HUMAN_REVIEW`；当前 null 草案和三个错误变体保持 `NOT_READY`。5/5 artifact hashes 通过；新增测试 5 passed，PR #6 全套 CPU suite 103 passed。

该检查器不读取数据、不执行 search/evaluator、不认证 reference，也不设置 `physical_score`。当前只把字段和失败条件机械化；它没有替研究者冻结任何物理数值。

## 理论上学习型 smear 可能获益的情形

1. **有限预算搜索：**在同一目标和同一函数族下，精确全局优化器不会被另一个方法理论保证优于；学习器可以通过 warm start、低成本代理、预条件或 proposal policy 更快接近可行解，但需在冻结的 call 与 walltime 预算下实测。
2. **跨任务摊销：**如果一个训练好的 proposal 在多组独立 gauge/operator 实例复用，可摊销训练成本。至少应报告训练总成本、每任务推理成本，并比较 `K*(C_SLSQP-C_infer) > C_train` 是否成立；只有 `C_SLSQP>C_infer` 时才有 break-even 区间。
3. **表达能力与物理约束：**若物理目标依赖非谱局域结构、配置依赖系数、Gaussian sandwich、正定性或可实现的测量风险，learning parameterization 可能以更适合的 inductive bias 优于当前固定 Chebyshev 族。但这需要证明 gauge/cubic covariance、Hermiticity、positivity、归一化和真实 measurement gradients；现有代码没有这些生产合同。

这些是可检验假设。热核合成结果只说明：对谱目标，若不增加谱域/阶数或匹配生产约束，增加同一 synthetic generator 的样本不会改变问题的识别缺口。

## 当前下一位 Optimize agent 开工顺序（2026-11-25 defaults）

1. **D04 reference：**只有独立 E-chain 数据 receipt 与授权到位后，按上文 candidate-blind PP(n=0,8,16)+A4P 联合两态方案拟合并做二/三态、`t_min=2,3,4` 稳健性；5%相对误差、`p>=0.01`、位移不超过2.5 paired SE、PP/A4P兼容且无边界命中全部通过，才解锁 D04 risk。旧 single-cosh/reference 失败不能由 synthetic objective 替代。
2. **共同训练/评估合同：**冻结并 hash objective、fit/dev/test IDs、seed、初始化与 precision；确认三个 arm 使用同一风险定义、输入实例和可比 evaluator。trainable-W、同预算 SLSQP、degree-4 Bernstein/Chebyshev P4 均应用同一 `G^2 P4(H) G^2` 外层 sandwich 对照规则；训练/选型 IDs 与最终 confirmation 不重叠。
3. **公平预算与失败分母：**每臂最多160次 objective/gradient oracle calls、一个预定起点、不作预算外重启；完整记录 training、setup、fit、solve、inference、I/O、failure recovery 与 total walltime。缺失/无效/失败样本留在 assigned-arm planned 分母，风险比为零或未定义即 `ABSTAIN_RATIO_UNDEFINED`。
4. **confirmation：**只能使用研究者/数据 owner 授权且此前未观察的 IDs。采用 `rho=R_comparator/R_trainableW`、10,000 paired block draws、length-5 primary、2/10 敏感性区间 hull；F=2、每项单侧 `alpha=0.025`。两项 hull lower bounds 均须严格超过 `r_star=1.111111...`（至少10%相对风险下降），才支持双重优势。
5. **授权与诊断：**GPU/QUDA 另需资源 owner 授权；当前所有 physical receipts 和 cost ledger 仍空，不启动真实训练或解封 confirmation。每个阶段交付 paired-risk/成本图或完整文字流程，以及 planned/valid/failed/abstained 分母、失败原因和 hashes。

## 历史开工清单（已由 2026-11-25 自选输入更新替代）

1. **D04 reference：**填写 contract 中独立物理目标、单位、关联量、数据来源/receipt、通过与失败门，并完成独立审阅。原有 reference gate 未通过时，所有对照只能写 diagnostic；不能把 proxy loss 改名为物理 improvement。
2. **D06 `r_star`：**由研究者预先填写最小有意义效应、单位、方向和物理解释；不得根据这三个合成 panel、标准误或 optimizer 残差反推。
3. **D11 强谱基合同：**填写 degree-4 Chebyshev 的 operator `H`、谱区间/缩放映射、恒等项和归一化、系数约束、正性/投影约束、失败处理，以及它与学习族之间允许不同的参数维数。旧 source-closed replay 的系数是无约束最小二乘，仅适用于诊断目标；PR #6 另有受约束 degree-4 候选，见本文末尾，仍待 D11 审阅。
4. 冻结完整公平预算：同一 objective、输入实例、fit/heldout ID、初始化规则、候选/重启数；分别计训练、setup、objective/gradient calls、求解、推理、I/O、失败恢复与总 walltime。训练数据不能与确认 panel 重叠。
5. **CPU stress grid 与 null contract diagnostic 已完成**（见上方 companion PR #6）。不要重复同一热核/局域教师网格来增加“样本量”；三个 panel/cell 不支持总体推断。如继续纯 CPU 方法学工作，只新增能区分具体机制的预注册对照，并给出 paired loss、预算/耗时定义、KKT/residual、planned/valid/failed/abstained 分母和图/流程描述。
6. 物理合同与阈值通过审阅并另行授权后，再用独立配置/ensemble 做 `F=2` paired confirmation：trainable-W 对同预算 SLSQP；trainable-W 对固定 strong spectral baseline。两条对照都要报告多重性调整的单侧区间、失败/abstain 和训练摊销；只有两条均越过事前效应阈值才支持双重优势。

## 停止规则与扩展接口

若 trainable-W 只优于同预算 SLSQP、没有优于固定谱基，应表述为有限预算优化/摊销结果，不能称为更优物理 smear；若差异低于事前 `r_star` 或跨独立 panel 不稳定，停止扩大同一 fixture。当前已有独立的 Optimize contract inspector，但尚未接入通用 `TaskSpec`、candidate/operator registry、measurement capability 或 evaluator；框架侧扩展顺序见[可扩展部分地图](../../doc/framework_extensibility.md)。

任何物理结论前都需要保留 `NOT_READY/UNCERTIFIED` 与 `physical_score=null`，直到 reference、效应阈值、operator contract、两条比较预算及数据/GPU 授权全部闭合。

### D11：从现有代码提取的合成算符候选（供审阅，不冻结物理合同）

在 companion PR #6 的 [operator diagnostic](https://github.com/Greyyy-HJC/LatticeArena/blob/75537cbbbef0e1234dbb53a71c81b3a0476ef2f4/tasks/optimize_gaussian/trainable_w_operator_diagnostic.py)、[kernel](https://github.com/Greyyy-HJC/LatticeArena/blob/75537cbbbef0e1234dbb53a71c81b3a0476ef2f4/tasks/optimize_gaussian/kernels.py) 与 [shift convention](https://github.com/Greyyy-HJC/LatticeArena/blob/75537cbbbef0e1234dbb53a71c81b3a0476ef2f4/tasks/optimize_gaussian/features.py) 中，CPU 对照实际构造

```text
H psi(x) = sum_mu [U_mu(x) psi(x+mu) + U_mu†(x-mu) psi(x-mu)]
X = H / 6
P4(X) = sum_{k=0}^4 c_k T_k(X),  T_0=I, T_1=X, T_k=2 X T_{k-1}-T_{k-2}
```

这里 U 是三维、周期边界的合成 SU(3) links，空间轴顺序为 z/y/x，d=0,1,2 分别对应 x/y/z；H 是正反向最近邻平行移动项之和。若 link 矩阵严格 unitary，则每个方向的正向/反向平移算符范数为 1，故由三角不等式 `||H||_2 <= 6`，`X` 的谱落在 `[-1,1]`。这解释了当前代码的 Chebyshev 映射，但结论依赖 unitary links、相同三方向权重和当前周期移位约定；各向异性、非 unitary 的有效链接或不同边界需重算界。

当前系数由 fit rows 上的 `np.linalg.lstsq` 一次性拟合成共享实数 `c_0...c_4`；loss 是对合成、trace-normalized heat-kernel teacher `G=exp(0.18 H)/(tr exp(0.18 H)/dim)` 的平均相对平方 Frobenius operator error。`T_0=I` 只说明恒等基函数被包含，**不表示输出被 trace/行和归一化**。当前系数无约束，代码没有 PSD 投影；它记录设计矩阵 rank，但没有 rank-deficiency failure rule。由于教师本身是 `H` 的谱函数，这只是一个与教师家族匹配的合成谱对照，不证明它是生产中最强或可用的 smearing。

文字流程：`production links + boundary/axis convention -> 明确 Hermitian H 和可证谱区间 -> 固定 affine map 与 degree-4 basis -> 冻结系数是否跨配置共享、如何只用 fit/reference 拟合 -> 冻结输出 normalization 与 positivity 规则 -> 计入 coefficient-fit/setup/inference 全成本 -> 在 untouched confirmation 上比较`。

可供用户审阅的物理合同草案可沿用上述 basis family，但必须另行选择：生产 `H` 的方向权重/链接/边界及谱界；系数是否固定、如何用独立 D04 reference 拟合；是否要求 `p_4(x)>=0` 对全部 `x∈[-1,1]`；输出是否另作 normalization；rank/非有限系数/正性失败如何处理。companion PR #6 已对现有三组系数做连续区间数值审计，但训练代码仍不施加 positivity constraint，contract inspector 保持 D11 字段为 null 是正确的。本节只把合成基线和诊断明示，不填充 null、重命名物理 reference 或关闭 D11 gate。

### D11 正性诊断：现有 degree-4 系数的连续区间检查

我对 [source-closed replay 的原始系数](https://github.com/DianjunZhao/LatticeArena/blob/75537cbbbef0e1234dbb53a71c81b3a0476ef2f4/tasks/optimize_gaussian/trainable_w_operator_cpu_replay_20261123_v1/raw.json) 做了只读复核；输入 SHA-256 为 086f17a9ac7da9869014b554b0a1d364922a0b116d95745d9f9384936031fb9e。对应的 [可复现报告与图](https://github.com/Greyyy-HJC/LatticeArena/blob/337bc99b87952c4c866c6f55ea000427c00273b5/results/optimize_gaussian/chebyshev_interval_positivity_20261124_v1/README.md)、[诊断脚本](https://github.com/Greyyy-HJC/LatticeArena/blob/337bc99b87952c4c866c6f55ea000427c00273b5/tasks/optimize_gaussian/chebyshev_positivity_diagnostic.py)、[区间函数](https://github.com/Greyyy-HJC/LatticeArena/blob/337bc99b87952c4c866c6f55ea000427c00273b5/tasks/optimize_gaussian/chebyshev_interval.py) 与 [专项测试](https://github.com/Greyyy-HJC/LatticeArena/blob/337bc99b87952c4c866c6f55ea000427c00273b5/tests/test_optimize_chebyshev_interval.py) 已提交到 companion PR #6。对每组 Chebyshev 系数求 p4'(x) 的全部实根，再和区间端点 x=±1 一起计算 p4(x)，不靠有限网格采点估最小值：

| panel seed | min p4(x), x in [-1,1] | argmin x | p4(-1) | p4(1) |
|---:|---:|---:|---:|---:|
| 2026112201 | 0.313406899943 | -1 | 0.313406899943 | 2.669528094588 |
| 2026112202 | 0.313027858941 | -1 | 0.313027858941 | 2.670170688984 |
| 2026112203 | 0.313506535758 | -1 | 0.313506535758 | 2.670263798375 |

因此，对这三个 fit 得到的多项式，若 `H` Hermitian 且 `spec(H/6)⊂[-1,1]`，谱定理给出 `p_4(H/6) ⪰ 0`；观测到的最小值约 0.313。该连续区间结论只针对这三组 synthetic heat-kernel panel 的 fitted coefficients，浮点求根结果留有很大正裕量。它没有改变实现事实：`np.linalg.lstsq` 没有施加 positivity constraint，也没有投影；因此不能推出任意配置、物理 `H` 或未来重新拟合的系数保持正性。

文字流程：`frozen fit coefficients -> endpoints + every real stationary point in [-1,1] -> interval minimum -> conditional PSD statement for Hermitian H -> retain physical gate`。若后续合同要求 PSD，可评估把连续区间最小值检查写成 fail-closed rule；容差、舍入裕量和生产谱区间仍需冻结，不能把本次三 panel 结果写成普遍保证。

## 2026-11-24 本轮复核记录

- 在 companion PR #6 源码/测试提交 [f25acaba](https://github.com/Greyyy-HJC/LatticeArena/tree/f25acababd490729328af1189a5934f56bde7f89) 上，Optimize/Wilson 9 个专项 CPU 测试模块：**65 passed**；这不是完整仓库套件，也不验证物理排序或生产 walltime。后续报告提交为 [337bc99b](https://github.com/Greyyy-HJC/LatticeArena/tree/337bc99b87952c4c866c6f55ea000427c00273b5)。
- PR #5 Pion 的 9 份已跟踪报告 manifest / 74 个 payload 文件，以及 PR #6 Optimize 的 4 份 manifest / 38 个文件和 Wilson 的 6 份 manifest / 32 个文件，SHA-256 全部匹配；新增正性报告分母为 3/3/0/0，8 个文件哈希通过。
- Pion 新增的 [bounded block-risk contract inspector](../pion_projection/bounded_block_risk_contract.py) 与 [文字诊断报告](../../results/pion_projection/bounded_block_risk_contract_diagnostic_20261124_v5/README.md) 是 CPU-only schema check；8 个 Pion bounded-risk/dependence 专项模块共 67 passed。v5 还验证 `B` 至少覆盖采样覆盖半径与 common-bias `C` 之和。它不读取物理数据，也不改变三个 task 的 readiness。
- 本轮重新核对三份 task TODO：34 个相对链接均指向 tracked 文件；28 个 GitHub commit/tree 链接跨 7 个 commit tree 均存在。
- 本轮未读取封存数据或运行 GPU/QUDA；两个比较族保持 trainable-W vs same-budget SLSQP 和 trainable-W vs fixed degree-4 Chebyshev，NOT_READY/UNCERTIFIED 与 physical_score=null 不变。
- Optimize 仍待冻结的研究者输入包括 D04 独立物理 reference、D06 事前 r_star、D11 经审阅的算符/约束，以及生产公平预算与数据/GPU 授权；不得用合成目标填写这些字段。纯 CPU 网格和 null contract diagnostic 已覆盖，不要重复同一教师族增加样本数。

## 2026-11-24 companion PR #6 Chebyshev interval handoff

- [连续区间正性报告](https://github.com/Greyyy-HJC/LatticeArena/blob/337bc99b87952c4c866c6f55ea000427c00273b5/results/optimize_gaussian/chebyshev_interval_positivity_20261124_v1/README.md) 与 [`chebyshev_interval.py`](https://github.com/Greyyy-HJC/LatticeArena/blob/337bc99b87952c4c866c6f55ea000427c00273b5/tasks/optimize_gaussian/chebyshev_interval.py) 检查现有 degree-4 Chebyshev 三个 synthetic fit：在 `[-1,1]` 上最小值分别约 `0.3134069, 0.3130279, 0.3135065`，均位于 `x=-1`；分母 planned/valid/failed/abstained=`3/3/0/0`。图展示完整区间多项式，表列端点和驻点得到的最小值。
- **条件结论：**若算符 `H` Hermitian 且 `spec(H/6)⊆[-1,1]`，这三组多项式对应该 synthetic 算符为 PSD。当前 `np.linalg.lstsq` 仍未约束或投影 positivity，数值求根也不是 rigorous interval arithmetic；不能推广到其他 links、生产 `H` 或重拟合系数。
- 这项工作闭合的是“现有三组系数在明确区间内的 CPU 诊断”，不是 D11：D04 独立 reference、D06 最小可接受效应 `r_star`、生产算符/谱界、归一化和失效规则、统计/计时预算及 GPU/data 授权仍需研究者冻结。两项比较继续同时保留为 trainable-W vs same-budget SLSQP 与 trainable-W vs fixed degree-4 Chebyshev；没有授权/输入前不做物理或 sealed 比较，也不重复相同 synthetic teacher 网格。
- 本轮在 PR #6 当前提交 `337bc99b87952c4c866c6f55ea000427c00273b5` 上重跑 9 个 Optimize/Wilson 专项 CPU test modules：**65 passed**。这是代码/代数/合同验证，不是物理排名、生产性能或 GPU 证据。

## 2026-11-24 Pion Markov-score follow-up

- PR #5 的 Pion Markov pseudo-gap 路线新增条件式 `conditional_block_risk_markov_lcb` 和 [v3 诊断报告](../../results/pion_projection/bounded_block_risk_markov_gap_20261124_v3/README.md)。它按 corrected arXiv v5 theorem version 记载 stationary radius，并实现有外部 post-burn-in TV receipt 时的 `F*delta` error-budget 与 `2*delta` mean-shift correction；synthetic 例保留 planned failure denominator。它不估计 Pion block-chain gap/TV 或项目覆盖率，也不改变 Optimize 决策门。
- 此历史快照的 Pion bounded-risk/dependence 专项 suite 为 **78 passed**；后续 theorem-contract closure 的当前结果为 **93 passed**，manifest/hash 与本地 TODO 链接复核见 Pion TODO 最新记录。此软件闭环不修改 Optimize 的 D04/D06/D11 缺项，也不改变 Optimize 两项指定比较及 `NOT_READY/UNCERTIFIED` 状态。

## 2026-11-24 Pion theorem-contract closure

- PR #5 的 [Markov 报告](../../results/pion_projection/bounded_block_risk_markov_gap_20261124_v3/README.md) 与 [contract v6 报告](../../results/pion_projection/bounded_block_risk_contract_diagnostic_20261124_v6/README.md)现明确要求 fixed score function、stationary 起点或有 receipt 的 post-burn-in TV 界，以及 `F*delta < alpha_score`。synthetic fixtures 不估计 Pion chain gap、TV、anchor 或 coverage，也不读取物理数据。
- Pion 的 time-varying `f_i` Eq. (3.26) 路线现已在 PR #5 作为可选 helper/contract 分支实现，要求冻结 `k_ps`、逐 residue class variance-sum 上界、score schedule、`M/k_ps` 和 anchor-shift receipts；它尚不核验 receipt 内容。此 Pion 方法学更新不能填 Optimize 的 D04 reference、D06 `r_star` 或 D11 operator contract。
- 下一位 Optimize agent 按本文件既有顺序继续：由研究者提供可审阅的 D04 reference、事前 `r_star`、D11 算符及正性/归一化/失败规则，再冻结同预算 trainable-W vs SLSQP 与 trainable-W vs degree-4 Chebyshev 的 setup/call/total-walltime 对照；缺字段时保留 `NOT_READY/UNCERTIFIED`，不要重跑相同 synthetic teacher grid。

## 2026-11-24 PR #5 head 71589ab 时的跨任务快照（已被后续记录取代）

- PR #5 当前 head 为 `71589ab`，新增的 Eq. (3.26) time-varying Markov helper、合同分支和 v7 结构报告已提交；Pion 当前专项 suite 为 **116 passed**，已跟踪报告为 **14 manifests / 116 payload files**，哈希均匹配。此更新只同步 Pion 交接，不改变本任务结论。
- Optimize 仍没有研究者可审阅的 D04 reference、D06 `r_star` 或 D11 生产算符假设与最终正性/归一化合同（PR #6 已有待审阅候选）；两项比较仍为 trainable-W vs same-budget SLSQP、trainable-W vs fixed degree-4 Chebyshev。现有 CPU 方法学工作已覆盖，本任务下一步受这些输入和后续测量授权约束，不新增同族 synthetic 网格。


## 2026-11-24 Astra 研究输入审阅与 D11 更新

GPT-6.2-Astra 子代理完成只读审阅后已关闭。研究输入可形成候选合同；它不能代表用户选择 D04/D06，也不能授权 GPU、production data 或 sealed split。状态继续 NOT_READY/UNCERTIFIED、physical_score=null。

### D04：建议先冻结目标，再挑独立 reference

当前代码与合同没有证明要优化的物理量就是 pion mass。若研究目标确认为 pion ground-state mass extraction，可把独立多算符 correlator/GEVP 或经过独立审核的多态拟合作为 D04 候选，并以格点单位 a_t m0 报告；reference 的算符基底、source/sink、时间窗、协方差/重采样、boundary/unresolved-fit 失败规则和原始 receipt 必须先冻结，且不能让 trainable-W/SLSQP/谱臂的结果参与 reference 拟合或选窗。历史 M2 草案是可供方法审阅的起点（输入来源与边界见 [Pion TODO](../pion_projection/todo.md)），不是已验证基态，也不在 PR #5 当前树内。若 D04 目标不是 pion mass，应由研究者明确 observable、单位和物理解释，不能拿合成 operator loss 充当 D04。

### D06：给用户的量化选项，不回填阈值

为避免方向歧义，可把风险比定义成 rho = risk(comparator) / risk(candidate)，candidate 是 trainable-W；rho>1 表示 trainable-W 风险较低。若要表达最小相对风险下降，可选 5%、10% 或 20%，对应 rho_star=1/0.95≈1.0526、1/0.90≈1.1111、1/0.80=1.25。建议先由研究者选一个主要门槛，或对两个对照分别解释门槛；不能用合成结果、标准误或 SLSQP 残差推定。两个确认对比仍同时保留：trainable-W vs 同预算 SLSQP，以及 trainable-W vs 强谱基线。通过规则须检查 paired risk-ratio 的同时区间是否越过 rho_star，并计入 F=2；单位、风险定义、方向和置信水平均需写入 D06；若某臂风险可为零或近零，改用预先定义的 log-risk/绝对差或冻结稳定的分母下限，不能事后加 epsilon 救活比值。

### D11：现有强谱基线候选已更具体，但仍未冻结

PR #6 head 52bc4b7 新增了[约束 degree-4 Bernstein/Chebyshev 候选及响应图](https://github.com/Greyyy-HJC/LatticeArena/blob/52bc4b7/results/optimize_gaussian/chebyshev_constrained_baseline_20261124_v1/README.md)和[CPU 实现](https://github.com/Greyyy-HJC/LatticeArena/blob/52bc4b7/tasks/optimize_gaussian/chebyshev_baseline.py)。在等权、周期空间、unitary SU(3) links 的三维 covariant Laplacian H∈[-12,0] 下，令 y=I+H/12，并取 P4(y)=Σ_i b_i C(4,i)y^i(1−y)^(4−i)，约束 0≤b0≤b1≤b2≤b3≤b4=1。由 Bernstein 基非负且和为一，0≤P4≤1；导数系数非负使响应单调；P4(1)=1 固定低模归一化。该算符族给出了正性、单调性和归一化明确的强基线候选。实现尚未接入 train.main、没有用 D04 拟合系数、没有真实 W 训练或 GPU 路径；不能称为已认证最佳基线，也不能把图里的三条示意曲线说成测量结果。其主要响应图已经随报告保存，说明文字也列出 complete planned denominator=3/3/0/0（仅示意曲线）。

建议先请研究者确认生产 H 的方向权重、边界和 link 假设，外层 Gaussian sandwich 的 alpha/m/归一化，以及此 Bernstein 单调性是否为目标约束；确认后才把该候选写入 D11 合同。继续保留矩阵/矩阵自由一致性与 PSD 条件验证，生产 links、物理 objective 与 coefficient-fit/setup 成本须在同一公平预算中计入。

文字数据流：独立 D04 objective + 生产 H/spectrum → 冻结 Bernstein controls 与 Gaussian sandwich → 同 IDs/objective/precision 下比较 trainable-W、同预算 SLSQP、degree-4 谱臂 → 配对风险、同时区间及训练/推理/总 walltime；没有 D04/D06/D11 任一冻结项就停止在 draft。

**下一步给 Optimize agent：**不再扩大相同 synthetic teacher 网格；先把 D04 reference 候选、D06 风险比方向/选项、D11 新谱基候选并列成一页确认表，引用 PR #6 的新报告。收到明确选择后更新合同并做 CPU 合同/算子验证；在此前不训练真实 W、不触碰封存 split、不申请或假称 GPU 授权。

## 2026-11-24 bounded SLSQP fitter follow-up

PR #6 head [`e6cd1c9`](https://github.com/Greyyy-HJC/LatticeArena/commit/e6cd1c993761c69c4c12d96ecbf60389e733ddf8) adds a CPU `fit_control_points_slsqp` helper to the degree-4 Bernstein strong-baseline candidate and the accompanying [v2 report](https://github.com/Greyyy-HJC/LatticeArena/blob/e6cd1c993761c69c4c12d96ecbf60389e733ddf8/results/optimize_gaussian/chebyshev_constrained_baseline_20261124_v2/README.md). It optimizes only `b0..b3`, fixes `b4=1`, enforces box and monotonicity constraints, hard-caps objective calls, retains the best feasible evaluated point on budget exhaustion, and reports solver status and feasibility residual. Non-finite or non-scalar callback results fail closed. The report includes a text data-flow description and response plot; its five listed payload hashes match.

Verification on that PR #6 head: the two focused Chebyshev modules pass **23 tests**; the broader `tests/test_optimize*.py tests/test_wilson*.py` selection passes **88 tests**; Python compilation and `git diff --check` pass. These are CPU implementation checks. The report remains `REVIEW_DRAFT_NOT_FROZEN`, `physical_score=null`, `data_access=none`; coefficient-fit denominators are `0/0/0/0` because no D04 objective was supplied. The helper is not connected to `train.main`, has not fitted production coefficients, and does not close D11.

文字数据流：`研究者冻结的 D04 objective + fit IDs -> bounded SLSQP over b0..b3 with b4=1 -> hard call cap / best feasible controls / failure status -> separately frozen heldout confirmation -> paired risk and total-cost comparison`。当前只实现并测试了 optimizer plumbing；D04、D06 `r_star`、D11 production operator/positivity/normalization/failure contract、公平总预算和数据/GPU授权仍未齐备。后续先等这些输入，再将 callback 接到双方共用的独立 objective 和 fit IDs；同时保留 trainable-W vs same-budget SLSQP 与 trainable-W vs degree-4 Chebyshev 两个比较。不得用此 helper 的合成测试目标推断物理系数或效益，也不要重复跑同一 synthetic teacher grid。

## 2026-11-24 shared candidate-schema extension

PR #6 head [`133eae8`](https://github.com/Greyyy-HJC/LatticeArena/commit/133eae86eec57b3e8aeefb1cebdb2f242eafde21) registers `optimize-chebyshev-diagnostic-v1` in the shared candidate schema and adds [`candidate_contract.py`](https://github.com/Greyyy-HJC/LatticeArena/blob/133eae86eec57b3e8aeefb1cebdb2f242eafde21/tasks/optimize_gaussian/candidate_contract.py). It validates the illustrative degree-4 Bernstein controls and binds the task adapter, generic schema, Chebyshev implementation and hopping kernel hashes. The [report](https://github.com/Greyyy-HJC/LatticeArena/blob/133eae86eec57b3e8aeefb1cebdb2f242eafde21/results/optimize_gaussian/framework_candidate_contract_20261124_v1/README.md) contains a text data-flow diagram, `schema_fixture=1/1/0/0`, `coefficient_fit=0/0/0/0`, and six verified payload hashes.

Verification on this head: `tests/test_framework.py tests/test_optimize*.py tests/test_wilson*.py` passed **102 tests**; Python compilation and `git diff --check` passed. This closes only the Optimize candidate-schema registration step in the framework map. It deliberately does not create a complete Optimize `TaskSpec`, measurement capability, evaluator, or physical candidate contract; D04, D06, D11 production review and authorization remain open. Report status is `DIAGNOSTIC_SCHEMA_ONLY`, `physical_score=null`, `data_access=none`.

文字数据流：`versioned diagnostic contract -> validate five Bernstein controls -> candidate descriptor with source hashes -> CPU operator helper`；无效控制直接拒绝，D04/D06/D11、evaluator 与 production authorization gates 保持打开。下一步可在 D04/D06/D11 被明确冻结后再连接完整 TaskSpec、同源 objective 的 fit/eval IDs 和 measurement evaluator；在此之前不应把此 adapter 当作生产注册或用它生成物理分数。

## 2026-11-24 shared abstention-state follow-up

PR #6 [`710eeb2`](https://github.com/Greyyy-HJC/LatticeArena/commit/710eeb27b5d94de37a646ee52ec743be18894786) adds a terminal `abstained(reason)` state to generic `RunState`, distinct from a solver/evaluation failure. Optimize's contract inspector and candidate evaluator are not yet wired to it; absent D04/D06/D11 inputs still leave the existing draft `NOT_READY`, `physical_score=null` and no-data status unchanged. The repository-wide `python3 -m pytest -q` run passed **134 tests** on PR #6 head [`710eeb2`](https://github.com/Greyyy-HJC/LatticeArena/commit/710eeb27b5d94de37a646ee52ec743be18894786); this is CPU/software validation, not GPU/MPI physics validation.

文字流程：`D04/D06/D11 receipt 缺失 -> abstained(reason) terminal -> no physical score`; reviewed inputs start a new run rather than upgrading an abstained record in place. The generic state does not decide the effect threshold or validate an operator contract.

## 2026-11-24 shared TaskSpec split preflight

PR #6 head [`3b9c772`](https://github.com/Greyyy-HJC/LatticeArena/commit/3b9c77201be97d300dfce98b0c3e0d67dc57146f) now makes the generic `task_spec` reject malformed or overlapping inclusive split ranges. Optimize still has no complete TaskSpec because its D04/D06/D11 inputs and fit/eval split identities are unfrozen; this core check does not supply those values. The full repository test suite passed **134 tests** on that head.

文字流程：`future frozen fit/eval range descriptors -> generic inclusive-range validation -> task-owned ID/hash receipts -> evaluator`; Optimize has not reached this path, and its two comparisons remain unmeasured.

## 2026-11-24 shared ScenarioSpec/RunReceipt follow-up

PR #6 head [`f209015`](https://github.com/Greyyy-HJC/LatticeArena/commit/f209015337b08f26e34b733771d56142a5288806) 新增通用 [`ScenarioSpec/RunReceipt`](https://github.com/Greyyy-HJC/LatticeArena/blob/f209015337b08f26e34b733771d56142a5288806/core/receipts.py)。Optimize 的 [candidate contract adapter](https://github.com/Greyyy-HJC/LatticeArena/blob/f209015337b08f26e34b733771d56142a5288806/tasks/optimize_gaussian/candidate_contract.py) 和 [v3 文字诊断](https://github.com/Greyyy-HJC/LatticeArena/blob/f209015337b08f26e34b733771d56142a5288806/results/optimize_gaussian/framework_candidate_contract_20261124_v3/README.md) 是首个消费者：`schema_fixture=1/1/0/0`、`coefficient_fit=0/0/0/0`，后者明确表示没有 D04 objective；收据状态 `complete` 仅表示所有计划单位均已入账。该 report 不接 measurement evaluator，也不证明拟合或候选方法有效。

全仓 `python -m pytest -q` 在与 PR #6 head f209015 对应的代码树上为 **139 passed**；这是 CPU 软件验证，不是生产性能、GPU/QUDA 或物理排名证据。收据的 source/artifact hashes 已由 Optimize 测试逐项校验。

**下一位 Optimize agent：**不要重跑同一 synthetic teacher/初始化网格，也不要把 D04/D06/D11 空字段补成示例值。下一步先把研究者候选决策列为待审卡：D04 的独立 reference 路径与 fit/eval ID 规则；D06 的 paired risk estimand、方向/置信水平和最小有意义效应 `r_star`；D11 的 degree-4 Bernstein/Chebyshev 算符方向权重、边界/links、归一化/正性与失败规则。只有这些被明确冻结后，才把 shared receipt 接到真实 TaskSpec/evaluator，并用同一 ID/objective/precision 对比 trainable-W、同预算 SLSQP 与强谱基线；收据须分开记录训练、拟合、测量和 heldout 分母及失败原因，并保留图或文字数据流。封存 split 与 GPU/QUDA 生产授权仍是单独门槛；未获得授权时保持 `NOT_READY/UNCERTIFIED`、`physical_score=null`。

## 2026-11-24 GPT-6.2-Astra 输入与外部授权复核

Astra 的只读协议审阅没有产生负责人批准、数据授权、GPU 资源授权或可直接冻结的 D04/D06/D11 值。现有 [pooled learned-initialization 报告](https://github.com/Greyyy-HJC/LatticeArena/blob/7d05b0e779eba91a175bb333e91751ee5187eee3/results/optimize_gaussian/trainable_w_meta_init_20261123_v5/README.md) 已补足跨任务 warm-start 这一 CPU 方法学空缺，并与同预算 SLSQP 及 degree-4 Chebyshev 对照；这不代替物理输入和确认样本。PR #6 当前 head [`f9be949`](https://github.com/Greyyy-HJC/LatticeArena/commit/f9be949e83fdc6f22e98d2eb1c7ffe068fc9f9ab) 全仓 CPU tests **140 passed**；新增 Wilson 单位校验不改变 Optimize 结论。

**下一位 Optimize agent：**不要重复 meta-init 网格或把模拟 `10%` 目标认作 `r_star`。先分别冻结 D04 reference/fit-eval IDs，D06 paired-risk 方向/失败/低分母/置信与 `r_star`，D11 真实算符的方向权重、边界、归一化、零模与 PSD/单调合同；再给两项比较确定共同 objective、precision、预算与总成本账本。之后只能在研究者/数据 owner 已授权、未观察的独立数据上接 evaluator/确认；GPU/QUDA 还需资源 owner 授权。候选值未审、授权未到时保留 `NOT_READY/UNCERTIFIED`、`physical_score=null` 与 sealed data closed，并在诊断中分开报告训练、拟合、推理/求解、I/O、失败恢复和 heldout 分母。

## 2026-11-24 GPT-6.2-Astra 输入/授权复核

Astra 只读复核已结束，没有产生外部授权或可直接冻结的 D04/D06/D11 值。现有 CPU 证据已覆盖 trainable-W vs 同预算 SLSQP、trainable-W vs degree-4 Chebyshev，以及 pooled learned initialization + 同一 SLSQP 的独立 evaluation panels；[meta-init 报告](https://github.com/Greyyy-HJC/LatticeArena/blob/7d05b0e779eba91a175bb333e91751ee5187eee3/results/optimize_gaussian/trainable_w_meta_init_20261123_v5/README.md) 是 synthetic 方法诊断，不能推断物理风险或生产摊销。

**下一位 Optimize agent：**不要重复同一教师或初始化网格。先请研究者冻结三张决策卡：(1) D04 的目标/reference 来源、observable/unit、独立 fit/eval IDs 和验收/失败规则；(2) D06 的 paired-risk 方向、低分母/失败处理、同时区间、最小有意义效应 `r_star`；(3) D11 的真实 degree-4 算符自变量、链接权重/方向、边界条件、归一化、零模和 PSD/单调约束。用户已要求两项比较并行保留：learned-W vs 相同调用预算 SLSQP，learned-W vs 指定强谱基线。若未冻结或无法授权新的独立物理输入，继续只做已有可复现的 CPU 软件维护，明确报告 planned/valid/failed/abstained 和总成本分解；不要把 synthetic `10%` 门槛当 `r_star`，保持 `NOT_READY/UNCERTIFIED`、`physical_score=null`、sealed data closed。任何真实 GPU/QUDA 比较须另有资源 owner 授权。

## 2026-11-24 继续工作后的授权边界复核

用户对上一轮研究输入/外部授权问题的答复是：保持当前 `NOT_READY/UNCERTIFIED` 与数据封存状态。Astra 候选卡仅供以后审阅，不等于冻结 D04/D06/D11 或批准数据/GPU；因此本轮不把候选值写入 machine-readable contract，也不重复已有 synthetic teacher/meta-init 网格。Optimize TODO **保留**，因为任务仍未完成。

下一位 agent 只有在用户/研究负责人明确更新输入后，才继续依赖这些输入的工作：D04 target/reference、单位与独立 fit/eval IDs；D06 paired-risk estimand、方向、失败和低分母规则、同时置信区间及 `r_star`；D11 degree-4 算符的自变量/方向/权重、边界、归一化、零模和 PSD/单调合同。届时先更新合同和 CPU 诊断，再按共同 objective/precision/call 与总时长预算比较 trainable-W、同预算 SLSQP、强谱基线。诊断至少给 paired-risk/成本图或完整文字数据流、训练/fit/求解/推理/I/O/失败恢复/heldout 分层分母；封存 split 与生产 GPU 仍需独立授权。

## 2026-11-25 最新接手说明

上面 2026-11-24 的“等待研究者冻结 D04/D06/D11”已由用户要求助手自行选择输入的指示取代。选定方法、机器合同、160 次/臂的 objective-oracle 规划上限、文字数据流和 97 项 Optimize/Wilson CPU 回归结果见前面的 2026-11-25 更新；合同已在[PR #6 head `6afedb178f325c15f62b4786c4ed2f075ebcedd8`](https://github.com/Greyyy-HJC/LatticeArena/blob/6afedb178f325c15f62b4786c4ed2f075ebcedd8/tasks/optimize_gaussian/research_contract_v1.json)。当前仍未取得独立 E-chain reference、split/hash、实测 cost ledger、独立审阅或生产/GPU 授权；下一步是补这些证据并按授权执行三臂对照，不能把方法选择当成确认结果。

## 2026-11-25 接手复核：只剩实证与资源门

此前仍要求研究者选择 D04/D06/D11 的段落均为历史记录；本文件开头的候选默认值已按用户指示自选，不应再次询问相同方法选择。CPU 合同、SLSQP 对照、固定 degree-4 Chebyshev 对照、warm-start 诊断及其图/文字分母已完成；当前没有可通过重跑相同 synthetic teacher/grid 来关闭的剩余项。

剩余开放项按顺序为：

1. 取得未观察且获数据 owner 授权的独立 E-chain/reference 及 fit/eval IDs/hash，跑候选盲 reference gate。
2. 在同一批新 calibration/evaluation IDs 上记录 trainable-W、同预算 SLSQP、degree-4 Chebyshev 三臂的 paired risk 和完整失败分母；同一 objective/precision 下计 setup、训练、拟合、推理/solve、I/O、恢复与 total wall time。
3. 由独立审阅者核验 D04/D06/D11 的收据和生产算符合同；需要 GPU/QUDA 的步骤另需资源 owner 授权。

当前没有这些输入或授权。保持 `NOT_READY/UNCERTIFIED`、`physical_score=null`、sealed data closed；不要把旧 D0–99 或现有 synthetic panels 用作独立确认，也不要重复方法网格。

本次在 PR #6 当前 head `6afedb1` 重跑 Optimize/Wilson 专项 suite 为 **97 passed**，全仓 CPU suite 为 **145 passed**。这是软件回归证据，不填补上述数据、独立审阅或 GPU 资源门。

## 2026-11-25 D11 Bernstein-to-Chebyshev 数值实现收口

PR #6 新提交 [`679f62d`](https://github.com/Greyyy-HJC/LatticeArena/commit/679f62daea0caa98ebf44bca486dc4a3a3d02bfa) 将 `bernstein_to_chebyshev` 从 Chebyshev 节点采样加线性求解，改为 `y=(x+1)/2` 下固定的 degree-4 有理变换矩阵。其五行分别为

```text
a0 = ( 35 b0 + 20 b1 + 18 b2 + 20 b3 + 35 b4) / 128
a1 = (-56 b0 - 16 b1         + 16 b3 + 56 b4) / 128
a2 = ( 28 b0 - 16 b1 - 24 b2 - 16 b3 + 28 b4) / 128
a3 = ( -8 b0 + 16 b1         - 16 b3 +  8 b4) / 128
a4 = (  b0 -  4 b1 +  6 b2 -  4 b3 +  b4) / 128
```

矩阵元素是可由 float64 精确表示的二进有理数；符号上的基变换为精确恒等式，最终矩阵向量乘仍按 float64 运算，不宣称机器舍入误差为零。保留 `chebyshev_interval_minimum` 作为旧的无约束系数诊断；选定 D11 方法的正性门来自 `0<=b0<=...<=b4=1` 的 Bernstein 参数化与 `validate_control_points` fail-closed 校验，因此不再依赖对拟合 Chebyshev 系数做浮点驻点搜索来代替约束。

专项转换/合同/区间测试 **34 passed**；PR #6 全仓 CPU suite **146 passed**；`git diff --check` 通过。文本数据流：`校验单调 Bernstein controls -> 固定有理矩阵转 Chebyshev -> Clenshaw 应用 -> H/Hermitian/谱区间与共同 sandwich 检查`。这关闭的是 CPU 转换路径的不必要数值求解；测试不证明真实生产 links、物理 reference、同预算风险胜出、GPU/QUDA 性能或 rounding-level PSD 证书。

**继续边界：**旧 synthetic Chebyshev 三 panel 的连续区间报告仍只描述原有拟合系数；不把它扩展成生产声明。Pion 的真实 score-specific dependence/anchor/reference 收据，以及 Wilson C/F 的独立 calibration/污染界仍需对应数据与独立 review；Optimize 仍需 E-chain/reference、split/cost 收据和生产/GPU 授权。没有新的授权/数据前，不重跑同族 synthetic 网格，不打开 sealed data，三项任务维持各自 `NOT_READY/UNCERTIFIED`、`physical_score=null`（Wilson F 另保留 `GROUND_STATE_FORCE_UNCERTIFIED`）。

## 2026-11-25 Optimize 报告生成器与当前合同同步

后续 PR #6 提交 [`49673bb`](https://github.com/Greyyy-HJC/LatticeArena/commit/49673bbec8044ecd313a4237b4a7d3f3055e75f9) 修正了两个方法选择后仍称 D04/D06/D11“未冻结”的旧报告生成器：现从 `research_contract_v1.json` 载入已选默认值，将剩余事项明列为独立 E-chain/reference 与 split 收据、真实 paired-risk/成本账本、D11 production-link 审阅和 owner 授权。D11 符号上精确的 dyadic 有理变换及其 float64 实现也写入报告合同。

- [D11 图文诊断 v3](https://github.com/Greyyy-HJC/LatticeArena/blob/49673bbec8044ecd313a4237b4a7d3f3055e75f9/results/optimize_gaussian/chebyshev_constrained_baseline_20261125_v3/README.md) 附 `spectral_response.png`、所选 D04/D06/D11 默认值快照、开放 gate、文字流程和分母。图的三条曲线是合法控制点的代数示例，planned/valid/failed/abstained=`3/3/0/0`，不含拟合或物理样本。
- [候选注册图文诊断 v4](https://github.com/Greyyy-HJC/LatticeArena/blob/49673bbec8044ecd313a4237b4a7d3f3055e75f9/results/optimize_gaussian/framework_candidate_contract_20261125_v4/README.md) 收据绑定选定研究合同及源码哈希；schema fixture=`1/1/0/0`，coefficient fit=`0/0/0/0`。它仍不构造真实 TaskSpec/evaluator 或物理 score。
- 两份报告的 payload SHA-256 均匹配；Optimize 专项 **34 passed**，PR #6 全仓 CPU suite **146 passed**。提交 [`679f62d`](https://github.com/Greyyy-HJC/LatticeArena/commit/679f62daea0caa98ebf44bca486dc4a3a3d02bfa) 是固定有理变换本身，`49673bb` 更新了诊断生成器与报告。

该同步关闭的是“自选方法默认值已冻结但报告仍显示待选”的软件/文档不一致。Optimize TODO 仍保留：没有独立 E-chain/reference、split/hash、实测成本、production review 和 GPU/data 授权，不能执行真实三臂比较；Pion 与 Wilson 剩余边界仍分别依赖独立 coverage/reference/anchor 收据、C/F calibration 与污染界。三项不得据此升级状态或读取 sealed data。

## 2026-11-25 本轮继续复核：Optimize 的可执行工作边界

D04/D06/D11 的方法默认值已经自选并写进合同；比较族继续同时保留 trainable-W 对相同调用预算 SLSQP，以及 trainable-W 对固定 degree-4 Chebyshev 强基线。PR #6 head `49673bb` 上 D11 固定有理基变换、两份图文诊断和合同报告已同步；Optimize 专项 34 项及全仓 CPU 146 项记录见上文。本轮没有发现可以靠重复 synthetic teacher/grid 关闭的新方法缺口，因此不再用合成效应替代实证。

后续唯一可升级结果的路径是：取得与开发/确认用途相符且 candidate-blind 的独立 E-chain reference receipt；冻结并哈希共同 fit/eval IDs、seed 与所有 arm 的 objective/precision；在同一目标函数及失败分母下测三臂，逐臂记录 training、setup、solve、inference、I/O、failure recovery 和 total walltime；以 paired risk ratio interval hull 检验已选 `r_star` 并报告 planned/valid/failed/abstained。另由独立 reviewer 检查真实 production links 上 D11 的算符方向、边界、归一化、PSD/约束和舍入误差。真实 W 训练或 GPU/QUDA 比较仍须对应 data/resource owner 授权；未授权时保持 `NOT_READY/UNCERTIFIED` 与 `physical_score=null`。

文字数据流：`独立 E reference + immutable split/seed -> 三臂同 objective/precision/call cap -> paired risk + 全成本账本 -> D06 同时区间过 r_star -> D11 production-link review -> 独立审阅/授权 -> 才能进入新的确认阶段`。任何 reference、split、paired outcome、成本或 production operator 收据缺失，都只记为 blocker，不把合同完整、CPU 测试通过或 toy 优势写成学习型 smear 胜出。

## 2026-11-25 当前接手结论

较早要求研究者选择 D04 reference、D06 `r_star`、D11 算符形式的段落已被本文件开头的自选默认值取代；不得再次询问相同方法选择。Optimize TODO 保留，因为真实 reference、配对 split/cost receipts、production-link review 与授权仍缺失。没有新的、未封存且获授权的输入前，不训练真实 W、不读取 sealed confirmation、不申请或假称 GPU 资源；交付物仍须包括两种基线的并列风险/总成本图表或清晰文字流程和完整分母。

## 2026-11-25 PR #6 RunState 接线续记

PR #6 提交 [`a22cd23`](https://github.com/Greyyy-HJC/LatticeArena/commit/a22cd2304735dbd8e86c6210196f47c79dd47976) 将当前五个物理认证阶段接入通用 `core.RunState`。 [候选合同 v5](https://github.com/DianjunZhao/LatticeArena/blob/a22cd2304735dbd8e86c6210196f47c79dd47976/results/optimize_gaussian/framework_candidate_contract_20261125_v5/README.md) 明确 `operator/reference/numerical/statistical/comparison` 都因缺真实收据或授权而 `abstained`，`physical_ready=false`、`score=null`；独立 gate 分母 `5/0/0/5`。Schema 例子仍是 `1/1/0/0`，系数拟合是 `0/0/0/0`，RunReceipt `complete` 只表示这些状态均已记账。

报告保存 `core/state.py` 快照并绑定 source/artifact hashes。Optimize 合同、候选、Wilson lane 与共享框架四个专项模块 **37 passed**；新诊断仍未提供 D04、D06、D11 的实证 receipts、真实 W 结果或物理 score。后续获授权运行必须创建新的 RunState，不能从本次 abstained 状态直接转为 validated。


## 2026-11-25 本轮可用输入盘点（仅 tracked tree）

我核对了 PR #6 工作树中 Optimize 的跟踪输入包：现有 `.npz` 输入均为 `synthetic_inputs.npz`，位于 CPU replay、stress 和 meta-init 诊断目录；跟踪到的 paired-risk/solver CSV 是这些诊断的汇总。没有发现 production gauge-link 输入、独立 D04 reference receipt、共同 fit/eval ID manifest 或实测 total-cost ledger。盘点只覆盖 Git 跟踪内容，不包含 ignored cache 或外部存储。

下一步须取得独立 E-chain reference receipt、冻结 split/seed/hash 与可比成本收据，再按固定合同对 trainable-W、同预算 SLSQP、degree-4 Bernstein/Chebyshev 强基线做配对测量；生产 links review 与 GPU/QUDA 资源授权仍分别是独立门。

## 2026-11-25 继续推进：D-chain 数据可复用线索与剩余门

本轮只读本机已存在的 `d_chain_propagator_adapter_20261120/README.md` 与 `d_chain_propagator_reuse_audit_20261120/README.md` 摘要，没有打开 raw propagator cache。已有报告记载 D0–99 point propagators 的 hash 为 100/100 匹配、最大 residual `9.997510675285575e-11`，且在 Optimize 专属 RunSpec 绑定 solver settings、cache hashes、objective、代码后，可复用 fit IDs 0–59 与 development IDs 60–99，避免重复 point-source inversion。相同报告明确 Pion reference gate=`False`，并不等价于 Optimize 独立 D04 reference；真实 Optimize scoring 仍为 diagnostic。

本机 `cpu_next_package_b` 报告另记三折内选择均落在 `H^32`，trainable/SLSQP 比值为 `1.719, 2.014, 0.810`；fold 数小、reference gate 失败且没有 inner CV，因此这些只是探索性条件结果。它们不能替代选定合同下 trainable-W、同预算 SLSQP、固定 degree-4 Chebyshev 强基线的同 ID/同 objective 配对风险与总成本账本。上述本机 audit/package 结果位于 ignored `results/`，未随 PR #5 打包；本轮没有重新打开 cache、训练真实 W 或读取任何封存 split。

文字数据流：`旧 D0–99 hash/residual receipt -> Optimize RunSpec 绑定并复用 -> 先解决 candidate-blind reference 与 fit/eval split -> 三臂统一 objective/precision/call cap -> paired risk + 全成本 -> D11 production-link review -> 独立审阅/授权`。当前 RunSpec、独立 reference、成本/失败 ledger、真实 W 训练结果及授权仍缺；Optimize TODO 保留，状态继续 `NOT_READY/UNCERTIFIED`、`physical_score=null`。
