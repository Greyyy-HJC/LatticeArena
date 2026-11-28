# wilson_loop: finite-T Creutz 与基态离散静态力 handoff

## 2026-11-25 我自行选定的 C/F protocol defaults

用户已要求我自行给出 lane 输入。以下选择替代本文件早先“请研究者在两个候选中选择”的待办；它们是预先规定的研究设计，不是污染测量、owner 授权或物理结论。C 与 F 仍是不同 estimand。

### C lane：primary finite-T Creutz

- **主点与 kernel：**当前 pilot geometry/action `16^3 x 16`、周期边界、Wilson `beta=6.0`，`N_t=16`，`R=2,T=5`，主 kernel `spatial_stout_rho0.10_n5`。APE `alpha=0.4,n=5` 和 identity 只作为预先标明的描述性比较，不扩充 primary family。APE/stout 的弱场参数匹配不表示非微扰 smearing 半径相等。
- **Estimand：**主报告 `chi_stout(2,5)=-[log Wbar(3,6)+log Wbar(2,5)-log Wbar(3,5)-log Wbar(2,6)]`，无量纲。它是 finite-`T` Creutz observable；不重命名为 ground-state force。每个共同 paired bootstrap draw 先算四个 loop 的 ensemble mean，再对正均值取 log。
- **推断与精度：**C/F 为两个 co-primary lanes，`alpha_total=0.05`，Bonferroni `alpha_C=alpha_F=0.025`；每 lane 给双侧 97.5% interval，lane familywise coverage 至少 95%。C 的目标 half-width `<=0.02` dimensionless。使用相同配置 IDs 的 10,000 次 paired circular block bootstrap，length 5；length 2/10 作固定敏感性诊断，采用三种长度中最宽的区间。任一必需 loop mean 非正/非有限时保留该 draw 为失败；失败 draws 超过 1% 或未达到 precision gate 则 abstain，不删 draw 或更换主点。

### F lane：基态离散静态力与独立污染预算

- **主点/contrast：**`R=2` stencil，midpoint `R+1/2=2.5`；比较 stout candidate 与 identity reference：`Delta f0=[U0_stout(3)-U0_stout(2)]-[U0_id(3)-U0_id(2)]`。报告单位为 `a_s*a_t*F = Delta_R U0`（dimensionless lattice force），不声称 continuum force。
- **独立污染 evidence route：**每个 `R=2,3` 用 identity/APE/stout 三算符 static-source basis 做 `t0=1, td=2` 的 3×3 GEVP；两态与三态 finite-`N_t` joint spectral fits 覆盖 `t=3..7`，并比较 `t_min=3,4,5`。热项另需匹配同 action、`a_s/a_t`、空间体积和 kernel 的 `N_t=32` 数据；`x` mixed term 用联合模型的同时上界，不准默认为零。谱/热拟合只能在与最终 F confirmation 不重叠的 calibration IDs 上给界，并需保存每个模型、时间窗、失败、截断与参数相关性的图/文本收据。
- **预先接受的误差上限：**以正的 reference ground-state force 的同时下置信界 `f_ref,L>0` 为分母；若不成立则 `ABSTAIN_FORCE_SCALE`。在 lattice force units 中，excited-state `q` <= `1.5% |f_ref|`、finite-`N_t` thermal `h` <= `1.5%`、mixed `x` <= `1.0%`、谱模型/截断 `<=1.0%`，传播到双差后线性相加总系统界 `<=5% |f_ref|`。F 的统计区间 half-width `<=5% |f_ref|`；最坏线性总误差预算为 `<=10% |f_ref|`。finite-a 和 finite-volume 不属于这个固定格距/固定体积 estimand；若要报物理 continuum force，必须另开外推与预算。
- **多重性与传播：**C 占 `alpha=.025`；F 的 `alpha=.025` 分成统计 `.0125` 与污染上界 `.0125`。F statistical alpha 在 `Delta f0` 双侧区间与 `f_ref>0` 下界间各分 `.00625`。将 q/h/x 同时上界覆盖两个 kernels、`R=2,3`、`t=5,6` 的 24 个 kernel×R×t×term cells，另给两个 kernels×两个 R 的 4 个谱模型/截断项；Bonferroni 每项 `alpha=0.0125/28`。用有限 `N_t` 的精确 `log(1+q+h+x)` 四点传播到 `Delta_R Delta_t`，最后才与 force-unit budget 比较。

当前 `lane_contract.py` 的 C `primary.kernel_id` 只能写一个 kernel，适配本次选定的 stout absolute Creutz；F `primary` 记录 integer stencil `R=2` 与 stout/identity IDs。所有数值门槛都是拟议 protocol defaults，q/h/x、GEVP gaps/overlaps、`N_t=32` 数据和独立 receipts 仍不存在，因此继续 `NOT_READY/UNCERTIFIED`、`GROUND_STATE_FORCE_UNCERTIFIED`、`physical_score=null`。D0–99 已观察，绝不用于本候选的 confirmation；封存 split 与 GPU/QUDA 仍关闭。

文字流程：`同一新配置/paired blocks -> 4 个 loop means -> C: finite-T chi_stout(2,5) 与 0.02 half-width gate`；`独立 calibration IDs + 3x3 GEVP + Nt16/Nt32 joint spectrum -> 分别给 q/h/x/model force-unit 上界 -> F: Delta_R U0(2) candidate-reference -> 5% systematic + 5% statistical gates -> 缺任一收据即 ABSTAIN`。

**交接规则：**下文所有要求用户在 identity 与 APE/stout、R/T/kernel 或误差预算间选择的段落均为历史记录，现由本节 defaults 覆盖；不要再次要求用户选择这些方法输入。合同和 `lane_contract.py` 诊断在[PR #6 head `6afedb178f325c15f62b4786c4ed2f075ebcedd8`](https://github.com/Greyyy-HJC/LatticeArena/tree/6afedb178f325c15f62b4786c4ed2f075ebcedd8/tasks/wilson_loop)，其中保留 C/F blocker 与 q/h/x→force 文字流程；独立谱/thermal receipts、paired split plan、审阅与资源授权仍为空。不读 D0–99 或封存确认数据、不运行 GPU/QUDA。

## 2026-11-25 当前 Wilson 接手点

C 主点固定为 stout absolute `chi(2,5)`；F 主点固定为 stout-minus-identity ground-state discrete force at `R=2.5`。较早关于改选 identity 或 APE/stout primary 的段落属于历史状态。lane contract 已写入 PR #6；测量值、paired split、独立 q/h/x/GEVP/Nt32 receipts 和 owner authorization 仍为空，因此继续标明 `0` 个新物理测量、保留 `NOT_READY/UNCERTIFIED`，不读封存数据或运行 GPU/QUDA。

## 2026-11-25 自选输入后的 C/F machine contract 与证据流程

- [PR #6 的机器合同](https://github.com/Greyyy-HJC/LatticeArena/blob/6afedb178f325c15f62b4786c4ed2f075ebcedd8/tasks/wilson_loop/lane_contract_v2.json)记录固定方法：C=`chi_stout(2,5)`、`N_t=16`、positive-ensemble-mean log、dimensionless halfwidth 0.02；F=`stout-identity` 基态力差、midpoint `R=2.5`、独立 `3x3 GEVP` 与匹配 `N_t=32` thermal route。合同同时明列 alpha 预算：C/F 各 .025；F 的统计/系统各 .0125，力差区间与正参考力下界各 .00625，28 个系统 bound cells 各 `.0125/28`。q/h/x 与 force-unit 门槛是验收条件，不是测量结果。
- C/F 文字数据流：`paired loops -> C positive means/logs -> finite-T chi and precision gate`；`independent spectra + matched Nt32 -> q/h/x/model simultaneous bounds -> log(1+q+h+x) at four stencil points -> Delta_R Delta_t force-unit propagation -> total systematic/statistical budget`。关键 receipt 缺失、reference-force lower bound 不为正、无效 draw 比例>1%或精度失败时按 lane abstain。
- Optimize/Wilson 任务专项回归 `tests/test_optimize* tests/test_wilson*` **97 passed**，其中 Wilson contract 14 项；同时包含 Wilson finite-Nt algebra/toy、随机正谱实现与 alpha 预算闭合检查。该结果只确认 CPU 软件/公式与合同结构、单位、alpha 记账和 fail-closed fixture；物理测量分母仍为 `0/0/0/0`，q/h/x 与谱证据仍未提供，ground-state force 仍 `GROUND_STATE_FORCE_UNCERTIFIED`。

**下一位 Wilson agent：**只在独立 calibration IDs 获批后生成 GEVP/多态 joint fit、匹配 `N_t=32` thermal 与 mixed-sector receipts；每个 q/h/x 及模型项附拟合图/残差或文字谱诊断和同时上界。随后独立审阅单位换算与四点 force propagation，再申请未观察 confirmation 输入与 GPU/QUDA 资源授权。D0–99 不能作为确认数据，不能用 toy/随机正谱范围填污染值；无 receipts 时两条 lane 均不升级。

## 用户已选择的两个独立 estimand

1. **C lane：有限时间 Creutz observable** `chi_k(R,T)`。在 protocol 冻结后，它可作为有限 `T` 的直接观测量报告；不能自动改名为基态精度或改善。
2. **F lane：基态离散静态力** `f0(R+1/2)=U0(R+1)-U0(R)`。只有激发态、有限 `N_t` 热/绕回和混合项的总系统误差有独立可审阅上界时才可认证。

目前仍为 `NOT_READY/UNCERTIFIED`、`GROUND_STATE_FORCE_UNCERTIFIED` 和 `physical_score=null`。不读取 gap、validation、shared holdout；不启动 GPU/QUDA。

## 已完成的解析与源核对

- [零温正谱敏感度报告](https://github.com/DianjunZhao/LatticeArena/blob/6f1aa442d4f8b1121899fb6515bcd5b51501b161/results/wilson_loop/ground_state_bound_sensitivity_20261123_v1/README.md)：实现 `|chi-f0| <= log(1+Q exp(-Delta_min*T))` 的条件界，并在 `Q=1e-3…1e2`、`Delta_min=0.01…1.0` 格点单位、`T=4…7` 上生成 14,884 行解析网格和四面板图。`Q` 和能隙全是插图输入，不是估计值；这个公式不包含有限 `N_t` 热迹。
- [有限 `N_t` 热/绕回理论审阅](https://github.com/DianjunZhao/LatticeArena/blob/6f1aa442d4f8b1121899fb6515bcd5b51501b161/results/wilson_loop/finite_nt_thermal_bound_review_20261122/README.md)：从 trace 展开静态对与无源两套 Hilbert sector，保留 pair excited `q`、thermal `h` 和 mixed `x`。如果各项有独立非负 envelope，则先构成 `B_k(R,t)=log(1+q_bound+h_bound+x_bound)`，再把四个 loop 点的 `B` 传播进 `Delta_R Delta_t` 的力/Creutz 系统界。当前没有任何物理 `Q,H,X,Delta` 输入，推导不等于认证。
- 原文编号已复核：有限温热迹和谱分解在 Bala et al. Appendix A 的 Eq. (35)–(36)；零温 Wilson-loop 正谱展开在 Brambilla et al. Eq. (3)。更复杂空间路径/链接 smear 会改谱权重，不改固定 action 下能级位置这一点来自有限温论文对静态介子算符的讨论。来源与适用边界见上述报告文末链接。
- [框架扩展点地图](../../doc/framework_extensibility.md)建议将 C/F 写成两个 versioned estimand contract：共享 loop/block IDs 和 paired draws，但分别设数值门和系统误差门。

![有限T Creutz与基态力的零温激发态敏感度](https://github.com/DianjunZhao/LatticeArena/blob/6f1aa442d4f8b1121899fb6515bcd5b51501b161/results/wilson_loop/ground_state_bound_sensitivity_20261123_v1/contamination_sensitivity.png)

文字流程图：`W_k(R,t;N_t)` → `q: pair excitations ~ Q exp(-Delta_Q t)` + `h: no-pair thermal ~ H exp(-Delta_0(N_t-t))` + `x: mixed sector` → `C lane 保留有限t的chi`；`F lane 经 log 后做 Delta_R Delta_t` → 加总统计、激发态、热/绕回、混合、finite-a/volume 误差 → 与事前容差比较 → 缺任一独立 bound 时保持 `GROUND_STATE_FORCE_UNCERTIFIED`。

### 有限 `N_t` q/h/x 四点式 toy 验证（CPU 代数检查已完成）

[Finite-`N_t` toy report](https://github.com/DianjunZhao/LatticeArena/blob/6f1aa442d4f8b1121899fb6515bcd5b51501b161/results/wilson_loop/finite_nt_toy_validation_20261123_v2/README.md) 用手设、无量纲正谱 toy 参数生成

```text
log W(R,t;N_t) = log A0(R) - V0(R)t + log1p(q+h+x)
chi - Delta_R V0 = -Delta_R Delta_t delta
|chi - Delta_R V0| <= sum_of_four_points log1p(q+h+x)
```

5 个 `t=4...8` stencil 的直接 bias 与 `-Delta_R Delta_t delta` 恒等式最大差为 `7.63e-16`；四点保守 envelope 覆盖全部 5 个 toy bias。`q` 随 `t` 下降，而 complementary-time `h` 在图示参数下上升，展示两类污染的竞争。测试另将 `H=X=0` 的极限与已有零温正谱 `creutz_excited_correction` 逐点交叉核对。

![有限 N_t 的 q/h/x 污染和四点误差界](https://github.com/DianjunZhao/LatticeArena/blob/6f1aa442d4f8b1121899fb6515bcd5b51501b161/results/wilson_loop/finite_nt_toy_validation_20261123_v2/finite_nt_toy_validation.png)

这只验证符号、索引和误差传播。输入不是 measurement、fit 或独立物理上界；保守 envelope 也未证明有用。C lane 仍是有限-T Creutz observable；F lane 仍需要真实、独立的激发态/热/绕回/mixed-sector 界和事前容差，状态保持 `GROUND_STATE_FORCE_UNCERTIFIED`。

### 正谱参数范围的随机代数压力测试（已完成）

[Randomized finite-`N_t` stress report](https://github.com/DianjunZhao/LatticeArena/blob/6f1aa442d4f8b1121899fb6515bcd5b51501b161/results/wilson_loop/finite_nt_randomized_stress_20261123_v1/README.md) 对 4,000 个独立 seed case 重复计算同一四点恒等式和正污染 envelope。输入范围是 `N_t=4…64`、`t` 均匀取 `1…N_t-1`、`A0` 对数均匀 `[0.1,10]`、`V0` 均匀 `[0,1.5]`、各半径独立 `Q/H/X` 对数均匀 `[1e-3,1e2]`、各 gap 对数均匀 `[0.01,1]`。这些是压力测试区间，不是格点谱的先验或估计。

![随机正谱参数下的 Creutz 偏差与四点 envelope](https://github.com/DianjunZhao/LatticeArena/blob/6f1aa442d4f8b1121899fb6515bcd5b51501b161/results/wilson_loop/finite_nt_randomized_stress_20261123_v1/finite_nt_randomized_stress.png)

4,000/4,000 个情景的 `-Delta_R Delta_t delta` identity 与 envelope 检查通过；最大浮点 identity 差为 `3.20e-14`。`|bias|/envelope` 中位数 `0.0118`、95th percentile `0.117`、最大 `0.398`，显示这个解析 envelope 在抽样域里通常偏松。分母为 randomized cases `4000/4000/0/0`、四点 loop evaluations `16000/16000/0/0`（planned/valid/failed/abstained）。报告保存所有参数、逐情景诊断、源码快照和哈希。

该结果只加强 toy 代码在宽正参数域内的代数验证，并显示保守界的松弛程度会随任意参数域变化。它不证明真实 `Q/H/X/gap` 落在该范围，也不提供有限样本物理覆盖概率或 F-lane 误差预算。

### C/F lane contract inspector（结构诊断，companion PR #6）

[`lane_contract.py`](https://github.com/DianjunZhao/LatticeArena/blob/75537cbbbef0e1234dbb53a71c81b3a0476ef2f4/tasks/wilson_loop/lane_contract.py) 为 C/F 两条用户选定 estimand 提供独立 schema 检查；[v3 诊断报告](https://github.com/DianjunZhao/LatticeArena/blob/75537cbbbef0e1234dbb53a71c81b3a0476ef2f4/results/wilson_loop/lane_contract_diagnostic_20261124_v3/README.md) 保存当前 null draft、C-only/C+F 完整 fixture 和四类刻意损坏的 synthetic fixture，附文本决策流、逐 fixture/lanes 分母、源码快照及 5/5 artifact hashes。共享字段检查包括 action/geometry/边界/单位、kernel 与 split/paired-block receipts；C lane 检查 `R/T/kernel`、几何范围、正 ensemble-mean log 规则、失败处理、比较族、precision 与 alpha/multiplicity；F lane 检查候选/参考力差分、单位、dimensionless q/h/x receipts、已传播到 force units 的 excited-state bound、thermal/mixed/finite-a/volume/other force-level bounds、独立 receipt、alpha 分配与总误差预算。q/h/x 原始谱比值不能直接与力的 tolerance 相加。

v2 将 `finite_nt_toy.py` 的有符号有限-T Creutz 约定固定为 `chi = -[log W(R+1,T+1)+log W(R,T)-log W(R+1,T)-log W(R,T+1)]`，并要求 F stencil 表示 candidate force minus reference force。4 类负向 fixture 验证：整体反转 C 符号、整体反转 F 符号、把 F statistical precision 误标为 dimensionless、以及将公共 `kernel_ids` 置空时都 fail closed；最后一种返回 blockers 而不抛异常。七个 fixture 对两条 lane 的分母合计 planned/valid/failed/abstained=`14/6/8/0`；无效 fixture 的 `failed` 是预期诊断结果。v3 修正 README 结果表遗漏 `abstained` 列的问题，并新增断言核对表格列和总分母：planned/valid/failed/abstained=`14/6/8/0`，逐行满足 planned=valid+failed+abstained。Wilson 专项测试 `10 passed`，v3 报告哈希 `5/5`，`git diff --check` 通过；测试见 [`test_wilson_lane_contract.py`](https://github.com/DianjunZhao/LatticeArena/blob/75537cbbbef0e1234dbb53a71c81b3a0476ef2f4/tests/test_wilson_lane_contract.py)。

仍未冻结任何真实 primary `R/T/kernel`、单位、污染容差或 multiplicity；真实 q/h/x、finite-`N_t`、mixed-sector、格距和体积上界也尚不存在。v2 只验证合约程序能拒绝错误输入，不提供物理不确定度或认证。

## 解析公式与注意事项

零温正谱下写

```text
W_k(R,T)=A0_k(R) exp[-E0(R)T] [1+rho_k(R,T)]
rho_k(R,T) <= Q_k(R) exp[-Delta_min(R) T]
g_k(R,T)=log[(1+rho_k(R,T))/(1+rho_k(R,T+1))]
chi_k(R,T)=f0(R+1/2)+g_k(R+1,T)-g_k(R,T)
```

当 overlap 系数非负且能隙为正，`0<=g<=log(1+rho)`，可得单 kernel 的条件界。candidate-minus-identity 需要两侧污染界并相加。该界条件是正向、零温的谱和；有限时间窗拟合稳定或 bootstrap 区间很窄都不证明此条件。

有限 `N_t` 下，热迹可写成相对 ground-ground 项的

```text
W ~ exp[-V0(R)t] * [1 + q(R,t) + h(R,N_t-t) + x(R,t,N_t-t)]
```

增大 `t` 会压低 `q`，但缩短 `N_t-t`，可能放大 `h`。若每个污染项有可证的 envelope，定义其和的 log 上界 `B_k(R,t)`；保守的单 kernel 力/Creutz bias 界为参与的四个 `B` 之和。只有 `Q/gap` 的数值不能覆盖热迹和 mixed term，partition function 在差分中约去也不会抹掉 numerator 热项。

## 当前下一位 Wilson agent 开工顺序（2026-11-25 defaults）

1. **C lane：**不再选择主点。新授权的未观察配置上测 `chi_stout(2,5)`，`N_t=16`；每 draw 先算四个 paired loop mean，仅对正均值取 log。总 familywise `alpha=0.05` 按 C/F 两条 co-primary lane 各 `.025` 分配；C 的双侧 interval half-width 必须 `<=0.02`。使用10,000 paired circular-block draws、length-5主分析及length-2/10敏感性 hull；invalid draw 超过1%或精度未达标则 abstain。结果始终称 finite-T Creutz。
2. **F lane 的独立 calibration：**与最终 confirmation 不重叠的 IDs 上，按 `R=2,3` 的 identity/APE/stout 3x3 GEVP、`t0=1, td=2`、二/三态 `N_t` 联合拟合和匹配 `N_t=32` 热路径估计 q/h/x/model；交付能隙/overlap、残差/窗口稳定性图、同时界、失败数和 raw receipts。D0–99 已用于 development，不能作为这项 calibration 的确认样本。
3. **F lane 的误差门：**在 force units `a_s*a_t*F` 中检查 q<=1.5%、h<=1.5%、x<=1%、谱截断<=1%，线性相加系统界<=5%，统计 half-width<=5%，总线性误差<=10%；reference force 的同时正下界必须大于0。将 `log(1+q+h+x)` 在四个 stencil 点逐点传播后再做 `Delta_R Delta_t`。F lane 的 alpha=.025 中统计/系统各 `.0125`，统计再给 force-difference interval 与 reference-force 下界各 `.00625`；28个系统界 cells 各用 `.0125/28`。缺一项独立 receipt 或超预算即 `GROUND_STATE_FORCE_UNCERTIFIED`。
4. **confirmation与授权：**C/F 的 block计划、split hashes、候选身份和 independent physics review 完成后，仍须数据 owner 授权新未观察数据；GPU/QUDA另须资源 owner 授权。任何一项未授权时不读 sealed data、不提交作业、不计算 physical score。
5. **诊断交付：**两条 lane 分开列 planned/valid/failed/abstained、原因、paired interval、统计与系统误差；提供 C 的四点 loop/区间图和 F 的 q/h/x/model force-budget 图或完整文字数据流。C 的有限-T精度不能替代 F 的谱/热污染认证。

## 历史开工清单（已由 2026-11-25 自选输入更新替代）

1. **C lane contract：**冻结 primary `R/T/kernel`、有限 `N_t`/边界与空间单位、loop mean 的正值/log 规则、坏点处理、候选族、多重性、配对 block 方案与精度目标。它的结论名称必须保留“finite-T Creutz”。开发数据不得反复用于选择确认点。
2. **F lane contract：**明确 `U0`、`f0` 的 lattice/physical units 和静态源约定；冻结 force finite-difference stencil、目标 `R`/`T`、谱模型/截断、置信度和总系统预算。给 `Q_k(R), Delta_Q(R), H_k(R), Delta_0, X_k(R)` 的独立来源，或提交有完整失败诊断的多态/变分谱方案。
3. **q/h/x 代数验证已完成**（见上方 companion PR #6：固定 toy、零温极限交叉测试，以及跨 4,000 个宽正参数 case 的恒等式/envelope 压力图、逐例原始表和哈希收据）。不要把 toy 或随机范围推广为真实系统误差界。
4. 比较可审阅的证据路线：独立 static-source operator basis 的谱/overlap 稳定性；或 matched `N_t`（相同 action、`a_t`、空间体积、`R`、kernel）的热污染差异。任一路径均须冻结 rank/截断、重采样和失败/abstain 规则；短单 `N_t` 数据不可识别每个 sector 的上界。
5. 只有方案通过审阅并获数据/GPU 授权后才测量。报告 C/F 两条 lane 各自的 planned/valid/failed/abstained、paired intervals、统计误差、激发态界、热界、mixed 界、格距/体积误差及其图示。禁止把 C lane 数值重命名为 F lane。

## 停止规则

若 `R/T/kernel`、单位、失败和多重性规则未冻结，不做确认性比较；若 F lane 的任一独立谱/热/混合界无法以事前容差认证，则保留 `GROUND_STATE_FORCE_UNCERTIFIED`。有限-T 精度、窗口稳定、单一能隙假设或 narrow bootstrap 都不能代替缺失系统界。

## 2026-11-24 本轮复核记录

- 在 companion PR #6 当前代码提交 [75537cbb](https://github.com/Greyyy-HJC/LatticeArena/tree/75537cbbbef0e1234dbb53a71c81b3a0476ef2f4) 上，对 Optimize/Wilson 共 8 个 CPU 测试模块重跑：**61 passed**；其中 Wilson 覆盖有限 Nt toy、随机正谱恒等式、bound 和 lane contract。
- PR #6 中 Wilson 的 6 份已跟踪报告 manifest、**32 个**报告文件的 SHA-256 全部匹配；当前 lane-contract 报告 v3 的表格包含 planned/valid/failed/abstained，逐行与汇总分母恒等式成立。
- 没有读取 sealed data 或运行 GPU/QUDA；有限-T Creutz 和基态离散静态力仍为不同 estimand，NOT_READY/UNCERTIFIED、GROUND_STATE_FORCE_UNCERTIFIED 与 physical_score=null 不变。
- 下一步只有在用户冻结 C lane 的 primary R/T/kernel、精度/多重性规则，以及 F lane 的独立激发态/热/混合污染界与误差预算后，才可进入确认测量；不得由 toy 参数或单一 Nt 数据推定这些物理界。

## 2026-11-24 后续复核同步

- companion PR #6 新增 Optimize degree-4 Chebyshev 系数在 `[-1,1]` 连续区间上的正性诊断，报告包含图、逐 panel 区间最小值、分母和哈希。它只覆盖三个 synthetic fit；对 Wilson 的 estimand 或物理结论没有影响。
- PR #6 源码/测试提交 `f25acababd490729328af1189a5934f56bde7f89` 上，9 个 Optimize/Wilson 专项 CPU 测试模块共 **65 passed**；这不是完整仓库测试或生产验证。
- C lane 与 F lane 仍分别是有限-T Creutz observable 和基态离散静态力；没有封存数据或 GPU/QUDA 测量。C lane 仍缺 primary `R/T/kernel`、精度及多重性规则；F lane 仍缺独立污染上界与总误差预算，状态保持 `NOT_READY/UNCERTIFIED`、`GROUND_STATE_FORCE_UNCERTIFIED` 和 `physical_score=null`。

## 2026-11-24 Pion contract structure check follow-up

- PR #5 增加 [Pion contract inspector](../pion_projection/bounded_block_risk_contract.py) 和 [诊断报告](../../results/pion_projection/bounded_block_risk_contract_diagnostic_20261124_v5/README.md)，覆盖 null draft、两条依赖理论路线、anchor 的 `B/C` 共偏差结构与覆盖假设、alpha 分配及独立 reference gate。报告分母 `6/2/4/0` 统计合同检查，不是 Wilson measurement。
- Pion 的 8 个 bounded-risk/dependence CPU 专项模块共 67 passed；v5 还校验总 anchor 界覆盖随机半径加 common-bias `C`；该工作没有访问 Wilson 数据或改变 C/F lane 的任何物理状态。
- PR #5 当前共 9 个 tracked Pion report manifest / 74 个 payload 文件，PR #6 的 Optimize/Wilson 报告分别为 4/38 与 6/32，SHA-256 均已复核。C/F lane 仍缺各自 TODO 列明的研究者输入与物理证据；三份 task TODO 的 34 个相对链接与 28 个 GitHub commit/tree 链接本轮也全部解析通过。

## 2026-11-24 Pion Markov-score follow-up

- PR #5 的 Pion Markov pseudo-gap 路线现有条件式 block-risk LCB helper 与 [v3 图/文字报告](../../results/pion_projection/bounded_block_risk_markov_gap_20261124_v3/README.md)，按 corrected arXiv v5 theorem version 记载 stationary/nonstationary-TV 条件路径，含部分失败、全失败与 post-burn-in TV synthetic panel，并分别报告 planned block 和 comparison denominator。它没有访问 Wilson 输入、改变 C/F estimand，或提供 Wilson 污染界。
- 此历史快照的 Pion bounded-risk/dependence 专项 suite 为 **78 passed**；后续 theorem-contract closure 的当前结果为 **93 passed**，manifest/hash 与本地 TODO 链接复核见 Pion TODO 最新记录。Wilson 的 C lane primary `R/T/kernel` 与精度/多重性规则、F lane 独立污染界和总误差预算仍未冻结，状态保持 `NOT_READY/UNCERTIFIED` 与 `GROUND_STATE_FORCE_UNCERTIFIED`。

## 2026-11-24 Pion theorem-contract closure

- PR #5 的 [Markov 报告](../../results/pion_projection/bounded_block_risk_markov_gap_20261124_v3/README.md) 与 [contract v6 报告](../../results/pion_projection/bounded_block_risk_contract_diagnostic_20261124_v6/README.md)现明确要求 fixed score function、stationary 起点或带 receipt 的 post-burn-in TV 界，并检查 `F*delta < alpha_score`。这只是 Pion synthetic contract 诊断，不是 Wilson measurement，也不提供 Wilson 的 q/h/x 污染界。
- 当时的 Pion Markov helper 仅支持固定 score；Eq. (3.26) 的变时路线已在 PR #5 最新更新中独立实现并加结构合同，但这不改变 Wilson 的两条独立 estimand，也不提供 Wilson 的物理污染界。
- 下一位 Wilson agent 仍按本文件原顺序推进：C lane 等待 primary `R/T/kernel`、精度和多重性规则；F lane 等待独立谱/热/绕回/mixed 污染界及总误差预算。不得从 Pion 的 TV 校准、Wilson toy 或单一 `N_t` 推导这些量；收到相应研究者输入后再冻结 protocol 与执行授权。

## 2026-11-24 companion PR #6 head 337bc99 时的 handoff（历史快照）

- PR #6 当前报告头为 [`337bc99b`](https://github.com/Greyyy-HJC/LatticeArena/tree/337bc99b87952c4c866c6f55ea000427c00273b5)；lane contract v3 的文本决策流程和完整分母表已经包含两个独立 estimand、两种 stencil 符号 fail-closed 例、单位错配与空 kernel-ID 例。该结构诊断只检查字段/收据格式，未读取 Wilson measurements。
- C lane 仍要由研究者冻结 primary `R/T/kernel`、有限 `N_t`/边界和单位、正 ensemble-mean/log 规则、坏点、候选族、多重性、paired-block 方案与统计精度；最终名称仍为有限-T Creutz。
- F lane 仍需 independent `Q_k, Delta_Q, H_k, Delta_0, X_k` 或可审阅的多态/变分方案，并把激发态、热/绕回、mixed、finite-a、volume 等传播到 force units 后列入总误差预算；不能从 q/h/x 无量纲量直接与力容差相加。ground-state force 继续 `GROUND_STATE_FORCE_UNCERTIFIED`。
- Pion Eq. (3.26) 变时集中界只改善 Pion 的可选统计合同，与 Wilson 的 C/F 物理输入无关。没有新的 R/T/kernel、谱界、总误差预算和数据/GPU 授权时，不生成 C/F 物理分数，保持 `NOT_READY/UNCERTIFIED`、sealed data 关闭与 `physical_score=null`。
- 本轮在 PR #6 head `337bc99b87952c4c866c6f55ea000427c00273b5` 上连同 Optimize 一起重跑 9 个 CPU 专项模块：**65 passed**；Wilson 代码验证仍只覆盖 toy 恒等式、正谱范围和合同 fail-closed 行为。

## 2026-11-24 PR #5 head 71589ab 时的跨任务快照（已被后续记录取代）

- PR #5 当前 head 为 `71589ab`，Pion Eq. (3.26) time-varying Markov helper/合同分支与 v7 结构报告已经提交；Pion 当前专项 suite 为 **116 passed**，已跟踪报告为 **14 manifests / 116 payload files**，哈希均匹配。这些 Pion 更新不提供 Wilson 的谱/热/mixed 污染界。
- Wilson 的有限-T Creutz C lane 和基态离散静态力 F lane 继续是独立 estimand。当前任务没有未完成的 CPU-only 代数/结构诊断；下一步须先获得 C lane 的 primary `R/T/kernel` 和推断规则，以及 F lane 的独立污染界和力单位总误差预算，再决定确认测量授权。此前冻结的 `NOT_READY/UNCERTIFIED` 与 `GROUND_STATE_FORCE_UNCERTIFIED` 保持不变。


## 2026-11-24 Astra 输入审阅：C/F 两条 lane 的候选合同

GPT-6.2-Astra 子代理完成只读协议审阅后已关闭；它不能授予数据或集群权限。遵照用户此前决定，不读 sealed split、不启动 production GPU；两条 lane 仍为独立 estimand，状态 NOT_READY/UNCERTIFIED、GROUND_STATE_FORCE_UNCERTIFIED、physical_score=null。

### C lane：单一 finite-T Creutz 候选

本机历史审阅分支 codex/continue-research-todos-20261123、提交 f9603b4 中的 protocol_d_20261120.json（SHA-256 bf8048ff570df1eed251c4518ea1d4e4a224ee985d4aa4dd0ae0f342de65ced0；该快照不在 PR #5/#6 当前树内）固定 S16T16D、beta=6、16^4 全周期边界，development stencil 覆盖 R=1…4、T=4…8，并列有 5-step APE alpha=0.4 与 stout rho=0.1 这组弱场线性化匹配候选（rho=alpha/4 只匹配 leading weak-field update，不认证非微扰 smearing radius）。可供研究者审批的新确认 primary 候选为 R=2、T=5，对比 stout_rho0.1_n5 与 ape_a0.4_n5 的有限-T observable difference。R/T 位于已冻结网格内，kernel pair 也是 protocol 的弱场匹配对；但这个具体 primary 现在才提出，D0–99 已被观察，故绝不能用该 split 作确认，只能在单独授权且尚未查看的独立输入上评估。C 值按 chi_k(2,5)=−[log Wbar_k(3,6)+log Wbar_k(2,5)−log Wbar_k(3,5)−log Wbar_k(2,6)] 计算，差值单位为 dimensionless lattice observable，不称为 ground-state force 或风险改善。因为 D0–99 已用于 development，此候选只能在另行授权且未查看的确认输入上检验；不把旧 development 区间重新命名为确认结果。

另一个可审议选项是单一 `identity` kernel 的绝对 `chi_identity(2,5)`，历史 kernel 列表支持这个候选；它与上面的 APE/stout 配对差值不是同一 estimand。现有 `primary.kernel_id` 字段对应单 kernel 主点；若研究者想要 kernel-pair difference，需明写 contrast、方向、共同 block IDs 和多重性族，不能把两个候选悄悄当成同一个 C 主点。两项都未冻结。

**推断规则候选：**对两个 kernel 和四个必需 loop 使用共同的 configuration/block IDs；每个 bootstrap draw 先重算每个 loop 的 ensemble mean，再计算 log ratio 与 paired delta-chi，以保留相关性。任何必需 mean 非正/非有限时保留该 draw 的失败原因和 planned denominator，不删 draw。先只设这一组 R/T 和一个 kernel-pair contrast；若扩展到多个 R/T/kernel，使用预先冻结的 max-statistic simultaneous interval 或 Bonferroni。若 C 与 F 同为 co-primary，建议总单侧错误预算 alpha_total=0.05 并以 Bonferroni 各分 alpha_C=alpha_F=0.025；这只是清晰保守的候选规则，block independence、bootstrap coverage 和实际 alpha 仍需审阅/批准。protocol 中 block_length=5、5000 bootstrap 只能作为复核起点，不构成独立性或覆盖证明。

### F lane：基态静态离散力与污染预算

若两条 lane 用同一空间 midpoint，可提议 R_mid=2.5，即 f0(2.5)=U0(3)−U0(2)=a_t[V0(3)−V0(2)]，保持 fixed-a lattice quantity 的 dimensionless 约定；不要自动称为物理单位的连续静态力。提取需冻结谱模型/GEVP operator basis、时间窗、rank/truncation、mode tracking、协方差与失败/弃权规则。可选独立证据是扩展静态源 operator basis 的多态 GEVP/多时间拟合，或相同 action、a_t、体积、R 与 kernel 的 matched-N_t 比较；仅有窗口稳定、单一 Nt、bootstrap 很窄或正谱 toy 都不构成污染上界。

研究者需给出独立收据支持的 Q_k(R)、Delta_Q(R)、H_k(R)、Delta_0、X_k(R) envelope，或通过审阅的谱/变分方案，并把每项传播到 force units 后纳入总预算。建议预算表至少分列统计项、激发态项、热/绕回项、mixed-sector 项、谱截断/拟合项；只有声称 continuum/physical force 时才另加入格距/体积换算误差。保守接受条件可写作所有 force-unit 项的总上界不超过研究者冻结的 epsilon_F，且独立 receipts 和整体 alpha 分配闭合。q/h/x 原始无量纲振幅不得直接与 force tolerance 相加。

文字数据流：同一新配置与 paired blocks → loop means/logs → C lane finite-T delta-chi；同一 blocks → 多态谱/GEVP → U0(2),U0(3) → F lane f0(2.5) → 激发态+热+mixed+截断误差转为 force units → epsilon_F gate。任一步缺测/不辨识即 ABSTAIN；没有新增测量时分母保持 0/0/0/0。

**下一位 Wilson agent：**先交这份 C/F 候选卡供研究者冻结；随后只做协议/收据的 CPU preflight，不读 D100–119、D120–139、D140–199，不用已观察 D0–99挑 primary，也不从 toy 给污染项填数。任何确认测量还要由数据/集群 owner 单独授权。

## 2026-11-24 shared abstention-state follow-up

PR #6 [`710eeb2`](https://github.com/Greyyy-HJC/LatticeArena/commit/710eeb27b5d94de37a646ee52ec743be18894786) adds a terminal `abstained(reason)` state to generic `RunState`, distinct from `failed`. Missing C-lane protocol fields or F-lane contamination receipts can be represented as no-certificate states without calling them demonstrated physics failures. Wilson's lane inspector/evaluator is not yet wired to this generic state, so both lane statuses and all data-access restrictions remain unchanged. The repository-wide `python3 -m pytest -q` run passed **134 tests** on PR #6 head [`710eeb2`](https://github.com/Greyyy-HJC/LatticeArena/commit/710eeb27b5d94de37a646ee52ec743be18894786); this is CPU/software validation, not GPU/MPI physics validation.

文字流程：`missing C/F freeze or independent bound -> abstained(reason) terminal -> no score`; after independent review and authorization, a fresh run can proceed through the ordinary state transitions. This state model supplies no `Q_k`, thermal, mixed-sector or force-unit error bound.

## 2026-11-24 shared TaskSpec split preflight

PR #6 head [`3b9c772`](https://github.com/Greyyy-HJC/LatticeArena/commit/3b9c77201be97d300dfce98b0c3e0d67dc57146f) makes generic `task_spec` require named inclusive integer split ranges and reject overlap or reversed endpoints. Wilson's lane contract is not yet represented as a `TaskSpec`, so this generic check does not freeze its C/F IDs, block pairing, or confirmation design. The full repository test suite passed **134 tests** on that head.

文字流程：`future reviewed lane IDs -> declared split ranges -> generic format/disjointness check -> lane-specific receipt and paired-block checks`; the first check does not prove independence or contamination bounds. Wilson remains `NOT_READY/UNCERTIFIED` and `GROUND_STATE_FORCE_UNCERTIFIED`.

## 2026-11-24 shared ScenarioSpec/RunReceipt follow-up

PR #6 head [`f209015`](https://github.com/Greyyy-HJC/LatticeArena/commit/f209015337b08f26e34b733771d56142a5288806) 将 [`core/receipts.py`](https://github.com/Greyyy-HJC/LatticeArena/blob/f209015337b08f26e34b733771d56142a5288806/core/receipts.py) 接入 Wilson [lane contract](https://github.com/Greyyy-HJC/LatticeArena/blob/f209015337b08f26e34b733771d56142a5288806/tasks/wilson_loop/lane_contract.py)。[v4 报告](https://github.com/Greyyy-HJC/LatticeArena/blob/f209015337b08f26e34b733771d56142a5288806/results/wilson_loop/lane_contract_diagnostic_20261124_v4/README.md) 把 7 个合成 fixture × 2 条 lane 记录为 14 个结构检查：planned/valid/failed/abstained=`14/6/8/0`，8 个失败单元各有 blocker 原因。文字决策流、合同 JSON、CSV 分母、protocol 和两个源码快照共 6 个 receipt payload；`run_receipt.json` 绑定这 6 项，外层 manifest 覆盖 7 个文件（不含 manifest 自身），报告共 8 个文件且哈希均逐项匹配。`complete` 只代表 14 个合同检查均被记账，不能解释为物理验证、测量成功或独立样本。

**下一位 Wilson agent：**不要重复运行相同的 synthetic schema fixture，也不要将 toy 或 Pion 的 Markov/TV 结果当作 Wilson 污染界。先请研究者在单一 `identity` kernel 的绝对 `chi(2,5)` 与 APE/stout 配对差值这两个不同 C estimand 中选择，并冻结主点、有限-T loop/log 失败规则、paired-block 与 interval/multiplicity/precision 方案；单独冻结 F lane 的基态力模型/contrast、谱/热/绕回/mixed 污染界及力单位误差预算。现有候选（C 的 `R=2,T=5`；F 的 midpoint `R=2.5`）只是需研究者批准的候选，并非已冻结值；D0–99 已观察过，不能用来确认。冻结后先做不读数据的 protocol/receipt CPU preflight，输出图或文字决策流和按 C/F lane 区分的分母/原因；只有数据 owner 另行授权后才触碰未观察 confirmation 输入。保持 `NOT_READY/UNCERTIFIED`、`GROUND_STATE_FORCE_UNCERTIFIED`、sealed data closed 与 `physical_score=null`。

### Legacy Wilson v3 source-fingerprint audit

复核 PR #6 的旧 `lane_contract_diagnostic_20261123_v3` manifest 时，README、`contract_diagnostics.json`、`lane_summary.csv` 和 `protocol.json` 四个报告 payload 的哈希匹配。第五项 `lane_contract.py` 是源代码 fingerprint，但 v3 目录未保存该源文件快照；当前 PR #6 源文件哈希为 `0a52b23e…d92c3`，manifest 中为 `2894d19d…acf7c`，在本地可见的该文件 Git 历史版本中也未找到匹配字节。因此只能复核 v3 的四个报告 payload，不能从当前 checkout 复现其源代码版本。将 v3 保留为历史诊断，不以它支持当前合同；PR #6 的 [v4 报告](https://github.com/Greyyy-HJC/LatticeArena/blob/f209015337b08f26e34b733771d56142a5288806/results/wilson_loop/lane_contract_diagnostic_20261124_v4/README.md) 已保存源码快照并通过本轮 manifest 核验。后续归档继续把 source snapshot 放进报告目录，再生成 manifest。

## 2026-11-24 GPT-6.2-Astra 审阅与 C 单位 fail-closed 修复

Astra 的只读审阅没有替数据/资源 owner 授权，也未给 C primary 或 F 污染界赋值；C/F 仍分别保持有限-T Creutz 与基态离散静态力 estimand，状态 `NOT_READY/UNCERTIFIED`、`GROUND_STATE_FORCE_UNCERTIFIED`，`physical_score=null`。

PR #6 提交 [`f9be949`](https://github.com/Greyyy-HJC/LatticeArena/commit/f9be949e83fdc6f22e98d2eb1c7ffe068fc9f9ab) 现在要求 C lane 的统计精度目标使用 `dimensionless`，因为其估计量是 Wilson-loop ensemble means 的有符号四点 log 组合。新增错误单位 fixture 把能量单位送入 C precision 字段并被拒绝；F lane 原有 force-unit 检查保持独立。 [v5 文本诊断](https://github.com/Greyyy-HJC/LatticeArena/blob/f9be949e83fdc6f22e98d2eb1c7ffe068fc9f9ab/results/wilson_loop/lane_contract_diagnostic_20261124_v5/README.md) 分母为 8 fixtures × 2 lanes：planned/valid/failed/abstained=`16/7/9/0`，每个失败 lane 都记录 blocker；6 个 receipt payload 和 7 项 manifest 哈希逐一匹配。全仓 CPU suite 在 PR #6 head `f9be949` 为 **140 passed**。这是合同/软件检查，不是 Wilson 测量或污染认证。

文字流程：`null C/F draft -> C stencil/geometry/unit check (C precision must be dimensionless) -> F force-unit/q-h-x/budget check -> missing field gives NOT_READY -> complete synthetic shape gives human-review-required only -> authorized measurement and independent physics review remain separate -> physical_score=null`。

**下一位 Wilson agent：**不要重复这 16 个合同 fixture，也不要从 Pion 的 gap-bound 或 Wilson toy 推导 F 污染界。请先让研究者在单一 identity 绝对 `chi(2,5)` 与 matched APE/stout `delta-chi(2,5)` 两个不同 C estimand 中选择，并冻结 `R/T/kernel`、loop mean/log 失败规则、paired-block、precision/置信与 multiplicity；随后单独冻结 F 的谱提取/时间窗、独立激发态和热/绕回/mixed 界及同单位误差预算。历史候选 `R=2,T=5` 和 midpoint `R=2.5` 均仍是待审值，D0–99 已观察过，不能用来确认。任何未观察确认数据或 GPU/QUDA 测量均须相应 owner 授权。提交前可做不读数据的协议/收据 preflight，并用表/图或文字数据流列出 C、F 各自 planned/valid/failed/abstained 与 blocker；缺输入时保持当前状态。

## 2026-11-24 继续工作后的授权边界复核

用户对研究输入/外部授权问题的答复是：维持当前 `NOT_READY/UNCERTIFIED`、`GROUND_STATE_FORCE_UNCERTIFIED` 与封存状态。Astra 给出的 C/F 候选是审阅材料，不是用户选定的 C primary、F 污染上界或测量授权；因此本轮不再重复同一批 synthetic contract/toy 诊断。Wilson TODO **保留**，因为两个 lane 都仍有未闭合的科学输入。

下一位 agent 在得到新的研究者决定后再推进：C lane 须先明确选择单 kernel 绝对 `chi(2,5)` 或 matched APE/stout `delta-chi(2,5)`，并冻结 `N_t`/边界、loop mean/log 失败策略、paired block、precision、置信与 multiplicity；F lane 须给出独立的谱/GEVP 或 matched-`N_t` 证据，逐项约束激发态、热/绕回、mixed-sector、谱截断和总 force-unit 误差预算。输出保留 C/F 分开的 planned/valid/failed/abstained 表、污染/误差图或文字数据流。D0–99 不得转为确认样本；任何未观察 confirmation 输入与 GPU/QUDA 仍需相应 owner 单独授权。

## 2026-11-25 最新接手说明

上面 2026-11-24 的“等待用户选择 C estimand、primary 与预算”已由用户要求助手自行选择输入的指示取代。选定的两条 lane、机器合同、q/h/x→force 文字流程和 97 项 Optimize/Wilson CPU 回归结果见前面的 2026-11-25 更新；合同已在[PR #6 head `6afedb178f325c15f62b4786c4ed2f075ebcedd8`](https://github.com/Greyyy-HJC/LatticeArena/blob/6afedb178f325c15f62b4786c4ed2f075ebcedd8/tasks/wilson_loop/lane_contract_v2.json)。当前仍没有新物理测量或独立 GEVP/q/h/x/Nt32 receipts，confirmation 和 GPU/QUDA 授权仍关闭；下一步仅在独立 calibration IDs 与对应 owner 授权到位后开展污染测量与 review。

## 2026-11-25 接手复核：C/F 方法选择完成，数据门仍关闭

此前要求在 C 候选间再选择 estimand、或重新选择 primary 精度规则的段落均为历史记录；本文件开头的 C/F 默认值已经按用户指示自选。合同、有限 `N_t` toy、C/F 单位/符号检查和可读的误差决策流程已覆盖 CPU 侧工作；不再重复相同 contract/toy fixture。

接下来真正需要的证据分开记录：

- **C lane:** 在新的授权 calibration IDs 上测 `chi_stout(2,5)`；保存四个 loop 的正 ensemble means、paired-block bootstrap draws、坏 draw/失败原因及精度和多重性收据。D0–99 已观察过，不能改作 confirmation。
- **F lane:** 在独立 calibration IDs 上提供 `R=2,3` 的 3×3 GEVP、两/三态和时间窗敏感度、匹配 `N_t=32` 热污染、`q/h/x` 分别传播到 force units 的界与总预算；由独立审阅确认各界未重复计数且覆盖概率/误差目标闭合。
- **后续 confirmation:** 只有 calibration gate 通过、独立 review 完成且 data/resource owner 明确授权新的未观察 IDs 与 GPU/QUDA 运行后，才可打开 confirmation。

目前这些输入、收据和授权都没有出现在可审阅范围。继续保持 `NOT_READY/UNCERTIFIED`、`GROUND_STATE_FORCE_UNCERTIFIED`、`physical_score=null` 与 sealed data closed；不使用 toy 或历史样本补造污染界。

本次在 PR #6 当前 head `6afedb1` 重跑 Wilson/Optimize 专项 suite 为 **97 passed**，全仓 CPU suite 为 **145 passed**。这些测试验证合同、公式与代码路径，不提供 C loop 测量、F lane 的污染上界或 confirmation 证据。

## 2026-11-25 后续接手：Optimize 合同更新后的 Wilson 状态

PR #6 最新 head [`49673bb`](https://github.com/Greyyy-HJC/LatticeArena/commit/49673bbec8044ecd313a4237b4a7d3f3055e75f9) 本轮更新仅涉及 Optimize 的 D11 Bernstein/Chebyshev 转换和图文报告生成器；Wilson source、lane contract 与其物理默认值没有改动。当前 head 上重跑 `tests/test_wilson*.py`：**37 passed**；全仓 CPU suite：**146 passed**。这是共享代码回归与 Wilson 合同/toy 软件验证，不是新的 C/F 物理测量或独立污染证据。

Wilson 仍按已选定的有限温 Creutz C lane 与基态离散静态力 F lane 分开报告。继续缺少 C lane 新 calibration IDs 的四 loop 正均值/paired-block 精度收据，以及 F lane 独立 3×3 GEVP、多态/时间窗、matched `N_t=32` 热路径、q/h/x→force 同时界和独立审阅；D0–99 不可改作 confirmation。没有这些资料和 data/resource owner 授权时，保持 `NOT_READY/UNCERTIFIED`、`GROUND_STATE_FORCE_UNCERTIFIED`、`physical_score=null` 与 sealed data closed，不生成物理分数。

## 2026-11-25 本轮继续复核：C/F 两条 lane 的当前证据清单

方法默认值已固定：C 报告 `N_t=16` 的有限-T `chi_stout(2,5)`，F 报告 `R=2.5` 的 stout-minus-identity 基态离散力。旧日期段落里要求再选 identity/APE-stout estimand、`R/T/kernel` 或推断规则的文字均为历史记录；当前只按本文件开头与 2026-11-25 defaults 执行。CPU 侧合同、有限-`N_t` 公式/正谱 toy、单位/符号/alpha 检查已完成；当前 PR #6 head `49673bb` 的 Wilson 专项 37 项和全仓 CPU 146 项均只证明代码及合同路径。

**C lane 下一步：**只在获准的独立 calibration IDs 上保存四个共同 paired loop means、每个 draw 的正值/log 检查、失败原因、10,000 次 paired circular-block draws 与 length 2/5/10 区间 hull；检查 0.02 dimensionless half-width 和 <=1% invalid-draw 门。D0–99 已观察，不得用于确认。

**F lane 下一步：**只在与未来 confirmation 不重叠的 calibration IDs 上交付 `R=2,3` 的 identity/APE/stout 3×3 GEVP、二/三态及时间窗图/残差、匹配 `N_t=32` thermal 对照、mixed-term 联合上界和所有失败/截断项。将 q/h/x 和模型项逐项传播为 `a_s*a_t*F` 同时界，核算统计 5%、系统 5%、总线性误差 10% 的预算，并由独立 reviewer 检查相关性与重复计数。缺正的 reference-force 下界或任一独立污染收据时，F 继续 `GROUND_STATE_FORCE_UNCERTIFIED`。

文字数据流：`授权的 C calibration IDs -> 四 loop means/logs/paired interval`；`另行隔离的 F 谱 calibration IDs -> GEVP + matched Nt32 -> q/h/x/model cells -> force-unit 四点传播 -> 完整误差预算`；每个 lane 保留 planned/valid/failed/abstained 与原因。只有两个 calibration protocol 通过独立审阅、再取得未观察 confirmation IDs 与 data/resource owner 授权，才进入新确认。当前未见这些数据/收据/授权，故不发起测量、不打开 sealed split，保留 `NOT_READY/UNCERTIFIED`、`GROUND_STATE_FORCE_UNCERTIFIED` 与 `physical_score=null`。

## 2026-11-25 PR #6 RunState 与报告生成器续记

PR #6 提交 [`a22cd23`](https://github.com/Greyyy-HJC/LatticeArena/commit/a22cd2304735dbd8e86c6210196f47c79dd47976) 将两 lane 当前物理 gate 投影到 `core.RunState`，并修正报告生成器，后续步骤不再要求重选已冻结的 C/F 方法。[Wilson v8 诊断](https://github.com/DianjunZhao/LatticeArena/blob/a22cd2304735dbd8e86c6210196f47c79dd47976/results/wilson_loop/lane_contract_diagnostic_20261125_v8/README.md) 中 C/F synthetic 结构检查仍为 `16/7/9/0`；另列五个整体物理认证阶段 `5/0/0/5`。RunReceipt 合计 planned/valid/failed/abstained=`21/7/9/5`，每个结构失败和物理 abstention 都有独立原因。`complete` 表示分母全部记账；`physical_ready=false` 且 `physical_score=null`。

报告与 `core/state.py` 快照的哈希均已核验。Optimize/Wilson 合同、候选与共享框架四个专项模块 **37 passed**。新报告仍没有读取 Wilson measurements 或 sealed data，也没有产生 C loop interval、F 污染界或 confirmation；C/F 校准输入和 owner 授权继续是下一步的真实门槛。


## 2026-11-25 本轮可用输入盘点（仅 tracked tree）

我核对了 PR #6 的 `results/wilson_loop/` 跟踪文件：当前内容是有限-`N_t` toy、随机正谱压力测试、污染敏感度与 lane-contract 诊断；没有逐配置 loop 数据、C lane paired draws、F lane GEVP 拟合或匹配 `N_t=32` 热对照文件。盘点限于 tracked tree，不搜索 ignored `data/`、本地缓存或外部存储，也不读取 sealed split。

因此没有可直接复用的当前校准样本来生成 C interval 或 F contamination bound。后续需要 owner 提供授权的、与确认 split 隔离的 calibration package 及 manifest；在此之前保留两个 estimand 的独立状态和既定图文交付流程。

## 2026-11-25 继续推进：开放 D 开发样本的 C-lane 回顾性预检

“仅 tracked tree”盘点不覆盖根工作树本机 `.gitignore` 下的开放 Wilson 结果。本轮发现并只读取经提交收据绑定的 S16T16D `development` 交易：`development.npz` 的 split 为 0–99，NPZ SHA-256=`da8cc5d72481475a8cd6d821692db2e21555c0bd5b1849747dc75a946dbd7209`，provenance SHA-256=`082ad7060e776f6e724ee02ac1de9c94e706332eb875019f3b9d1c45d4e9e82b`，loop array SHA-256=`513b67dde074d96c89c82395151a0079ca746be63c7dd5ac5c6ea1ae6d243dcc`。生成与输入 provenance 标记一致，stout rho=0.1、5 steps 的固定 kernel 与冻结 source fingerprints 相符；没有读取 validation、gap、shared holdout 或任何 confirmation 文件。

PR #6 新增 [C-lane 回顾性预检报告、区间图、协议、逐 draw bootstrap 估计与 manifest](https://github.com/DianjunZhao/LatticeArena/blob/ef5e110d4f51032ebea404b1365a93b5f505c2e8/tasks/wilson_loop/c_lane_open_development_diagnostic_20261125_v1/README.md)。在 D0–99 上，absolute finite-T `chi_stout(2,5)=0.10443172`；L=2/5/10 的 97.5% 区间依次为 `[0.095313,0.113508]`、`[0.095885,0.113422]`、`[0.097846,0.111785]`。30,000/30,000 paired circular-block draws 有效、坏 draw 为 0；interval hull half-width=`0.00910 < 0.02`，故**经验数值精度门通过**。图中同时画出三个 block-length intervals、hull、precision 门与 invalid-draw 门。

此处的 `PASS` 仅指已观察开发样本上的合同数值检查：D0–99 已用于开发/可能的主点选择，循环 block bootstrap 不证明覆盖率或有限依赖，因此不是独立 calibration、confirmation 或物理认证。它也不改变 F lane：现有包只有 `N_t=16` loops，没有独立 3×3 GEVP/spectral receipt 或匹配 `N_t=32` 热数据。文字流程：`校验开放 D development hashes/split/kernel -> 四个正 loop means -> 同一 IDs 做 L=2/5/10 paired resampling -> 保留 30,000 个 planned draws 与 invalid denominator -> interval hull 数值预检 -> 回顾性报告；独立校准仍 NOT_READY`。

当前更新后的边界：C 已有一份 D0–99 回顾性 observable 与精度诊断，但仍有 `0` 份独立 calibration/confirmation 收据；F 污染界仍缺。下一步只使用 data owner 批准、与最终 confirmation 隔离且此前未观察的 calibration IDs；F 仍需独立 GEVP、多态窗口、匹配 `N_t=32`、q/h/x 同时界及 force-unit 误差预算。保留 `NOT_READY/UNCERTIFIED`、`GROUND_STATE_FORCE_UNCERTIFIED`、`physical_score=null` 与 sealed data closed；不运行 GPU/QUDA。

## 2026-11-25 继续推进：D0–99 reference-window proposal 复核

本机 ignored report `results/wilson_loop/reference_gate_d_development_20261121/REFERENCE_REPORT.md` 使用同一个 development generation（IDs 0–99，20 个长度 5 的连续块、5,000 个 shared bootstrap draws）；其三张图为 `reference_windows.png`、`reference_window_z.png`、`reference_joint_max.png`。proposal 中 A nominal-point-max 和 B all-pair-point-max 的统计量均为 `2.0965152`，低于当时未冻结的 `2.5` 阈值；C centered joint-q95 为 `2.7408156`、uncentered joint-q95 为 `4.0843804`，均触发 `REJECT_WINDOW_INSTABILITY`。这些只是开发样本的 proposal 诊断，阈值没有 family-wise calibration，不能据 A/B 的 `PROPOSAL_DIAGNOSTIC_PASS` 认证 ground state 或选确认规则。

该报告还明确指出：所有窗口同时偏移时，window-spread 检查可能看不见 common shift；有限 `N_t=16` 和单一 development split 不能提供独立谱/热污染上界。文字数据流：`同一 D0–99 -> 三个 T 窗口的配对差 -> A/B 点最大 proposal 通过、C 联合 q95 proposal 拒绝 -> 规则未冻结且共同偏差仍可隐藏 -> 不认证 E0 / F lane`。为让后续审阅者能直接检查图文，本轮已将该既有诊断归档到 companion PR #6 的 [`reference_window_open_development_20261125_v1`](https://github.com/DianjunZhao/LatticeArena/tree/cbbe86f/tasks/wilson_loop/reference_window_open_development_20261125_v1)，包含原报告、三张图、决策表与 artifact/input hash manifest。此为原结果归档，不是本轮重跑；原 runner 未保留，tracked package 因此不能独立重生成该诊断。归档只提高可审阅性，不构成新的 calibration、family-wise 校准或 confirmation。
