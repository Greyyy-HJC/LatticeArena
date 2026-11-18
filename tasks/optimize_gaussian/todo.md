# optimize_gaussian：有限 CPU 闭环与后续学习实验

审计基线：`21a96cf`，2026-11-18。**保留 todo：尚有明确未完成验收。**
本轮主 agent 实跑 80 tests 全过，用时 5.03s；未做 GPU 复测。
本文件是下一轮任务，不把已有产物重复列为未修 bug。
负结果、证据不足和设计否决均可作为有限任务包的合格结论。
完成以下 CPU 包不要求学习模型取得正收益；新GPU物理实验与这批CPU验收分开记录。

## 1. 可关闭的完成项与证据边界

| 完成项 | 证据路径 | 结论范围 |
| --- | --- | --- |
| 闭合 plaquette、link-centred 与横向对称特征 | `features.py`、学习核专属 tests | 旧 gauge/A1+ bug 已修，不再重做旧诊断 |
| 完整 BCB 响应/执行与不同场 Hermitian | `tests/test_optimize_gaussian.py::test_submission_matches_linear_response_and_is_hermitian` | 已闭环到 submission，不仅是 base correlator |
| BCB 数值档案 | `results/optimize_gaussian/audit_20261118_bcb/operator.json` | geometry=(3,4,5)，steps=1；submission/direct 最大绝对差2.274e-13，Hermitian最大5.684e-14 |
| stale basis 核心漏洞修复 | `train.basis_key`、mutation tests | 依赖内容、gauge/propagator内容、IDs已参与身份；不等于生产metadata可信性完成 |
| audit 核心修复 | `audit.py`、`results/optimize_gaussian/audit_v2/` | manifest分类、明确输出路径和拒绝覆盖已实现 |
| 统一入口历史点值重放 | `unified_v1/unified.json` | local/SLSQP点值重现；不等于学习核已完整CV |
| outer-training内重选线性候选 | `unified.select_on_train`、dcc0531 | fold内 scales、幂次与SLSQP重训已实现；身份/图/估计量仍见包B |
| 单算子非对易表达例证 | `teacher_v1/teacher.json` | 证明已知teacher存在谱外方向；不是物理学习胜出 |

代码地图：`features.py` 提供 gauge 特征，`kernels.py` 实现边输运，
`submission.py` 执行 K=BC(a)B，`train.py` 收集固定 W 响应并用 Powell 训练 theta。
`unified.py` 管理线性候选诊断，`teacher_check.py`/`p3_teacher.py` 是 toy 实验。
上游 reference/statistics 与公共框架由对应 task/总纲负责，不在这里另建评分体系。

## 2. 已有数值与图应如何解释

v9 `results.json`：fit=0–19、dev_val=20–59、历史 test=70–99。
raw/block2 learned risk=0.0092357620，Gaussian base=0.0196001510，比值2.122201826。
这不是对SLSQP的2.12倍改善；70–99多次观察，不能再称新盲测。

正式接口目录：`results/pion_projection/s16t16_v1/candidates/63c33e619e06704a/`。
folded/block5风险：SLSQP=0.0077027080，learned=0.0077207899。
R_SLSQP/R_learned=0.9976580312，95%CI=[0.7799244162,1.1756779868]。
结论是**未检出差异**，不是等效或超越；参考未认证时仅为条件诊断。
raw/block2与folded/block5必须分栏，不混成一个指标。

![v9 开发风险](../../results/optimize_gaussian/v9/fig_risk.png)

该图是同一数据划分上三个固定随机 W × λ 的开发风险。
它既不是三个数据种子，也不测试训练 W 的网络上限。

![现有冻结权重诊断分解](../../results/optimize_gaussian/unified_v1/fig_risk_decomposition.png)

此图来自 `fold_local`，使用全development已冻结的权重。
它不是 `fold_local_search` 中重新选权重/幂次的图，不能作为无选择泄漏CV成绩。
新 reselected 组 tuned/SLSQP 比值为1.704565、2.120560、0.811089。
三折均选 power index=5（H^32）；SLSQP只在2/3折优于tuned。
这些数值依赖当前估计量，包B修正前不得升级成完整风险结论。

## 3. 共用执行规则

所有新增路径均为**拟议**：`results/optimize_gaussian/cpu_next_<runid>/`。
每个包独立保存配置、输入摘要、原始JSON/表、图及解释，不覆盖任何旧结果。
少量身份/残差诊断可用充分数值文字表代替图；其余须fig+data+文字。
先记录HEAD、dirty状态与依赖，保留他人改动；按实际修改运行相关测试，接口变更时补集成回归。
资源默认CPU/NumPy/SciPy/现有NPZ；相关新增测试按需运行。

## 4. CPU包A：生产metadata与状态闭包（P0）

当前缺口不是stale basis内容摘要，而是“输入摘要对应什么可信生产状态”。
collect/evaluate仍直接读取gauge/propagator；生产metadata残差/协议状态未闭环。
run_manifest缺硬件/精度/残差，reference_bootstrap内容未单独摘要。

动作：

1. 复用上游输入验证：geometry、boundary、精度、反演残差、gauge对应关系和cache metadata。
2. reference文件与bootstrap masses记录内容摘要、模型身份和来源split。
3. 记录软件/硬件/计时；未知字段写unknown，不用当前本机信息填旧GPU运行。
4. 区分verified、archived_diagnostic、unknown；历史缺字段不伪造补全。
5. audit末尾“new GPU runs NOT_RUN”改为“本审计未执行新GPU测量”，避免否认v9档案。

产物：拟议包A目录的identity覆盖表、输入mutation测试结果、验证receipt文字。
验收：metadata协议/残差/gauge对应关系不符即拒绝可信入口；内容未变不误拒绝。
历史不完整可只读诊断但不得自动授予verified。
停止：逐字段实现并测试后关闭；无法恢复的历史证据明确缺失即可，不启动远程补采。

## 5. CPU包B：统一估计量与reselected CV证据（P0/P1）

当前 `risk_with_fixed_amplitude` 点q使用outer-training amplitude，
但replica幅度amps从outer-test bootstrap重拟合，late inverse也取test。
因此point bias与replica variance不是同一算法，amplitude_source标签覆盖不足。

动作：

1. 明确estimand：训练冻结amplitude还是测试估计amplitude；point/replica使用同一算法。
2. 若传播训练幅度不确定性，联合重采样训练状态并维持与mass的相关性；不混用测试拟合幅度。
3. 明确late covariance属于训练状态还是测试估计量，单独记录来源。
4. 增加已知模型合成回归，检验point/replica一致性与不确定性解释。
5. 冻结权重历史组和fold内reselected组分别画图、分别命名。
6. 每fold保存train/test block IDs、weights、scales、mass/reference状态、hash和失败样本。
7. 每起点记录nfev、success、约束可行性与预算；最佳幂次保存index及实际power。
8. 若调λ/seed，须inner selection；没有inner CV时只称outer-training内拟合/重选。

全流程防泄漏：通道A_n、base amplitudes、late inverse、reference/masses、
谱区间、QR/SVD、正则和seed选择均不得读取outer-held-out数据。
历史全60cfg拟合状态不能直接带进新CV。
窗口/常数若事前固定可共享；每项来源必须可检查。
learned候选必须fold内重训，否则只标archived，不混入reselected排名。

产物：拟议包B目录的estimand文字表、fit-state身份表、合成诊断图、
两组独立风险分解/配对图及底层JSON；参考失败比例单列。
验收：point/replica同算法；外层ID不进入任何训练估计状态；新图确实取reselected组。
参考失败则diagnostic_only；不得挑fold、挑口径或称等效。
停止：上述闭环完成即结项；方差大或参考不识别是可接受负结论。

## 6. CPU包C：真正跨links teacher与2×2（P1）

当前 `p3_teacher.py` 写 H_train=H_test、Q_train=Q_test。
独立的只有probes，coverage/docstring中的held-out links/跨样本PASS应纠正。
eps=.1时谱类误差0.01268238、增广6.744e-16，commutator=.39935044。
这仅是同一已知算子的系数恢复，不是学到gauge→Q，也不是训练W。

动作：

1. 为train/test独立生成links，保留分开的seed与IDs。
2. 使用同一确定的gauge-conditioned Q[U]规则，避免给每个样本任意teacher答案。
3. train拟合全局系数，冻结后在新links/probes预测；同时保留cross-probe对照。
4. Q必须经过gauge/O_h/Hermitian属性检查；随机scalar V不能自动声称gauge-conditioned学习。
5. 实现CPU两类（谱/嵌套空间）×两优化器（SLSQP/Powell），另列least-squares oracle。
6. 统一约束、起点、预测信息；分别parameter/effective-rank matched与budget matched。
7. 报告多seed、epsilon扫描、有效秩、训练/测试误差、nfev、wall-time与失败。

产物：拟议包C目录的epsilon–误差图、2×2误差/成本矩阵图、属性残差表、原始数据及解释。
验收：epsilon=0谱类可恢复；epsilon>0新类是否提升由新links误差决定。
若已知Q包含teacher只能证明系数迁移，必须明确尚未学习Q映射。
停止：完成预设seed/epsilon/预算即可，无收益也结项；toy无需等待GPU。

## 7. 后续物理学习路线：假设与门槛，不自动执行

SLSQP是优化算法，不是核函数类。当前基线是七通道扩散多项式及其优化流程。
H=Vdiag(h)V†的任意p(H)共享本征向量；非对易空间算子可混合这些方向。
表达能力扩大是否有用，取决于真实overlap/噪声信息、独立块数、参考和成本。

| 路线 | 候选及必要约束 | 进入物理实验前的门槛 |
| --- | --- | --- |
| A嵌套空间修正 | K=P_u(H)+Σtheta_j B_jQ_j[U]B_j，保留identity与原基线可行点 | 固定特征响应、OperatorContract、训练内有效秩；theta=0严格恢复基线 |
| B强谱/路径基 | wider polynomial/Chebyshev；逆路径配对、O_h轨道和谱放大约束 | 先排除七幂覆盖不足；谱范围/正交化fold内估计 |
| C真正训练W | gauge-only网络输出实边系数；K对quark field严格线性 | raw features/局域密度缓存或动态可微收缩；有限差分梯度、成本与内存预算 |

类包含仅保证相同约束/正则下训练global infimum不差，不保证优化、泛化或成本。
改变W会改变tanh特征，固定W的C_fit不能训练任意W。
后续至少比较real/uniform/spatial-off/shuffled，假特征也需保持gauge/O_h。
不能随意shuffle links或颜色矩阵；可对scalar特征预注册平移/群处理。
真实优于shuffled才支持局部信息解释；强谱基解释全部收益也是有效反证。

新GPU设计仍须固定实际意义阈值（不能只写“例如1.2”）、预算/轮次、
样本数power规则、主终点及多重比较口径，再进入可用GPU环境实施。
独立新数据同时测强谱类与空间臂，分列inversion/setup/apply/train/IO/total成本。
成功需预注册配对区间下界超过实际意义阈值，并对参考/块长稳定。
区间跨1为证据不足，不是等效；达到预算无收益则归档结束。

## 8. 公共框架适配与本轮完成条件

公共框架总纲由wilson todo负责，Gaussian仅交付适配：
OperatorContract（线性/对称性/identity）、MeasurementBundle（输入/响应/状态/成本）、
SearchAdapter（constraints/budget/seed/失败与收敛记录）。
产物是一个CPU端到端适配实例及字段验证文字表；不能仅引用core接口存在。
多动量扩展另立little-group与复相位协议，不沿用零动量A1+标签。

本轮包A/B/C及适配产物完成后，可关闭这批CPU任务。
尚未进行的GPU实验或真实W训练列后续研究，不妨碍CPU包有限结项。
是否删除todo由用户要求与剩余已授权范围决定；不以coverage文字代替逐项验收。
