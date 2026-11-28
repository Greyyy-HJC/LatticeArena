# optimize_gaussian：CPU 验收结论与后续研究

### 当前控制状态（2026-11-28；以下替代早期“等待选择”交接）

研究合同已给出可审阅的方法默认值：D04 用独立、candidate-blind E-chain PP/A4P 联合拟合；D06 保留 trainable-$W$ 对同预算 SLSQP 和 degree-4 Chebyshev 两项对比、`F=2`、`r_star=10/9`；bounded block-risk 是用户已接受的并列 estimand；D11 degree-4 Bernstein/Chebyshev 强谱算符和 160-call 单任务预算也已写入合同。D06 bounded-risk Markov 路线现除两态 reversible stress 外，又有非可逆 19 态已知-gap/初始-TV 图文校准，但仍只是合成覆盖证据。D04 iid 与 AR(1) nested-bootstrap reader 已改成将失败 refit 作为最不利上尾超越计入保守 p 值；既有 v1 报告的所有内层拟合均成功，因此历史数值不变。Wilson F-lane 有限-$N_t$ q/h/x sector 对应也已作为独立审阅前的方法提案登记，见 [`tasks/wilson_loop/todo.md`](../wilson_loop/todo.md)。TODO 仍开着，因为 E-reference 与配对 solver receipts、独立统计/算符审阅、生产成本测量和相应授权缺失。维持 `NOT_READY/UNCERTIFIED`、`physical_score=null`；不读封存 split、不启动 GPU/QUDA。下方早期时间戳条目保留审计历史，方法选择以本节和文末最新 handoff 为准。

2026-11-28 跨任务复核补充：Pion 除 `m_eff(4)` 正两态谱不可识别性反例外，又构造出两组 ground mass 相差 `0.30`、振幅全正且在四个预选时间比值内约 `1.8e-15` 重合的周期谱；这仍只是方法反例，不是 Optimize 数据，也不替代其 D04 reference。Pion 另有仅在 per-unit positivity/质量上界成立时的充分污染阈值；Wilson F lane 的 24 个遗漏静态态重复面板显示固定 2×2 GLS delta 上界在该失配 toy 下 familywise 为 `0/24`。这些结论都不是 Optimize 的 reference 或 solver 数据，不能替代 D04 E-reference、D06 paired macroblocks 或 D11 SU(3) receipt。本轮不重跑已有 Optimize synthetic 网格；在新授权输入之前，不再用 seed-only 扩样来制造“推进”。

### 2026-11-28 D06 有界风险输入的实数与范围 fail-closed 补齐

检查 D06 两个 risk-decision 入口时发现，它们把调用者提供的矩阵直接 `np.asarray(..., dtype=float64)`。NumPy 会对复数发出 warning 后静默丢弃虚部；超出 float64 的 Python 大整数会向外抛 `OverflowError`；字符串和布尔也会被隐式转成看似有效的风险值。现让 iid empirical-Bernstein 与 Markov pseudo-gap 两路共用严格实数数值数组转换：只接受整数/浮点数组，显式拒绝 complex、object/string、bool 与不可表示形状/转换；之后再由原有 `[0,1]` 有界性检查挡住非有限、负数或超界值。q→bounded-loss 映射本身未改，solver failure 仍必须显式标记并计最大损失 1。

文字诊断：`raw block score matrix -> require rectangular int/float dtype -> reject complex/object/string/bool without projection -> float64 conversion -> finite [0,1] and failure-mask checks -> iid or Markov decision`。负控覆盖复数带非零虚部、`10**400`、数值字符串和布尔矩阵；窄范围回归 **37 passed**，本轮 PR #6 全套 CPU tests **496 passed**。这关闭的是输入类型/数值转换旁路，不验证 macroblock 来源、独立性或 `gamma_ps`/TV receipt，也不校准 bounded block-risk coverage；D04 风险比 estimand 保持独立，D06 两项对比仍是 trainable-W vs 同预算 SLSQP 和 degree-4 Chebyshev，`r_star=10/9` 不变。没有访问 measurement/calibration/sealed bytes、没有新跑 synthetic panels、没有 GPU/QUDA；物理状态仍 `NOT_READY/UNCERTIFIED`、`physical_score=null`。

### 2026-11-26 D04 raw E-chain reader 与联合拟合诊断

新增 [`d04_raw_reference.py`](d04_raw_reference.py)：从同一 `SourceByteSnapshot` 读取严格 schema 的 manifest、source-protocol JSON 与 raw correlator NPZ；核对 protocol bytes 是否匹配 manifest 摘要，并核对 `S16T16`/quenched Wilson/`kappa=0.14`、PP channel/dimension order、升序 config/trajectory IDs、全部 source hashes 和 closed split；明确拒绝 complex data 而不隐式取实部。按合同作 PP-even 与 A4P-odd fold，共用 `t=2..8` covariance、25% 对角 shrinkage 和 `1e-10` relative eigenvalue floor，跑六种 `tmin={2,3,4}` × `nstates={2,3}` joint fit，每种三 deterministic starts，并记录不收敛、边界和近简并谱诊断。图文/结果范围见[报告](d04_raw_reference_cpu_20261126_v1/README.md)。

synthetic fixture 的六拟合分母为 `6/6/0/0`，18 个 multistart 中 17 个收敛、1 个失败；`tmin=3` 两态拟合恢复输入 `a_t m_0=0.32` 为 `0.3200056`。三态拟合出现小能隙，报告保留该弱可识别警告。测试也覆盖捕获后改写源文件、folding、封存 split、坏 ID、complex 数据、stale bytes 与 protocol hash 不匹配；本次新增定向测试 **7 passed**。所有这些都是合成 CPU 软件证据。

这个入口仍**不输出 gate-ready summary**。之后新增的 D04 synthetic MBB 报告已给出依赖敏感度下的 bootstrap mass interval 和 PP/A4P/window/model paired-SE 位移；但它没有 E-chain dependence/provenance receipt、独立 reviewer/授权，也未校准拟合 GOF 的有限样本行为，故只关闭 CPU 方法实现，不关闭 reference gate。下一步仍需独立复核 PP/A4P 符号、channel source/sink 约定、block bootstrap 与 mass-shift 统计定义；随后从同一次获准读取派生 raw receipt 与 D04 envelope，再接同 snapshot gate。没有经审阅的获准 E-chain bytes 前不生成 reference pass，不读 sealed data、不启动 GPU/QUDA，`NOT_READY/UNCERTIFIED`、`physical_score=null` 不变。

## 下一次执行包（默认研究输入已自选）

不要再重复同一合成 generator 的 panel、block length 或 seed 扫描。下一步只接入**明确获准的 development/calibration bytes**；没有完整 bytes 与来源/审阅收据时，继续 `NOT_READY/UNCERTIFIED`、`physical_score=null`。

1. **D04 reference package：**独立于 Optimize score/evaluation chain 的 candidate-blind E-chain；包含零动量 PP-even 与 local A4P-odd correlators、升序 configuration/trajectory IDs、`S16T16`/Wilson/`kappa=0.14`/时间边界与协议 hash、源文件 SHA-256。按 `research_contract_v1.json` 固定 joint two-state PP(n=0,8,16)+A4P 拟合：主窗 `t=3..8`，`tmin={2,3,4}`、`nstates={2,3}`，25% 对角 shrinkage 与 `1e-10` 特征值 floor；不得按 arm 改窗或协方差。D04 必须同时满足相对 mass error `<=5%`、`p>=0.01`、PP/A4P 与窗口/模型差异 `<=2.5` paired SE、无边界命中；任一失败就不计算确认性风险比。
2. **D06 三臂 paired package：**相同 macroblock IDs/配置/随机数计划上的 trainable-W、同预算 SLSQP、degree-4 Chebyshev；锁定 order 与时间点 `(2,3,4,5)`，每个 block 保存三臂四时 q 值、failure mask、solver status、objective/gradient calls、residual/KKT 和分项时间。主判定保留 `rho_SLSQP=R_SLSQP/R_W` 与 `rho_Cheb=R_Cheb/R_W` 两项，`r_star=10/9`、总单侧 alpha `0.05` 对两项 Bonferroni 分配、10,000 paired draws、block length `{2,5,10}` 的 lower-bound hull 必须对两项都超过 `10/9`；零/非法风险直接 `ABSTAIN_RATIO_UNDEFINED`，失败臂按冻结 penalty 留在分母。Markov 路线另外需要同一 paired macroblock chain 的独立 `gamma_ps`、burn-in、post-burn-in TV 和固定 score-support receipts；它不能由轨迹间距或 ESS 代替。
3. **D11 operator package：**同一 source snapshot 绑定 spatial link bytes、维度/边界、operator code、SU(3) provenance/determinant-unitarity receipt 与 `H`/source metadata；按合同的 degree-4 Bernstein controls、`H∈[-12,0]`、`y=I+H/12`、`P4(y)=sum b_i C(4,i)y^i(1-y)^(4-i)`、`0<=b0<=...<=b4=1` 和 `G^2 P4(H) G^2` 计算。160 objective/gradient-call cap 内同 budget，不 clipping、不以数学 U(3) 谱界冒充物理 SU(3) provenance。

**图文验收物：**同一配对 panel 的两项 log-risk-ratio 区间图（标出 `10/9` 门和完整分母）；每臂 calls/time/residual/KKT/训练与 setup 成本分解图；D04 所有窗口/态数诊断；D11 positivity、spectral response 与来源 hash 流程图。报告必须保留 `planned/valid/failed/abstained`，并说明数据 split、CPU/GPU、精度、线程数和是否计入训练成本。Independent review 与相应数据/设备授权仍是进入真实评估前的条件；不能用新 schema 或合成图声称物理 gate 通过。

### 2026-11-27 当前 CPU 验证快照

当前 PR #6 checkout 的 Optimize 专项文件集 `tests/test_optimize*.py` 为 **159 passed**；Wilson 专项文件集 `tests/test_wilson*.py` 为 **195 passed**（含 11 个 transfer-trace reader/report hash checks）。Optimize+Wilson 定向集合为 **354 passed**，PR #6 当前完整 CPU suite **437 passed**。PR #5 Pion 当前完整 CPU suite **268 passed**（包含同快照 common-bias receipt consistency checks）；Pion TODO 与其工作树分开更新。所有结果只代表 CPU 软件/公式回归；没有产生真实 E-reference、生产 paired solver/cost、SU(3) operator evidence、独立 review 或数据授权，不能升级 `NOT_READY/UNCERTIFIED` 或设置 `physical_score`。

### 2026-11-26 历史附件链接审计

本文件 115 个 Markdown links 均已可解析：当前 checkout 内的目标保持相对路径；原历史报告/决策文件已从当前 checkout 移除的 61 个链接，现固定指向 fork 中仍可达的 Git commit permalink。历史正文、数字与结论未改写；这些旧报告仍只作为其原有范围内的历史诊断，不是新的物理证据。

### 2026-11-26 trainable-W / SLSQP / degree-4 Chebyshev 扩样比较

为避免旧版每 cell 仅 3 个合成 panel 就判断 learned-W 稳定性，将 [`trainable_w_stress_diagnostic.py`](trainable_w_stress_diagnostic.py) 扩成可预设独立 panel 数，并增加 cell-median 配对 bootstrap bands 和 Chebyshev-4 的直接 heldout contrast。紧凑可审阅包（含 240 行逐 panel 配对表、协议、三张图和 SHA-256）在[报告目录](trainable_w_replication_20261126_v1/README.md)；完整 trace 与合成输入数组由 `protocol.json` 指定的 seed/source 可重建到本机 `results/` 输出目录。

扩样固定每格 20 个独立 synthetic panel，共 12 格、240 panels；每 panel 两个 fit link + 两个独立 heldout link，并行配对 trainable-W SLSQP、trainable-W L-BFGS-B、固定-W width-2 SLSQP、与 joint width-2 参数数目相同的 fixed-W width-6 SLSQP，call caps 为 32/128。`spectral_heat` 目标额外对照 degree-4 Chebyshev；它在同一 fit/heldout links 上用线性最小二乘拟合，不与非线性 SLSQP 匹配 calls/walltime。图把近 1 的 SLSQP 对照与更宽动态范围的 Chebyshev 对照分开。每格 median 的 10,000-resample percentile band 仅描述这组合成生成器下 Monte Carlo 中位数不确定度，没有 coverage calibration，也不是物理置信区间。

结果：panels `240/240/0/0`，fit links `480/480/0/0`，heldout links `480/480/0/0`，solver fits `1920/1920` 返回；566 次达到 call cap 且 optimizer 非 success，全部保留为有限预算观察，没有丢弃。local-environment 六格中，trainable-W 相对 fixed-W width-2 的每格 win fraction 为 `0.30–1.00`（格均值 `0.63`）；对等参数数目 fixed-W width-6 为 `0.30–0.65`（格均值 `0.45`），不支持稳定的 learned-W 优势。spectral-heat 六格相对 width-2 / width-6 的 win fraction 分别为 `0.35–0.50` / `0.50–0.75`。对刻意由谱热核生成的 `spectral_heat` 目标，Chebyshev-4 每格中位 heldout loss 都较低：trainable-W/Chebyshev loss ratio 的 cell median 范围约 `16–8.3×10^6`，六格几何平均约 `4.3×10^3`；所有 cell 的 bootstrap log-ratio lower band 仍大于 0。noise=2% 时中位比降为约 `16–35`，无噪声下差距扩大到约 `7.1×10^4–8.3×10^6`。这说明目标家族与强谱基相匹配时谱基有明显优势；不说明物理 smear 的排名，也不证明相同成本下 Chebyshev 更优。

测试覆盖 seed-plan 均衡/唯一性、bootstrap band 确定性和输入拒绝；本轮 focused `tests/test_optimize_trainable_w_stress.py` **9 passed**。合成输出 hash、图和描述均自包含于报告包。下一步仍需新 gauge-independent DGP 与真实 operator/score evaluator 才能判断 learned smear 的物理能力；先由独立 reviewer 核对合成 loss 是否可作为筛选代理，再取得独立 E-chain D04 reference 和授权后，才能将两项用户选定对照（trainable-W vs 同预算 SLSQP、trainable-W vs degree-4 Chebyshev）接入 production。保持 `NOT_READY/UNCERTIFIED`、`physical_score=null`，不读 sealed split、不启动 GPU/QUDA。

### 2026-11-25：D06 percentile hull synthetic coverage pilot 显示欠覆盖

实现 [`d06_bootstrap_coverage_diagnostic.py`](d06_bootstrap_coverage_diagnostic.py)，用已知平稳 Gaussian AR(1) 生成三臂配对 `q_theta(t)`（trainable-W、同预算 SLSQP、degree-4 Chebyshev），以精确 AR(1) 有限样本 `Var(mean)` 作风险比真值。每个外层 circular-block panel 内，D04 风险采用经验 circular-block bootstrap 的解析无限内抽样矩；外层再重抽 L=2/5/10 的配置块，构造 comparator/trainable-W 风险比及每长度 0.025 lower quantile，取 hull 最低值，同时评估 F=2 familywise coverage。已分别做 40/60/100 配置、phi=0/0.4/0.8 的 1,000-draw pilot，以及 `(n=60, phi=0.8)` 的合同 10,000 draws 复核：全量表、Wilson Monte Carlo 区间、分长度诊断与 denominator 见[样本量敏感度报告 v7](../../results/optimize_gaussian/d06_percentile_hull_coverage_cpu_20261125_v7/README.md)及[10,000-draw 欠覆盖复核 v8](../../results/optimize_gaussian/d06_percentile_hull_coverage_cpu_20261125_v8/README.md)。两包的 artifact hashes 和模拟脚本 hash 均匹配。

结果**不支持将当前 bootstrap percentile hull 当作已校准推断**：每格 250 个 simulated panels，P/V/F/A=`250/250/0/0`。`n=60` 的 F=2 joint coverage 在 phi=0、0.4、0.8 分别为 `95.6% [92.3%,97.5%]`、`91.6% [87.5%,94.4%]`、`75.6% [69.9%,80.5%]`；phi=0.8 时 n=40/100 也仅 `79.2% [73.7%,83.8%]` / `78.8% [73.3%,83.4%]`。合同 10,000 outer draws 的 n=60、phi=0.8 复核仍为 `76.4% [70.8%,81.2%]`；SLSQP/Chebyshev 的 marginal lower-bound coverage 为 `86.8%` / `85.6%`，都低于单比较 97.5% 目标。L=2/5/10 的 F=2 coverage 分别为 `64.4% / 72.0% / 76.0%`，最低值 hull 只小幅改善到 76.4%。因此该欠覆盖不是把 outer draws 从 1,000 增到 10,000 就能消除的本模拟现象。

解释边界：这证明的是**本报告具体的嵌套 block-bootstrap 实现**在所选 AR(1) 生成器/风险真值下欠覆盖；它不是实际 gauge 链的 phi 估计，也不能自动否定未来不同的 production estimator。但它足以让 D06 percentile 规则保持 `UNCALIBRATED/NOT_READY`，即使数组 reducer 算术正确也不能报告确认性 PASS。继续工作应保留同一 D04 风险比 estimand、F=2 alpha 与 r_star，对更长 block / studentized 或经偏差校准的区间做事前比较，扩展 MC 误差预算并让独立统计审阅者确认风险真值/嵌套算法；只有在已知 AR(1)、强尾/方差差异及配对相关情景下均恢复名义覆盖后，才把所选方法接入真实 evaluator。不得用本模拟调参后直接读取 sealed IDs。

### 2026-11-25：D06 配对 influence-HAC 候选与全网格覆盖诊断

为同一 `D04` 风险比写出有限 `L` empirical-risk 的解析 influence，并构造配对比值 Newey–West/Bartlett 单侧下界；HAC 方差的带宽 `2/5/10/20/30` 与 D04 `L=2/5/10` 全部取最低下界。先在和 v8 相同的 250 个 `(n=60, phi=0.8)` 面板、相同随机种子上，用 10,000-draw percentile hull 作同样本比较，见[定向比较 v4](../../results/optimize_gaussian/d06_hac_coverage_cpu_20261125_v4/README.md)：percentile 与 HAC 的 F=2 覆盖率为 `76.4%` 与 `93.2%`；HAC-minus-percentile 配对差 `+16.8 pp`，面板 bootstrap 95% 区间 `[+12.0,+21.6] pp`。HAC 的自身 Wilson 95% 蒙特卡洛区间为 `[89.4%,95.7%]`，未证明达到 95% familywise target。

随后扩展到已归档的 `n=40/60/100 × phi=0/0.4/0.8` 网格，每 cell 250 面板，同种面板同时对比 HAC 与 1,000-draw percentile hull，图文和全量配对变化见[九格覆盖报告 v2](../../results/optimize_gaussian/d06_hac_grid_coverage_cpu_20261125_v2/README.md)。HAC F=2 覆盖率按 `phi=(0,0.4,0.8)` 分别为：`n=40: 98.4%,97.6%,95.2%`；`n=60: 94.4%,96.8%,94.4%`；`n=100: 90.0%,90.8%,82.8%`。相对 percentile 的 familywise 配对变化分别为 `+3.2,+8.0,+16.0 pp`、`−1.2,+5.2,+18.8 pp`、`−4.8,−2.4,+4.0 pp`。每格 P/V/F/A=`250/250/0/0`。因此 HAC 在部分中高相关 cell 改善明显，但在 `n=100` 各 cell 和 `n=60,phi=0` 没有改善；它不是全网格稳定校准的替代方法。

解析风险梯度通过逐配置有限差分核对，新增 HAC/grid 汇总测试 5 项；本报告仍是 Gaussian AR(1) 合成方法比较。全网格 percentile 基线只有每长度 1,000 draws，目标合同为 10,000；HAC 又依赖未被证明的渐近线性化、弱依赖与矩条件，当前没有 heavy-tail、非平稳、D04 reference-fit 不确定度或 optimizer failure 覆盖。故 D06 继续 `UNCALIBRATED/NOT_READY`。后续应冻结更广 DGP 与 Monte Carlo 停止规则，比较 studentized / bias-corrected / conservative ratio bounds，并由独立统计审阅者复核后，才考虑真实 evaluator 接线；不能根据这 9 格选出最优带宽，也不读 sealed IDs。

### 2026-11-25 D06 HAC 对比反演（Fieller-style）未修复欠覆盖

新增 [`d06_hac_fieller_diagnostic.py`](d06_hac_fieller_diagnostic.py)：把 `R_comparator/r - R_trainableW` 的 HAC-studentized 单侧检验直接对 `r` 反演，构造风险比下界；同一套输入上与 influence-ratio delta HAC 对照，保留 `L=2/5/10`、bandwidth `2/5/10/20/30` 的下界 hull 与 `F=2` familywise rule。精确有限样本风险真值来自高斯 AR(1)、有限四阶矩 Student-t5 AR(1)、period-2 异方差 AR(1)；另外扫描配置相关 `phi=0/0.4/0.8`、cross-arm common loading `0/0.8`、`n=40/60/100`，每 cell 1,000 个独立面板。报告含逐 cell Wilson Monte Carlo 区间、两种方法 coverage/release 图、raw paired panels、协议和哈希，见[图文 stress v1](../../results/optimize_gaussian/d06_hac_fieller_stress_cpu_20261125_v1/README.md)。分母 `54,000/54,000/0/0`；名义 95% coverage 的 MCSE 约 `0.0069`。

结果未支持 Fieller-style 反演作为替代：全部 54 个 cell 中，ratio-delta 的 familywise coverage Wilson 下界在 `15/54` 个 cell 达到 `0.95`，contrast-inversion 在 `1/54` 个 cell 达到；**两者都没有达到全网格要求**。跨每个 DGP 的等权平均 family coverage，ratio-delta / contrast-inversion 分别是 Gaussian `0.9424/0.9143`、Student-t5 `0.9447/0.9138`、periodic volatility `0.9409/0.9119`；各 DGP 的最差 cell 约为 `0.882–0.894` 与 `0.822–0.849`。Fieller-style 有略高 release rate，但与更差 coverage 同时出现，不能解释为检出能力提高。结论仅限这组已声明风险定义与 DGP；没有包括 D04 reference-fit uncertainty、mean drift、solver failures 或 lattice 数据。新增方法测试 **7 passed**；本 checkout `tests/test_optimize*.py` 总计 **88 passed**。

下一步不要把 ratio-delta 或 contrast inversion 接入确认 evaluator。现已对**不同的 bounded block-risk estimand**实现并压力测试有限样本界，见本文件末尾；它只在 independent macroblocks 假设下有效，未替代当前 D04/D06 合同。需要独立审阅后决定是否版本化为新 estimand，并先取得独立 block 或有效依赖校正凭据；否则维持 `ABSTAIN/NOT_READY`。之后仍需同预算 SLSQP 与 degree-4 Chebyshev 同时保留，真实 trainable-W、reference-fit 传播及盲测 gate 不因 CPU 诊断而改变。

### 2026-11-25：D04/D06 风险统计量的 fail-closed CPU reducer

新增 [`risk_statistics.py`](risk_statistics.py)，将已选择的 D04/D06 公式做成纯数组 reducer：D04 按 `t=(2,3,4,5)`、`ddof=1` 计算每时刻 `Var_boot(q)+[E_boot(q)-1]^2` 并取平均；缺 draw、NaN/Inf 或 draw 数不足时保留计划分母并返回 `risk=null`。D06 固定 `F=2` 的每比较 `alpha=0.025`、`r_star=10/9`、`L=(2,5,10)`，要求 comparator/trainable-W 每个长度的风险 draw ID 一一相同且唯一，风险严格正；使用 linear 0.025 quantile 并对三种长度取最低值。任一配对缺失、零风险、非有限或除法溢出均 `ABSTAIN`；不丢弃坏 draw、不补 epsilon、不生成物理分数。

可复算文字诊断见 [`d04_d06_risk_statistics_cpu_20261125_v4`](../../results/optimize_gaussian/d04_d06_risk_statistics_cpu_20261125_v4/README.md)：D04 手算 fixture 分母 `3/3/0/0`、风险 `0.784722222`；D06 三个四-draw synthetic fixture 分别展示阈值通过、未通过、零风险弃权，缺失/无效例保留 `12/11/1/0`。report JSON 绑定 reducer 和研究合同源码 SHA-256，artifact hashes 完整。`draft_plan_20261125_v6` 已把 reducer 加入 no-launch RunSpec 草案的源码依赖映射；该草案仍为 `DRAFT_NO_LAUNCH`，物理门 `5/0/0/5`，输入分母 `0/0/0/0`。

**此项完成的是统计公式的数组层实现和失败处理，不是 bootstrap 实施或校准：**外层按 block length 的 paired configuration 重抽样、每个外层 draw 内 D04 `q` 分布、参考拟合不确定性传播、draw ID 的配置来源证明、真实 evaluator/RunSpec 接线仍未做。纯统计 helper 现有 4 项 synthetic coverage tests 加四 draw arithmetic fixture，不能取代上段所示欠覆盖；正式确认仍需校准后的区间方法及真实合同 10,000 draws。加入影响函数梯度与九格覆盖汇总后，当前整个 Optimize 测试文件集 `tests/test_optimize*.py` 为 **81 passed**（CPU only）。

下一步：percentile hull 与 influence-HAC 都已在部分指定 AR(1) cell 中显示欠覆盖；先冻结涵盖强尾、异方差、跨臂相关、reference-fit 传播的校准网格、MCSE/停止门与 complete denominator，再比较 studentized、偏差校正和保守风险比界。独立统计审阅者需复核风险真值和区间实现。接入当前 task-specific trainable-W、同预算 SLSQP 与 degree-4 Chebyshev 三臂前，先交 CPU fixture 与 coverage 图/表，再申请 production/GPU。当前没有合格 D04 reference 或生产数据授权，因此任何中间结果继续 `diagnostic_only`、`physical_score=null`；不读 sealed IDs。

### 2026-11-25 本 checkout 的 TaskSpec/RunSpec 接线审计

新增 [`draft_plan.py`](draft_plan.py) 将已选 D04/D06/D11 默认值传入 `core.specs.task_spec` 和 `core.specs.run_spec`，生成了[数据-free 草案和文字数据流](../../results/optimize_gaussian/draft_plan_20261125_v2/README.md)（旧 v1 保留为历史快照）。合同及源码 SHA-256 已绑定；只登记已开放的 D0–59 fit 与 D60–99 development-selection 元数据，没有读取配置字节、生成输入 hash、启动训练或连接 evaluator。schema 计划分母 `1/1/0/0`；五个物理认证阶段 `5/0/0/5`，`launch_allowed=false`、`physical_score=null`。

新增 [`test_optimize_draft_plan.py`](../../tests/test_optimize_draft_plan.py) 锁定这个 no-launch 接线必须保留用户选定的 F=2 两项对照：trainable-W vs 同预算 SLSQP、trainable-W vs 固定 degree-4 Chebyshev 强谱基，并检查 D06 `r_star=10/9`、160-call arm budget、空 data hash、未授权 GPU/production、5 个 physical gate 全部 abstain。新增测试与四个相邻合同/基线测试共 **36 passed**。这是计划结构回归，不是 RunSpec runner 或物理比较。

**重要的分支核验：**上述 builder/runner 源码并非丢失；它们在另一条本地 `dianjun` 分支（快照 `54f0e97`），但没有进入 PR #6 checkout。只读检查发现那条分支的 `train.py::main` 用 `model_space(p, seed)` 生成固定随机 `W`，训练/优化的是 `theta`，所以并非当前 primary 的 trainable-W；其旧 `protocol.json` 还硬编码旧单态 `m0`，并采用已观察的旧 `fit/dev_val/test` 切分。直接复制它会把旧 estimator/reference 和非 trainable-W runner 带入新合同。故本轮不整支合并、不启动它；新模块只完成通用 schema 的计划层接线，不是生产 RunSpec，也不关闭 worker 集成项。后续若继续接线，应从 `trainable_w.py` 与当前 D04/D06/D11 出发，选择性复用经审查的 worker/hard-timeout 基础设施，重做 task-specific runner 和产物回执，再用独立 CPU fixture 验证。执行前仍须有独立 E-chain reference、production evaluator/operator review、完整且已授权的输入 manifest。不要把本草案的 `validation` 键误读成最终 blind confirmation。

文字流程：`research_contract + 源码 -> SHA-256 -> 通用 TaskSpec/RunSpec -> 空数据 hash + 无 runner + 无授权 -> launch_allowed=false -> 五个物理阶段 abstained`。

### 2026-11-25 最新交接：F=2、D04/D06/D11 默认值已给出；生产证据门仍未通过

用户已确定主比较保留两个 estimand 对照（`F=2`）：trainable-W 对同预算 SLSQP、以及 trainable-W 对固定强谱基。用户随后要求由执行者自行提出缺失规则。以下工作默认已写入 [`research_contract_v1.json`](research_contract_v1.json)，状态仍是 `DRAFT_NOT_READY` / `METHOD_DEFAULTS_SELECTED_EVIDENCE_NOT_READY`；它们是供独立审阅的明确提案，不是外部 reviewer 批准，也不表示 production 或 GPU 授权。旧段落中“等待用户给 D04/D06/D11 数值”的要求均由本节替代。

- **D04 reference / risk：**目标提案为 S16T16、quenched Wilson、`κ=0.14` 的零动量价夸克 isovector pseudoscalar 能量，单位 `a_t m_pi`；以独立 E-chain 作 reference。reference 采用候选盲的 joint two-state PP(`n=0,8,16`)+local A4P spectral fit，主窗 `[3,8]`，检查 `tmin=2,3,4` 及 2/3-state；PP even fold、A4P odd fold、shared states、diagonal shrinkage 0.25、eigenvalue floor `1e-10`，所有 arms 共用窗口、协方差、blocks 和 reference draws。风险为 `qθ(t)=Cθ(t)/(Aθ f0(t))`，`Rθ=mean_{t=2..5}[Var_boot(qθ(t))+|E_boot(qθ(t))-1|²]`，振幅窗 `[6,8]`。候选 gate 为相对质量误差 `≤0.05`、`p≥0.01`、PP/A4P 与窗口/模型位移 `≤2.5` paired SE、无边界命中。E-chain correlator receipt、拟合结果和独立 review receipt 均为空；因此 D04 仍未实测。
- **D06 最小效应：**提案 `r_star=10/9=1.111111…`，定义 `ρ=R_comparator/R_trainableW`，要求至少 10% 风险下降；两项比较共用 one-sided `α=0.05`，Bonferroni 每项 `0.025`，10,000 paired block-bootstrap draws，主 block length 5、敏感度 2/10。三种 block length 区间的 hull 下界都必须大于 `r_star`；零风险或比值无定义则 `ABSTAIN_RATIO_UNDEFINED`，不得加 epsilon。
- **D11 强谱臂：**提案为 degree-4 Bernstein/Chebyshev 家族。对周期三维、等权、unitary 3×3 links，`H=Σ_i(T_i+T_i†−2I)`，`spec(H)⊂[-12,0]`，`y=I+H/12∈[0,1]`；这里 U(3) unitarity 足以证明谱界，真实 QCD production 仍需独立 SU(3) provenance/determinant 收据。`P4(y)=Σ_i b_i C(4,i)y^i(1−y)^(4−i)`，`0≤b0≤…≤b4=1`，故 `0≤P4≤1`、单调，且谱端点 `P4(y=1)=1`；这不表示任意 gauge 场都有 `H=0` 本征态。共同外层为未逐步归一化的 `G²P4(H)G²`，`G=I+0.10Σ_i(T_i+T_i†)=1.6I+0.1H`，`spec(G)⊂[0.4,1.6]`；`6.5536=1.6^4` 仅在存在 `H=0` 本征态时才是该态响应，不是配置级归一化因子。合同拒绝非幺正 links、错误形状/边界、非有限值和不可行控制点；D11 独立 review 仍为空。
- **预算提案：**每臂单一预注册起点，最多 160 objective/gradient calls；训练、setup、solve、inference、I/O、失败恢复和 total walltime 都须计账，目前真实秒数与 instance/fit/heldout/seed hashes 为空，amortization task count 暂按 1。

**已归档的图文诊断及可支持结论：**

### 2026-11-25：D11 与实际 hopping / Clenshaw 实现的合同交叉审计

源代码和代数交叉审计见[报告](../../results/optimize_gaussian/d11_operator_contract_audit_20261125_v1/README.md)：planned/valid/failed/abstained=`9/7/0/2`。七项在当前周期 `(z,y,x)` 存储、`(x,y,z)` 方向索引、成对 Hermitian hopping、unitary links 谱界、Chebyshev 标度、Bernstein 正性及共同 Gaussian sandwich 上一致；两项 abstain 是物理 QCD SU(3) provenance 尚未单独核验、以及 production objective/RunSpec 未接通。报告绑定 `research_contract_v1.json`、`contract.py`、`candidate_contract.py`、`features.py`、`kernels.py`、`chebyshev_baseline.py` 和 `trainable_w.py` 的 hash。审计中发现并修订了旧措辞：U(3) 已足够支撑数学谱界，但 CPU helper 只验 `U†U≈I`；此外 `y=1` 是谱上端点，不保证为某个实际 gauge operator 的本征值。

同一更正已同步至 `research_contract_v1.json`、`contract.py`、candidate adapter 及 [约束谱基报告 v4](../../results/optimize_gaussian/chebyshev_constrained_baseline_20261125_v4/README.md)；[contract diagnostic v3](../../results/optimize_gaussian/contract_diagnostic_20261125_v3/README.md) 和 [candidate contract v6](../../results/optimize_gaussian/framework_candidate_contract_20261125_v6/README.md) 也由当前源重建。强谱方法合同现把零模因子写为条件项。所有都仍是 CPU/结构证据，不是物理比较。

1. [`trainable_w_stress_20261123_v4`](../../results/optimize_gaussian/trainable_w_stress_20261123_v4/README.md)含 paired heldout-loss 图、预算收敛图、conditioning/baseline 图和完整失败分母：36/36 synthetic panels、288/288 solver runs 返回，93 次到达预算上限且 solver 报 non-success，未排除样本。local-environment 的 6 个 cell 中 joint-W 的 cell median 都优于固定 width-2，但 0/6 优于等参数数目的固定 width-6；spectral-heat 的 paired cell medians 接近 1。样本只是每格 3 个 synthetic panels，不能推断总体或物理排序，也没有使用 D04 risk。
2. [`trainable_w_meta_init_20261123_v5`](../../results/optimize_gaussian/trainable_w_meta_init_20261123_v5/README.md)含训练/评估数据流图与成本诊断：12 个训练 panel、12 个独立评估 panel，108/108 solver runs 返回。学习得到的 warm start 仍与 SLSQP 联用；在该窄 synthetic 分布，10%/1% 初始拟合损失目标的 online calls 均值为 3.58/4.25，对 cold SLSQP 的 4.58/6.92；将 192 panel-equivalent 训练/诊断成本计入后，算术 break-even 是 K=193/73。它不是 learned-smear 优于物理 SLSQP 或强谱基的证据，也没有可比生产 walltime。
3. [`chebyshev_constrained_baseline_20261125_v4`](../../results/optimize_gaussian/chebyshev_constrained_baseline_20261125_v4/README.md)给出约束推导、响应图、dense/matrix-free 路径和 bounded-call SLSQP helper；只做 illustrative controls，objective fit 分母为 `0/0/0/0`，尚未接 `train.main` 或生产 evaluator。[`framework_candidate_contract_20261125_v6`](../../results/optimize_gaussian/framework_candidate_contract_20261125_v6/README.md)只是 schema flow，物理五阶段均 abstain（`5/0/0/5`）。旧的 unconstrained Chebyshev 三面板 positivity 图不证明新约束拟合在 production operator 上的数值稳定性。

**下一位 agent 的具体工作：**先按实际 `kernels.py` 与生产字段逐项复核 `H` 的维度/shift/边界、link unitarity、source 与 `a_t mπ` 约定，并由独立 reviewer 签署 D04/D11 合同；不一致则发新版本合同并保留旧快照。之后把合同接入真实 `RunSpec` / `train.main`，生成 hash-bound run plan 和 fail-closed physical `RunState`，交付图或文字数据流及 planned/valid/failed/abstained 分母。等 E-chain reference receipt 和数据授权到位后，先跑固定预算 CPU dry-run，再申请独立 GPU/QUDA smoke，最终才可做同配对的两项比较与盲测；报告须同时给风险区间、calls/time best-so-far、分项 walltime、residual/KKT、失败样本和成本摊销。任何一步未满足时维持 `NOT_READY/UNCERTIFIED`、`physical_score=null`，不读 gap/validation/sealed IDs。

### 2026-11-22 历史交接：F=2 与 degree-4 候选（下文早期 D04/D06 未定表述已被上方 2026-11-25 提案替代）

### 2026-11-21 完整性复核：保留 TODO

**历史边界确认（2026-11-21，已由上方2026-11-22 handoff细化）：**用户要求维持 `NOT_READY/UNCERTIFIED` 与数据封存状态。已选定的确认性对照仍为 trainable-W vs 同预算 SLSQP、trainable-W vs 固定强谱基（F=2）；D04 reference、D06 `r_star` 与生产阈值不代填。当前可完成的D11与C1/C2接口/文档一致性审计已完成；在研究值、物理reference与预算获审阅前，无需继续扩CPU toys或接入生产入口。不运行GPU/QUDA、不读gap/validation/sealed IDs、不生成物理score。

**实现状态：CPU parameterization prototype completed / production integration and physics comparison still NOT_RUN。**
隔离的 [`trainable_w.py`](trainable_w.py) 已实现共享实W/v/c、空间中心化与现有Hermitian hopping kernel的解析梯度训练；
[图文诊断](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/tasks/optimize_gaussian/trainable_w_cpu_prototype_20261121/README.md)保留独立synthetic teacher、4 fit/3 heldout links、逐例误差/完整trace与失败分母。
新增测试 **9 passed**，相邻feature/kernel聚焦 **7 passed, 48 deselected**；覆盖梯度、实际W更新、gauge与48个含反向link-anchor处理的cubic/parity变换。
它仅是coefficient-fit原型，不是gauge-ensemble physics risk、不比较SLSQP，也不改变生产train.py或研究者TBD；TODO仍须保留。

**2026-11-21 历史交接（已由上方 2026-11-22 状态更新）：**D04 reference 与 D06 最小效应的可审阅备选表、D11 degree-4 candidate 和共享objective/预算流程现已列在[研究决策单](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/tasks/optimize_gaussian/protocol_decision_review_20261121.md)与[`protocol_freeze_draft.md` §7](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/tasks/optimize_gaussian/protocol_freeze_draft.md)。不重复起草；研究值仍未冻结，后续 agent 只按上方指令检查其与当前CPU算符接口的一致性。

当前核心问题仍未完成：真实生产数据上的 trainable-$W$ / learned initializer 与 Gaussian、同预算 SLSQP、强谱基的盲测尚未运行，P3 生产 2×2 也未运行，因此本文件不能删除。真实 D 链 point-propagator 输入、trajectory identity 和 residual 已由 adapter 100/100 核验；这只关闭输入集成，不等于重新采样模型 metadata、训练 learned $W$ 或完成生产 score。关于包 B 覆盖事故的报告已对照[`cpu_next_package_b/README.md`](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/cpu_next_package_b/README.md)：`unified_v1/` 曾被同名输出覆盖，旧 fold 未纳入 Git 且该目录无法从当前仓库恢复；这是仓库内可恢复性的结论，不排除外部备份；不要重复本地恢复搜索。旧比值 1.719/2.014/0.810 仍作为三折描述值保留在[任务README](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/tasks/optimize_gaussian/README.md)与[包B报告](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/cpu_next_package_b/README.md)，其 reference gate 三折均失败且 fold 间不构成独立误差估计；它们只是历史说明，不是可复现的新 artifact 或当前方法排名。**历史全仓 CPU suite** 为 **274 passed, 1 skipped, 1 warning**（2026-11-20，`PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q -p no:cacheprovider -rs`）；同一历史快照的 Optimize 主测试文件为 **53 passed**、solver contract 为 **11 passed**。这不是当前源码下的全仓复跑。本轮已有较新的 Optimize 六模块定向 CPU suite **113 passed**（含 solver contract 当前 **13 passed**），以及 worker receipt focused **24 passed**、standalone hard-timeout **9 passed**；这些 pytest 主要使用隔离 CPU fixture，不证明 production 训练或 GPU/QUDA 中断安全。另见下方 synthetic `train.main` 子进程诊断，它只验证 CPU 控制流，不代表真实输入或物理比较。

**当前完整 CPU/GPU-free suite（2026-11-22）**：在 matrix-free Chebyshev/sandwich 更新后的工作树运行 `PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q -p no:cacheprovider -rs`，最终 **418 passed, 1 skipped, 1 warning**。跳过项是缺少 CuPy 的 stout parity test；warning 是 Wilson projection rejection 测试触发的 `RuntimeWarning`。这证明当前 CPU 路径的回归通过，不证明 QUDA/GPU 执行、reference validity 或物理结论。2026-11-21 的385 passed及本次增量前的413/415/417均为历史快照。

另一个已完成的 CPU 算符级桥接诊断见[报告与图](trainable_w_operator_cpu_20261121_final/README.md)：36/36 optimizer runs 均有收据，有限差分梯度误差最大约 `1.23e-11`，压缩与直接目标差最大约 `1.67e-16`；在该 heat-kernel 合成目标和三个小 panel 上，没有稳定的 trainable-$W$ 优势，degree-4 Chebyshev 谱基误差更低。此结果是条件性负诊断，不是生产排序；不要再扩相同目标/标签来替代冻结物理目标、reference 和确认性比较族。

复核结论：包 B 覆盖事故与 README 记录相符，但“当前 Git 无法恢复”只限定于仓库内，不排除外部备份。剩余事项也不能笼统写成“都需要 GPU”：RunSpec 与 `train.optimize` 的 solver gate CPU 接线已有独立测试闭环；待冻结的是研究者决策阈值、估计目标、成本边界及统计规则，并需将训练产物 receipt 带入真实 RunSpec/worker 审查。真实训练、metadata 复采、生产 2×2 与未观察 blind comparison需要相应生产输入、GPU 和 split 授权。任何 CPU 完成项都不替代真实 learned-$W$ 是否优于 SLSQP / 强谱基这一主问题。

**本次裁决与下一工作边界：**“所有未做项均需 GPU / 授权”不成立：研究决策选项和若干CPU接口检查已在封存下完成，但研究者要求保持 `NOT_READY/UNCERTIFIED`，因此不把TBD写入生产协议。C2/C3样本量警示属于Pion的不同模拟设计，不应用来汇总Optimize的C2/C3或物理证据。D04/D06仍是物理研究决策；D11候选与C1/C2接口一致性审计已完成。下一步要等物理reference、`r_star`、Chebyshev算符/约束和每臂预算获审阅后，才能接入真实 objective / RunSpec / paired production comparison。不要重复synthetic扩样、重建包B旧fold，或在reference gate失败时产生物理分数。若之后获批运行，诊断须含图或可查文字流程、planned/valid/failed/abstained分母及适用边界。

**后续生产阶段的历史开工顺序（当前不执行）：** task-specific [`runspec_builder.py`](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/tasks/optimize_gaussian/runspec_builder.py) 的输入绑定、CLI no-launch 和 CPU preflight 已完成；builder→worker 有假 child artifact fixture，也新增了真实 `train.main` 在 worker 子进程中的 synthetic CPU integration（见[图文报告](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/cpu_train_main_supervisor_v2/README.md)）。可审阅的 [`protocol_freeze_draft.md`](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/tasks/optimize_gaussian/protocol_freeze_draft.md) 把 split/reference 边界与研究者需决定的最小效果、双预算、KKT 和统计门分开；其中的 `TBD` 按用户当前决定继续保持，不在本轮逐项冻结。待这些研究决策后续获审阅、reference可用且单独授权时，才用builder生成no-clobber RunSpec并运行训练/比较。届时所有诊断须有文字说明、身份/状态与适用边界；软件接线或preflight smoke报告数据流/receipt/trace，不伪造paired-risk或物理成本；实际方法对照提供paired CI、calls/time收敛、setup/training/inference/solve/total成本及KKT/residual、超时、失败/弃权分母。

### 2026-11-21：trainable-W 算符目标桥接与强基线负诊断（CPU 完成）

[图文、完整数值和失败分母](trainable_w_operator_cpu_20261121_final/README.md)及[`trainable_w_operator_diagnostic.py`](trainable_w_operator_diagnostic.py)已将目标从同模型族 coefficient labels 推进到完整算符作用量：新合成 SU(3) links 的 trace-normalized heat kernel，周期空间 L=3、完整 color/site identity probes，无随机 probe 误差。它和候选 K 共享空间/边界/trace=dim 约定，但不冻结生产 normalization。3 个独立 panel 各用 3 fit / 3 heldout links；每个 panel 运行 fixed-W width2（3 个拟合标量）、joint-W width2（7 个）与 fixed-W width6（7 个）× SLSQP/L-BFGS-B × 40/160 objective+gradient call cap。后两臂提供相同拟合参数数目的敏感性对照；它们函数族和初始化不同，不能称完全相同 capacity。SLSQP/L-BFGS-B 只作为每一参数化内部的求解器对照。完整36/36求解返回、4个预算截断/solver non-success、无样本排除；独立 fit/heldout links 的 gradient FD 最大差 `1.23e-11`，压缩二次目标与直接矩阵最大差 `1.67e-16`。原有模型测试 **9 passed**。

160-call 时 joint-W heldout relative squared Frobenius error 约 `0.006005–0.006146`，与 fixed width2 的差仅约 `1e-7` 绝对量级或更小。等7参数 fixed width6 在1个 panel略好、2个略差，未得到稳定 W 学习优势。degree4 Chebyshev 同 fit-target 的直接最小二乘误差为 `3.34e-8–2.40e-7`，degree1约 `0.0060`。该目标本来就是谱函数，谱基参数域、identity coefficient 和直接解成本也不同；结果支持这个目标下“扩大算符作用范围比微调单跳系数更有作用”的结构诊断，不能作为普适方法排名、物理 risk 或成本优势。图展示 heldout error、拟合 calls 曲线和等参数配对；原始JSON保存全部trace/参数/KKT连续诊断/失败状态，NPZ和protocol/source snapshot保存实际输入与源码身份。未运行GPU或生产数据，`physical_score=null`。

**接手决策：**不要继续用相同 heat-kernel 或同模型 coefficient labels 扩样争取学习优势。这两个实现门已经完成。若继续真实 learned smear，先冻结物理目标/reference、kernel归一化/正定性与强基线，选择多跳/ Gaussian sandwich/谱+局域修正中的有物理依据的算符族，再按下文生产决策单做训练接线及配对GPU开发。SLSQP和L-BFGS-B在每个模型内比较；模型之间另报告自由参数数、算符范围、setup/training/inference与solve总成本。当前CPU负诊断不解决失败的Pion reference，也不授权读取shared holdout。

### 2026-11-21：真实 `train.main` 在 worker 下的 synthetic CPU integration

[`train_main_supervisor_diagnostic.py`](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/tasks/optimize_gaussian/train_main_supervisor_diagnostic.py) 已用一个隔离 worker 子进程实际调用 `train.main`，而不是仅在 builder 生成后用假 child 替换命令。保留真实 SciPy Powell、`train.optimize`、solver contract、protocol hash recheck、RunSpec worker supervision 和 parent artifact receipt；只把 collect/evaluate、risk observable 与 coefficient map 换为固定 synthetic fixture。结果见[报告和 best-so-far 图](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/cpu_train_main_supervisor_v2/README.md)：worker=`completed`，训练产物 4/4 hash verified，3/3 synthetic inputs identity-verified；Powell success，32 objective calls，KKT diagnostic `computed`，残差约 $2.86\times10^{-19}$。没有初始化 GPU/QUDA，也没有读取 archived/production 数据；协议阈值只供 fixture 走通，全部标 `diagnostic_only` / `physical_score=null`。这关闭的是训练入口—外层 worker—产物 receipt 的合成 CPU 接线缺口，不是 production RunSpec、GPU hard-timeout、learned-$W$ 优势或物理 PASS。

### 2026-11-21 CPU preflight：训练入口支持 hash-bound protocol，生产 RunSpec 仍未闭环

`train.main` 现在接受成对参数 `--protocol PATH --protocol-sha256 SHA256`；子进程读取前重新计算 protocol 原始字节 hash，再解析 JSON。漏传任一参数或 hash 漂移均在准备输出目录、读取测量输入和 GPU 初始化前拒绝。文本数据流为：RunSpec 文件 hash 校验 → child 显式 protocol 路径/hash 再校验 → 新输出路径保护 → 输入/receipt 检查 → 才可能进入训练。新增 `tests/test_optimize_training_protocol.py` 与原 Optimize 主测试合计 **59 passed**；其中 6 项新用例验证正确 hash、错误 hash、参数配对和非 object JSON 拒绝。没有启动训练、读取生产缓存或初始化 GPU。

这关闭的是 protocol 传递/child recheck 的 CPU 接口缺口，不自动形成可运行的生产 RunSpec。worker RunSpec 现可在 `outputs.artifacts` 声明 `results.json`、`manifest.json` 等路径；这些产物可位于训练进程新建的输出目录下，预检只要求最近已有目录祖先存在并拒绝任何已存在目标。只有进程组完全退出且子进程正常结束后，父回执才逐项写入 SHA-256，失败/超时/残留子进程则标 `not_accepted` 且不计算产物 hash，声明产物缺失时为 `incomplete_or_unreadable`。文本数据流是：RunSpec/input hash preflight → frozen protocol child recheck → supervised child → 清理残留进程 → 成功产物逐文件 hash → parent receipt。`tests/test_optimize_hard_timeout.py tests/test_optimize_worker_runspec.py` 本轮 **24 passed**；7 个新增场景覆盖成功 hash 回链（含新建的 training output 目录）、失败产物拒收、成功但缺少产物、预存产物 no-clobber、leader 成功退出但留下子进程时拒绝其产物，以及 artifact 完整度对应的 CLI 退出码。没有启动训练、读取生产缓存或初始化 GPU。solver gate、预算阈值、可中断 GPU/QUDA 语义和物理 reference 仍须分别闭合。

任务专用 [`runspec_builder.py`](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/tasks/optimize_gaussian/runspec_builder.py) 从显式 protocol 生成 worker schema v2 spec，推导互斥的 `fit/dev_val/test` 配置ID，绑定 manifests、逐配置 gauge/propagator/sidecar、reference/bootstrap、可选 Pion measurement receipt，以及 Optimize/统计/监督器源码 hash；child argv 绑定 protocol 原始字节 SHA，产物清单覆盖 `results.json`、`manifest.json`、`data.npz` 和各 seed 的 `basis_s*.npz`。新增 `tests/test_optimize_runspec_builder.py::test_built_runspec_crosses_supervisor_boundary_and_hashes_cpu_fixture_outputs`：确认 builder 生成的命令目标为 `tasks.optimize_gaussian.train`、协议路径/hash 成对绑定；随后仅在隔离 fixture 中用假 worker 替换命令，经过真实 `worker_runspec.run_frozen_worker` 完成输入 hash preflight、子进程监督和全部声明产物的 parent hash receipt。builder 测试 **11 passed**；连同 worker RunSpec 测试 **26 passed**。完整的文字数据流与限制见 [CPU preflight 报告](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/task_runspec_preflight_cpu_v1/README.md)。这个测试没有执行 `train.main`，没有读生产缓存或初始化 GPU；当前 v9 protocol 不可复用作 blind，Pion reference 失败，真实 D 输入与允许运行的阈值/预算方案也未冻结。后续必须用新审阅 protocol 和实际生产输入生成一份新的 RunSpec，并在任何 GPU 启动之前完成 review/authorization；不得把 CPU orchestration fixture 当作训练或授权证据。

本轮定向跨任务复跑（Optimize builder / worker / training protocol / 主测试，Pion 主测试，Wilson force controls）**144 passed in 31.95 s**；该集合验证了本轮相关路径，不等同全仓 CPU suite。

本轮发现并关闭了同类的**代码级覆盖入口**：`unified.py` 原先固定写入已存在的包 B 目录，`train.py` 默认也会写入历史 v9 目录。现在两者都要求新输出路径；`train.py --resume` 只接受 basis 缓存目录，且缓存仍需通过既有 protocol/input identity 检查，包含最终结果或其他文件时一律拒绝。相应回归测试验证拒绝发生在数据读取和 GPU 初始化之前。历史产物未改写。

P3 的 CPU 前置设计与接口诊断已补充到 [`doc/optimize_p3_design.md`](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/doc/optimize_p3_design.md)。在下方“nested solver contract”新增接线之前，generic SearchAdapter 只消费 scalar risk，有限差分 KKT 也仅为 diagnostic-only 观察；两者的历史状态不能当作当前 solver gate。CPU solver gate 后续已按显式 protocol 接入 `train.optimize`，但 P3 方案仍未冻结，改善阈值、样本数/功效、multiplicity、abstain 与成本限制都待研究者确定。当前接线证据与边界见新增 CPU 接线报告，不等于 production 收敛/物理门。

### 2026-11-20 Powell 只记录 receipt：CPU 通过，生产 gate 未实现

`train.optimize()` 在保留原 `risk/theta/nfit`、loss、Powell 选项和 dev-val 选优的基础上，新增 `optimizer_receipt`，
在新运行的每个 grid 点与 best 中保存原 success/status/message、nfev/nit、自计 objective calls、elapsed、
精确边界命中/越界索引及非有限状态。字段表和文字流程见 [`P3 设计 §9`](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/doc/optimize_p3_design.md)。
新增四个 focused cases **4 passed, 36 deselected**，仅用固定微型输入、受控 optimizer 状态和 pytest 临时目录；没有 GPU、
生产数据读取、synthetic 扩样或历史结果修改。该实现是 `diagnostic_only` 的可观测性增强，不拒绝 optimizer 的有限失败结果，
也不实施 KKT/residual/逐臂ID/walltime 门槛。当前 main 为 optimizer/loss 异常增加下述失败收据，但仍原样传播异常。

receipt 现另含 `objective_trace`：每次正常完成 objective 调用的 index、elapsed、原 loss value/finite 与 finite-only
best-so-far（首个有限值前 null；非有限原值以明确字符串编码，但返回给 Powell 的 loss 不变）。新 run 可以据此画 calls/time
两套 best-so-far 图；历史 run 没有 trace，只有之后授权的新目录重跑才能得到新轨迹，禁止反推或回填旧记录。
trace/异常 focused tests **7 passed, 36 deselected**；包含 mixed NaN/±Inf/finite、单调best-so-far、原返回保持与
`scipy.minimize`/loss 异常原样传播。本轮另复跑完整 task 文件 `tests/test_optimize_gaussian.py`，**47 passed**；
连同 Pion / Wilson 任务的全仓 suite 为 **222 passed, 1 skipped, 1 warning**。
正常 receipt 记录 optimizer 返回；当前新增的 failure receipt 另记录异常，不返回 partial fit。

### 2026-11-20 Powell 盒约束 KKT 诊断（CPU，不是 gate）

每个正常 optimizer 返回后，receipt 现在独立记录 `box_kkt_diagnostic`：对 `risk + L2` 在最终theta处做相对步长
`1e-5*max(1,|theta_i|)` 的有限差分；内部用central、靠近下/上界用forward/backward；保存活动索引、梯度、原单位与缩放后的KKT violation、
探针调用数/时间。探针调用不记入Powell trace，避免best-so-far被诊断点改变；不影响 risk/theta/nfit、选择或结果门。
非法点、探针异常、非有限值和退化区间均有状态，`threshold=null`，不会因诊断失败拒绝候选。因为风险函数可能非光滑且有限差分依赖步长，
它不是严格KKT证明，也不是QUDA residual；production gate、solver停止/弃权阈值、双预算hard stop仍未实现。

定向测试覆盖内部二次目标、下/上界KKT符号、违反边界条件、非有限和异常探针；`tests/test_optimize_gaussian.py` 全文件 **53 passed**。
这些是CPU接口/受控数学验证；没有真实GPU、生产训练、walltime benchmark或物理score。

### 2026-11-20 optimizer/loss 异常的原子失败收据

main 在经输出保护的新/仅 basis 未完成目录内，为每次 optimize 绑定协议/代码 hash、seed/lam2 和此前完成 grid。
minimize/loss 抛异常时将已完成 objective_trace、进入/完成计数、耗时、异常类型/消息与上述身份原子写入
`optimizer_failure.json`（`diagnostic_only`，physical_score=null），再重抛**原异常对象**；不返回 partial fit、不继续选优或test。
同目录临时JSON完成并fsync后用no-clobber hard-link发布，已有failure文件和已完成结果目录均拒绝覆盖。
写盘失败会给原异常附加原因后继续抛原异常，临时文件清理；IO失败时不能保证有收据。直接调用 optimize 而未提供run context不写收据。
覆盖仅为 minimize/loss 的 Exception；输入/collect/GPU/dev-val/test/进程中断不在本项覆盖内，不能称为全部失败覆盖。
字段/状态表见 [`P3 设计 §9`](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/doc/optimize_p3_design.md)。历史结果未修改，新执行必须重新绑定当前源码hash。
失败/正常receipt相关定向测试 **11 passed, 36 deselected**；与原预算/配对/输入残差/cache/manifest 检查合并为
**24 passed, 23 deselected**。全部使用固定微型输入、模拟optimizer与pytest临时目录，无GPU、生产/holdout读取或benchmark扩样。

下一位 agent 先按 P3 待冻结参数表与研究者确定收敛/截断/弃权与双预算口径，再把已接入的 nested contract receipt 纳入真实 RunSpec/parent receipt 与结果审查；
不得从 `success=True` 或字段齐全推断物理通过。train.py 改动后旧 adapter/provenance 的冻结 source hash 仅代表历史代码；
新执行须绑定当前 hash，不能回写旧收据来制造当前版本的集成证据。真实 trainable-W/initializer、P3 生产 2×2 和盲测仍为 `NOT_RUN`，TODO 保留。

下一位 agent 先完成真实测量前的 protocol / feature / objective / 双预算 / source-sink / residual 门槛冻结，并生成绑定输入、代码、环境与设备的 RunSpec 和 receipt；为 D 链 0–99 复用已核验 point-propagator 时先冻结 Optimize RunSpec，再执行 P3 2×2 和真实学习型 smear 对照；不得重做相同 inversion。交付必须包括 paired-risk 区间图、best-so-far 收敛图、walltime 与训练摊销图、KKT/residual 和失败表，以及解释样本、预算截断和适用边界的文字报告。现有 CPU synthetic 路线已有诊断，不得只更换 seed 重复扩样来替代物理比较。

### 2026-11-20 receipt adapter 的真实输入集成通过

新 adapter 已在远端把真实 Pion NPZ/provenance 与实际 D-chain gauge/propagator cache 一起送入 `train.validate_inputs`：[报告](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/d_chain_propagator_adapter_20261120/README.md)和[汇总收据](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/d_chain_propagator_adapter_20261120/adapter_check.json)显示 **100/100 verified、0 failure、100 个非空 trajectory ID**；最大 residual `9.99751e-11`。`adapter_check.json` 不保存逐 ID 成功明细，其 `trajectory_join_count` 仅统计非空 trajectory 字段；实际 trajectory 与 HMC manifest 的等值 join 由独立的[propagator reuse audit](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/d_chain_propagator_reuse_audit_20261120/README.md)核验。两份证据合并支持 D0–99 cache 身份与残差可复用；不能把 HMC crosswalk 归功于 adapter 单项。显式绑定的 receipt 对缺配置、缺 residual/trajectory 字段 fail closed；相关聚焦单测 **2 passed**。本验证不包含模型训练、P3 2×2、物理 score 或盲测。

### 2026-11-20 D 链 propagator 复用审计：输入可用，score 仍被 reference 阻断

Pion D development 已产生 IDs 0–99 的 point-source propagator 缓存。只读复用审计[报告](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/d_chain_propagator_reuse_audit_20261120/README.md)核验 100/100 `.npy` 字节 SHA、100/100 gauge SHA 与 HMC manifest、trajectory join、`complex128`、geometry `16^4`、source `[0,0,0,0]`、boundary `-1`、`kappa=0.14`、tolerance `1e-10`、maxiter `10000`；最大记录 residual 为 `9.99751e-11`。Optimize 的 fit IDs 0–59 / development IDs 60–99 可避免重复 point inversion；新 RunSpec 仍须绑定这些 cache hashes 与 Pion generation/provenance。cache JSON 未写 residual 字段；`train.validate_inputs` 现支持由显式协议绑定 Pion NPZ / provenance 的 SHA、generation 和 protocol hash，并从 provenance 取每配置 residual。聚焦回归为 **2 passed**，真实 D0–99 adapter 集成为 **100/100 PASS**。这还不是 Optimize-specific RunSpec、训练 receipt 或得分，未改写 cache 或 Pion 结果。

**当前不可做物理 score**：Pion D development single-cosh reference gate FAIL，后续两态诊断也发现晚时间窗 excited mass 命中上界。Optimize 若继续用旧 `m0` 或该失败 reference，风险数字只能标为 `diagnostic_only`，不得作为 learned-smear 优势。先定义可验证的新 reference protocol，再冻结 Optimize-specific RunSpec、feature、objective、score、双预算和代码 hash；不访问任何 blind ID。D IDs 100–139 为 embargo/gap，140–199 仍是三任务共享封存区。

### 2026-11-20 本地续查：生产对照需要新的执行环境与 blind IDs

当前桌面无 CuPy、PyQUDA、mpi4py、torch 或本地 raw gauge 副本；Kunshan DCU 已成功完成两条新 HMC 链。2026-11-20 live audit 核验旧500个 NERSC 文件与各自 manifest 500/500匹配，且 `S16T16`/B/C 重叠 payload逐ID相同，B/C不能作为独立blind ensemble。新 `S16T16D` 数据在远端有200个manifest绑定配置；HMC receipt/逐轨迹和200/200 payload hash诊断保存在[主链报告](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/wilson_loop/hmc_production_20261120/diagnostic/README.md)。原始 gauge bytes 未复制到桌面；Pion D0–99 point propagator cache 已在远端生成并核验 100/100，Optimize receipt adapter 对实际100个ID集成 **PASS**，因此这些 fit/development ID 不需重做 inversion；独立 Optimize RunSpec / optimizer receipt 仍未生成。v9旧90个 gauge hashes可核验，但propagator原始字节和运行/设备receipt仍缺。P3 2×2、trainable-$W$ 与真实 held-out比较尚未运行；无需再找已存在的 gauge bytes，也不要重跑同一 synthetic generator。

审计基线：2026-11-20 当前工作树；此前全套 CPU / GPU-free 测试为161 passed；输出保护改动后复跑 `python3 -B -m pytest -q -rs` 得 **170 passed, 1 skipped, 1 warning**。唯一跳过项是本机缺少 CuPy 的 stout parity test；warning 是已知 Wilson projection rejection `RuntimeWarning`。CPU 包 A–D 与 C 收敛审计已有实现和诊断；本文件保留，
因为关键研究问题——**学习型 smear 是否能在独立物理数据上超过 SLSQP 与更强算子基线**——仍未回答。

### 2026-11-20 新 HMC 链完成，Optimize 生产比较尚未运行

Kunshan job **128526595** 的 `S16T16D` 链已完成1396条轨迹、400条热化、996条生产，200个保存配置的payload hash 200/200验证；图文质量诊断见[报告](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/wilson_loop/hmc_production_20261120/diagnostic/README.md)。冻结的 Optimize split 为train0–59、development60–99、embargo100–139、blind140–199。IDs140–199同时预留给Pion validation、Optimize blind test与可能的Wilson比较；shared holdout尚未打开。Wilson已冻结 [D 链 RunSpec](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/tasks/wilson_loop/protocol_d_20261120.json)，只授权D0–99和D120–139且明确封存140–199。独立 `S16T16E` chain (job128528670) 是Pion anchor来源，不自动成为Optimize训练集。Pion D0–99 propagators 已由 Optimize adapter 对实际缓存集成 **100/100 验证**，但 Optimize 专属 RunSpec 和训练 receipt 尚未生成，trainable-$W$、生产 P3 2×2与 blind comparison 均为 `NOT_RUN`。

当前研究问题仍是学习型 smear 能否优于 SLSQP/强谱基；所有 production protocols、feature/parameterization、score与预算须先冻结。Pion、Optimize、Wilson共同消费 D IDs140–199时，三个分析必须在首次解盲前都锁定，后续结果在任务间相关，不能当作独立复现。输出paired-risk、KKT/residual、收敛/成本图和失败文字；任何真实 GPU测量都必须等生产工作解锁后再启动。

## 1. 当前证据和不能越界的结论

### A：可信输入与 reference 身份

`train.validate_inputs` 会检查 gauge / propagator 内容摘要、geometry、boundary、tolerance、iteration、precision 与
inversion residual。缺 metadata 降级为 `archived_diagnostic/unknown`，可信入口 residual 超界或输入不符时拒绝；
`reference_identity` 绑定实际消费的 bootstrap/reference 文件。历史 GPU environment 未知时不会拿当前机器信息回填。

### B：point / replica 同估计量和回归

- `risk_with_fixed_amplitude` 分开 training-frozen 与 validation-refit；fold artifact 保存 train/test block IDs、
  weights/scales/mass identity、search 预算与失败。
- [`cpu_close_20261119_B_regression_fitted`](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/cpu_close_20261119_B_regression_fitted/README.md)
  是 60 datasets × 200 draws 的真实分块 synthetic regression。point−replica 偏差均与 0 相容；training-frozen
  四时间点同时覆盖 60/60，验证 refit 53/60，二项 CP 区间分别 [0.940,1] 与 [0.774,0.952]。
- 这是 toy conditional interval，训练幅度不确定度没有传播到验证区间；不可将其称为真实数据 calibration。
  `cpu_next_package_b/regression/` 仅是 5×20 smoke。测试已改为写 pytest 临时目录，防止小测试再次覆盖归档数字。

### C：公平 operator-class 表达能力测试

[`cpu_close_20261119_C_matched`](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/cpu_close_20261119_C_matched/README.md) 使用全空间
色荷 covariant Laplacian $H$、Hermitian clover $B$、匹配的 point/color probes 和 train/test 独立 links；系数是同一
实数域 $[-5,5]$。$Q=H^2+\epsilon BHB$，类 B 又显式含 $BHB$，故 B oracle 在 $\epsilon>0$ 的零误差是
**构造性的可表示性**，不是模型从数据学到了 $Q[U]$，也不是物理学习优势。

三 seed mean：类 A oracle error 从 $\epsilon=0$ 的 $\sim4\times10^{-16}$ 增至 $\epsilon=0.1$ 的
 $3.27\times10^{-4}$；类 B oracle 约 $10^{-15}$。实际 SLSQP/Powell 在有限 probes / budget 下未达 oracle，
类 B Powell 约 $3.7\times10^{-3}$–$4.7\times10^{-3}$；`success=True` 不代表找到全局最优。图中误差线是三 seed
散布，远不足以作为发表级泛化区间。

![matched operator-class 误差与 seed 散布](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/cpu_close_20261119_C_matched/fig_crosslink_fair.png)

### D：BCB—response—SearchAdapter

`adapt_v2.py` 的评价器直接调用真实 BCB kernel，SearchAdapter 与 OperatorContract / MeasurementBundle 已接通；
search 与 exact response 差为 0 是同一个函数被调用两次的构造结果，不是 surrogate 精度。见
[`cpu_close_20261119_D`](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/cpu_close_20261119_D/README.md)。

### 已有 v9 GPU 重训和一次 held-out 评估

v9 用修正后的 link-centred A1+ 特征做过 GPU 重训：3 个 $W$ seed、$\lambda_2\in\{0,0.01\}$，并报告 test IDs 70–99 的结果。这里的 $W$ 是固定随机特征基、**没有训练**；拟合的是其系数 $\theta$。因此 v9 证明 GPU feature/系数重训与一轮 held-out evaluation 已执行，不等于用户关心的 trainable-$W$ 学习型 smear。test 风险相对 Gaussian base 为 2.12×；候选与 7 参数 SLSQP 的 $R_{SLSQP}/R_{learned}=0.998$，95% CI [0.78,1.18]，只是“未检出差异”，不是等效证明。reference gate 未通过，official candidate score 仍为 null。

v9 的 test IDs 已经评估并公开，不再是可以重复使用的 blind set。manifest 保存 protocol、feature、$W/\theta$、IDs、propagator cache 和 dependency hashes；引用的 provenance 还列出 PyQUDA/MPI 软件版本，但当前材料没有完整训练命令/设备型号与 driver receipt。Kunshan live byte audit 已把其 gauge 输入身份从“本地缺 raw 文件、尚不能验证”更新为“远端 90/90 raw-file hashes 匹配”；原始 propagator bytes 仍未恢复。后续应补齐可追溯的生产 receipt，并在新独立链或严格冻结且未参与调参的 held-out block 上评估，不要重跑或重命名 v9 test。历史 `unified_v1/` 在包 B 开发中曾被同名输出覆盖；
该 fold 目录未纳入 git，仓库内没有可恢复的旧文件。旧点值留档但旧 fold 内容不再有原身份，事故记录见
[`cpu_next_package_b/README.md`](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/cpu_next_package_b/README.md)。此结论限于当前仓库，不代表检查过外部备份。

补充归档核验：`results/kunshan_logs/pipe_127946355.log` / `pipe_127946906.log` 是较早的 `s16t16-pilot-v1` signed-mixture GPU 测量记录，不是 trainable-$W$ 训练。90/90 配置行与 pion `s16t16_v1` development/validation provenance 完全相同；v9 manifest 明确复用同一 validation provenance，test IDs 仍为70–99。因此这些日志确认了历史测量 receipt，但不能补成新 blind comparison，也不能把已观察的30个 test IDs再用作独立检验。逐配置哈希、残差/耗时图和 crosswalk 见[Kunshan archive audit](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/kunshan_logs/archive_audit_v1/README.md)。

**GEVP200 不能补成独立 blind ensemble（2026-11-20，live byte audit 已解开 hash 差异）**：旧 crosswalk 中 GEVP200 的 180 条记录与 snapshot hash 字段相同、与主 manifest 同 ID hash 字段不同；新审计发现两套字段各自都与远端 raw-file bytes 对上（GEVP200/C 为 180/180，旧 pilot/S16T16 为 90/90），而 S16T16、B、C 的重叠 gauge payload bytes 逐 ID 完全相同。全量矩阵与图见 [live archive audit](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/kunshan_logs/live_archive_audit_v1/README.md) 和[身份图](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/kunshan_logs/live_archive_audit_v1/identity_and_duplicate_payloads.png)。因此这些 hash 差异来自 NERSC header，不是不同 gauge-link payload；GEVP200 validation 已公开分析且 reference gate 失败，不能用作新的独立物理验证或 fresh blind test。记录中的旧 `measure.py` 源码字节已从 Git 历史恢复；下一步不再索取已在 Kunshan 归档中的 gauge files，而是补 v9 propagator / run receipt，并冻结新独立链或未触碰的 blind block，完成 P3 2×2 和 trainable-$W$/initializer 对 Gaussian、同预算 SLSQP、强谱基的配对实验。

## 2. 下一步高优先级：trainable-$W$ 学习型 smear vs SLSQP / 强基线（需新的生产数据）

### 研究问题

是否存在一个从 gauge-local 特征训练出来的 smear / optimizer，在**新的独立 held-out gauge configurations** 上同时改善
物理 loss、收敛成功率或总成本？目前答案未知。学习可能帮助表达能力、参数初始化或预算有限时的优化；同一 feature family、
同一参数域下的精确 global optimum 不会因为用了 neural network 而更低，但有限样本泛化、局部最优和计算成本可能不同。

### 预注册对照

1. 固定 geometry、action、source/sink、精度、收敛残差和 stopping rule；train/dev/test gauge IDs 事先隔离，test 不回看。
2. 固定确认性比较族F=2：C1=trainable-W vs同预算SLSQP（预固定参数化/多起点）；C2=trainable-W vs一个固定强谱基（建议degree4 Chebyshev，精确contract待审）。所有臂同一frozen objective/reference、paired IDs与双预算口径。Gaussian diffusion和fixed-W为secondary/mechanism；若扩充确认性主张，先扩大family并冻结错误控制。
3. 若声称“学习型 optimizer”优势，明确区分：学习 kernel weights、训练 gauge→kernel 的网络、学习初始化、或学习搜索策略；
   不能用固定随机 $W$ 的 $C_{fit}$ 回归代替真实网络训练。
4. 同一 gauge configurations、probe/source、validation target 和资源预算做配对；匹配数值精度、残差和成功判据。
   分列 setup/training、inference、solve、总 walltime、GPU memory、失败率与有效样本数。
5. 预声明主效应量、最小有意义改善、block/bootstrap 层级与 sample-size/power；F=2建议Bonferroni各one-sided (1−α/2)下界（family-wise至少1−α），α及r_star/样本量仍待定；描述性未校正CI不能替代同时确认性界。paired CI 按独立 configurations，而非
   把 optimizer draws 当独立数据。
6. 消融 real / shuffled / symmetry-preserving shuffled features、W seed、hidden width、regularization、basis order；
   排名若被谱基或 Chebyshev baseline 解释掉，应如实作为反证。

现有 P3 的 class×optimizer 2×2 预注册见 [`doc/optimize_p3_design.md`](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/doc/optimize_p3_design.md)：七幂类 / 嵌套 $Q$ 类 × SLSQP / Powell。它检验算符类收益能否与求解器效应区分，**本身不等价于 learned $W$ vs SLSQP 的盲测**；真实学习型 smear 对照仍需单独保留 Gaussian、同预算多起点 SLSQP、学习得到的 $W$ 和强 Chebyshev/spectral 基线。

### 可证伪的学习实现路线

| 路线 | 可能带来的收益 | 必须对照的诊断 |
| --- | --- | --- |
| gauge-local invariant features → coefficient initializer → bounded projection → SLSQP polish | 复用相似 ensemble 配置，减少收敛所需目标调用 | 与零初值/随机多起点 SLSQP 同总 objective-call 和 walltime；画每配置残差轨迹与累计训练摊销成本 |
| learned restart / trust-region proposal selector | 固定总预算下更快找到低 risk 区域 | 由 evaluator 限制 proposal 次数；比较 best-so-far risk、成功率、失败 seed 和预算曲线 |
| symmetry-preserving learned operator basis / W | 扩大或重排 smear 函数族，可能改善跨配置迁移 | 与 Gaussian、同阶 Chebyshev/spectral、强手工协变基线配对；做 shuffled-feature 与 basis-order 消融 |

实现边界已在[冻结方案草稿 §1.1](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/tasks/optimize_gaussian/protocol_freeze_draft.md)按当前源码具体化：v9 的 $W$ 按 seed 随机生成后固定，Powell 只优化 $(c,v)$；它不能充当 trainable-$W$ 结果。新的[`trainable_w.py`](trainable_w.py)与[CPU原型报告](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/tasks/optimize_gaussian/trainable_w_cpu_prototype_20261121/README.md)已用实际 feature/kernel 接口验证共享实参数 $W,c,v$ 的梯度、更新、link-anchor signed-permutation 协变和 Hermiticity，但目标仅为同模型族的 synthetic coefficient fit。生产 `train.py` 仍未集成这条训练路径；下一步须在协议决策冻结后把特征/参数更新接入真实测量与候选评分，再和fixed-$W$机制对照、Gaussian、同预算 SLSQP及指定 Chebyshev/spectral 强基线比较。CPU原型的对称性/梯度检查只验证实现，不证明正定、真实 held-out gauge 泛化或物理优势；reference gate失败时不得报物理分数。

同一有限维凸二次目标和相同约束下，oracle 的全局训练最优值是下界；学习方法更可信的优势主张是有限预算收敛更快、跨
配置 amortized 总成本更低，或在明确扩大的函数族上改善 held-out 物理目标。所有路线都要交 paired test-risk / residual 图、
成本表和失败案例文字；不得把训练集更低目标或同一目标的 surrogate 误差为零当作学习优势。

### 交付与诊断

- 新目录 `results/optimize_gaussian/gpu_2x2_<id>/`，保存 protocol、seed、输入/代码/设备 hash、raw traces、训练状态、
  残差和失败样本；新 blind test 使用一次后即标记已观察。
- 图：每个独立 seed 的 paired test-risk 差及区间、收敛残差/迭代轨迹、总成本分解；另附 baseline × optimizer
  表格和失败文字说明。
- 开跑前的拒绝 / 回归检查：验证实际 $W$ 和 gauge/propagator provenance、geometry / boundary / precision / residual 与 source/sink 约定；断言 train/dev/test IDs 两两不交、blind test 不参与拟合或选参；逐臂校验输入 links、objective、参数域、收敛标准和固定预算一致。失败、NaN、预算截断与 solver flag 必须分开保存，不能只用 `success=True` 判收敛。
- 主图应同时给每配置 paired risk 差及 CI、best-so-far risk 对 objective calls 曲线、walltime 和训练摊销分解、残差 / KKT 与失败率；objective-call 预算与 walltime 预算分别比较，不把其中之一代替另一个。配一段文字解释样本排除、失败、弃权和边界。
- 只有真实 GPU RunSpec / receipts 与同预算 held-out 结果齐全后，才可讨论 learned smear 优势。

当前工作环境未安装 CuPy、mpi4py、PyQUDA / pyquda_utils、torch，且没有 `nvidia-smi`；本项没有真实 GPU 证据，标 `NOT_RUN`。

### v9 归档 provenance 复核（2026-11-20）

[v9 provenance audit](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/v9_provenance_audit_v1/README.md)已从本地 Git 历史补齐可恢复的源码身份：manifest 声明的四个训练依赖有 4/4 个 hash 可在当前源码或 Git 历史中精确找回；features/kernels/submission 与当前文件相同，旧 train.py 的精确字节来自 commit `ea83f9e`；statistics.py 的 `statistics_sha256` 与当前文件相同。此后不再重复搜索这些源码 hash。

本轮对 `train.py` 加入输出保护后，v1 的冻结 audit `verify` 因源文件 hash 改变而正确拒绝把当前工作树冒充冻结输入；v1 历史产物保持原样。新的 [v2 audit](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/v9_provenance_audit_v2/README.md) 在独立目录记录当前 source hash 并通过 `verify`：manifest 的4个依赖仍有4/4个能从当前源码或 Git 历史恢复，原 test IDs 仍与已观察的 Pion validation 重合30/30。

尚未解决的是真正的运行与输入来源：v9 manifest 没有 training environment、硬件、driver、package-lock 或命令 receipt；其 gauge IDs 70–99 的 raw-file hashes 已在 live archive audit 中与远端 S16T16 文件90/90匹配，但原始 propagator bytes 仍未恢复，原 manifest 也没有绑定其十个输出文件的 hash。fit/dev-val/test 划分互不重叠，但 test IDs 70–99 与已使用的 pion validation 30/30重合，不能称为 fresh blind test。v9 仍是固定随机 `W` 下拟合系数，不是 trainable-W 结果。下一步补齐运行 receipt 与 propagator 输入 provenance；若不可恢复，明确保留缺项，并按新协议采集独立数据，不要把当前 source hash closure 说成完整复现。

## 3. 后续 CPU 方法学工作

### 已完成：condition number / rank / KKT / multi-start / 固定预算曲线

[CPU 收敛诊断目录](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/cpu_close_20261119_C_convergence/README.md)包含逐 seed、逐起点和逐
budget 原始记录、condition/rank 图、objective/KKT/test-error 预算曲线和文字说明。A 类设计为 8/8 满秩，B 类 10/10
满秩；列缩放后条件数 $5.52\times10^5$–$6.45\times10^5$，属于满秩但明显病态。每个 seed/class/epsilon 使用相同的
5 个起点，目标函数调用上限从每起点 12 到 400，并按 seed 汇总标准误。

在当前有限合成设计中，SLSQP 的 5 个起点均于最大预算内成功终止，KKT 残差约 $10^{-9}$，实际总调用约 380–510；
Powell 的 120 个最大预算起点都耗尽 400 次/起点，残差约 $10^{-2}$。预算耗尽单独记为截断，不算优化器内部失败。
此结果支持“当前 C 合成目标上 SLSQP 是更强的数值基线”，不证明 SLSQP 对真实物理 smear 总体最优。

### 可选的 CPU / 统计扩展（不阻塞真实 W 主问题）

- B 的 stress 已扩展到三个质量，但固定一种相对噪声结构；C 原有 20 个新 independent links/probes datasets 的最坏 Bernoulli MC SE 约 0.112。现已另做 N=100 独立 follow-up（seed 1100–1199），目标为最坏 Bernoulli MC SE 0.05；这仍不是 CP95 半宽 0.05 或发表级物理校准。后续 CPU 新噪声结构或更大 N 必须先预声明精度/功效，另开结果目录，给 dataset-level 区间、误差条、失败样本文字和图；不能把 synthetic 结果当作物理泛化或真实 $W$ 训练。

### 已完成：B 时间相关噪声 / 错误 reference / block length / train-amplitude propagation

[Package B stress 结果](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/cpu_close_20261119_B_stress_v2/README.md)用 200 个独立 dataset/cell，
扫描 configuration AR(1) $\rho=0,0.5,0.8$、time AR(1) $\rho_t=0.5$、block length 2/5/10 和 reference mass 偏移 0/0.02/0.05；
训练与 validation 按原配置顺序切分，中间留两个 block embargo。每份 JSON 都有 dataset-level CP 区间和 q-profile/shape 检查图。
训练 amplitude 与 mass bootstrap draws 配对传播；错误 reference 下 true q profile 的 interval coverage 可仍高，但 q=1 shape
interval 全窗通过率变为 0/200，说明 coverage 与 reference 有效性需要不同判据。初版 v1 因随机交错相关 blocks 不合格，已在
自身 README 标为 superseded；保留产物供审计，不引用其点值。

- B stress 仍是单通道、固定相对噪声与质量已知的 synthetic 方法检验；若继续扩充噪声/相关结构，需先预注册 MC 精度，不覆盖 60×200 基线。此类扩展可在 CPU 完成，不需要 GPU；真实 $W$ 与 blind test 则仍需 GPU / 生产输入。

独立的 time-AR scan 现有 [400 datasets/cell 图文诊断](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/cpu_close_20261119_B_timear_v1/README.md)：固定真实质量 $m=0.5$ 和 configuration $\rho=0.5$，扫描 time $\rho=0,0.5,0.8$ 与 reference offset $0,0.03$。正确 reference 的四时点同时覆盖率仅 0.6125–0.655（逐 cell CP95 已列），错误 reference 全部为 0/400；没有 gate，因此不得解释为 reference 认证或弃权率。此结果凸显相关 blocks 下当前 percentile intervals 欠覆盖，仍仅适用于声明的 synthetic amplitude fit。

### 已完成：C independent links/probes 扩样

[预注册、20 个新增数据集原始记录、两张诊断图与逐预算表](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/cpu_close_20261119_C_expanded_v1/README.md)：seed 100–119、$\epsilon=0,0.1$、每起点目标调用预算 25/100/400、3 个共同起点；80/80 class×epsilon×dataset 设计满列秩，缩放条件数 $5.42\times10^5$–$7.31\times10^5$。在 $\epsilon=0.1$、最大预算下，A/SLSQP test error $4.16\times10^{-4}$、B/SLSQP $3.57\times10^{-5}$；Powell 对应 $8.14\times10^{-2}$ / $1.10\times10^{-1}$，60/60 起点预算截断。20 datasets 的 paired intervals 是有限描述；目标预算匹配不等于 walltime 匹配，因为 SLSQP 解析梯度不计为 objective call。仍不回答 learned smear 对 SLSQP 的物理比较。

### 已完成：N=100 独立 C CPU follow-up

[新 seed 预注册、200 个 class×epsilon cells 原始 trace、配对区间和两张图](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/cpu_close_20261119_C_followup_n100_v2/README.md)使用 seed 1100–1199、$\epsilon=0,0.1$、3 个共同起点、25/100/400 objective calls/start。200/200 cells 成功；最大预算下没有 solver-before-budget failure 或 large-residual success，5165 个 budget exhaustion 均按截断保留。$\epsilon=0.1$ 时 Powell−SLSQP held-out error 在六个 class×budget 条件都为正且 dataset bootstrap CI95 不跨零；例如 A 类 400 calls/start 为 0.07974 [0.07855,0.08103]，B 类为 0.11097 [0.10991,0.11183]。该独立重复加强的是此 synthetic matched-link 目标上的 SLSQP/Powell 数值基线，不是 learned smear 优势，也不代替真实 GPU $W$。

第一次 N=100 follow-up 的 100 个 seed loops 在内存中完成，但写盘时错读 `protocol.json` 而未持久化任何 raw/summary；审计记录见 [v1 RUN_REVIEW](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/cpu_close_20261119_C_followup_n100_v1/RUN_REVIEW.md)。修正后用新 seed 与独立 v2 目录完成，不复用失败 run 的结果。

### 已完成：B 多质量 reference-shape sensitivity

[三质量 preregistration、原始记录与两张图](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/cpu_close_20261119_B_multimass_v1/README.md)按质量 $m=0.25,0.5,0.9$、reference mass offset $0,0.03$，每 cell 400 独立 datasets / 400 paired bootstrap draws 做配对对照。正确 reference 下四时点同时覆盖 $q=1$ 的比例为 0.730–0.770；错误 reference 下均为 0/400（CP95 上界 0.00918）。pointwise 覆盖仅 0.8975–0.9275，显示当前 percentile interval 本身欠覆盖；同时 q-profile 覆盖与 $q=1$ reference-shape 检查必须分开解释。此结论只适用于固定 mass、late-window unweighted amplitude 的 synthetic 设定，不等于生产 covariance fit 或 learned smear 优势。

### 已完成：有限预算 learned-initializer 可行性 toy（不等于 learned smear）

[预注册、逐实例结果与两张图](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/cpu_initializer_toy_v1/README.md)在 8 维异质 SPD 二次目标上用 200 个训练实例、100 个 IID held-out 和 100 个分布偏移实例，比较 learned initializer + SLSQP、zero/random 多起点 SLSQP、Powell 与一个固定 schedule 的 Chebyshev 迭代；预算为 8/24/64 objective calls。IID 下 budget=8 时 comparator-minus-learned risk：zero/random SLSQP 为 0.763（paired CI95 [0.634,0.903]），Chebyshev 为 0.113 [0.077,0.154]；budget=64 时 shifted 组 SLSQP 与 learned 差约 $-7.3\times10^{-12}$，已到数值收敛尺度，不能称实质优势。训练约 0.00291 CPU 秒，成本按 100 次部署摊销，并单列推理/求解耗时。解析标签让该 toy 对学习器有利；当前 Chebyshev 实现非最优；多比较区间未校正。因此只说明某些异质目标的**有限预算初始化**有算法可行性，不能推出 learned smear、真实 gauge 数据或物理风险优越。

已进一步在[项目内 $H/B$ cross-link objective](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/cpu_operator_initializer_v1/README.md)上做预注册测试：40 train / 10 validation / 40 IID held-out / 40 shift links，用 gauge-invariant traces/Frobenius summaries 学预测训练 links 上实际 SLSQP 得到的系数。预算 25 时，IID zero/random SLSQP minus learned error 为 0.04907（dataset CI95 [0.04508,0.05315]），shift 为 0.04868 [0.04431,0.05329]；但 IID strong spectral minus learned 为 $-1.90\times10^{-4}$ [$-2.16,-1.63]\times10^{-4}$，即该谱基线略优；shift 下差值为 $+4.08\times10^{-4}$ [$3.55,4.64]\times10^{-4}$，有限条件下 learned 略优。训练/选参约 1.339 CPU 秒；预算 400 时 SLSQP 误差已更低。谱基线不是最优约束 Chebyshev，shift 仅扩大 $\epsilon$。结论只限这一合成 operator objective：initializer 有时可改善低预算 SLSQP，但没有一致超过强谱基线或高预算 SLSQP；没有真实 $W$、learned smear kernel 或物理优势证据。

### 已完成：冻结 initializer 结果的训练成本摊销后处理

[独立成本图文与表](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/cpu_initializer_amortization_v1/README.md)只后处理已有 `cpu_operator_initializer_v1/raw.json`，没有重跑或改写 synthetic 数据。一次训练/选择成本为1.339 CPU秒；按每个已观测 cell 40条 test links 的部署 wall time 均值，initializer 相对 zero/random SLSQP 的摊销 break-even 为192–5,203次重复部署。相对该实现中的 fixed spectral baseline，六个 cell 的 spectral 单例耗时都更低，因此 CPU 时间上没有 break-even。配对 risk 仍取决于 population：IID 下 spectral 略优于 initializer，epsilon-shift 下 initializer 略优；高预算时 SLSQP 的 held-out risk 更低。该结果说明有限预算 warm-start 有条件可行，但训练开销与谱基成本使它没有稳定的总成本优势。计时只来自一个 CPU toy、每 cell 40条已观察样本；epsilon-shift 不是一般 gauge 分布迁移，谱基也不是最优 Chebyshev。图和 CSV 均不构成 learned smear 或物理优势证据；不应重复同一 benchmark。

## 4. LatticeArena 可扩展部分

- **物理任务层**：新的 smear family、observable、momentum / irrep、边界条件通过 `OperatorContract` 和 frozen protocol 声明。
- **测量层**：`MeasurementBundle` 提供新响应矩阵/field capability；真实采集走 measurement plan、RunSpec 和 preflight。
- **求解层**：`SearchAdapter` 接 Powell/SLSQP/learned optimizer，外部预算、统一 stop rule、收敛 trace 不依赖模型自报。
- **统计层**：共享 independent/paired block resampling、reference gate、置信区间与 multiplicity spec；物理认证由 receipt 链驱动。
- **研究层**：扩展 basis、对称性保持的 feature ablation、multi-momentum / multi-volume test；每个方向都保留 blind split、
  strong baseline、成本与失败诊断。

## 5. 下一个 agent 开工顺序

D01 的C1/C2确认性族已经确定（F=2）。本节第一步沿 **D01 → D04 → D06** 选择reference与最小效应，并落实D11固定degree4 Chebyshev建议的物理contract，记录研究者所选方案及理由，再按D02–D19其余项目回填统计、数值和执行约束；未批准项继续保留TBD。所有后续诊断仍须给出输入、planned/有效分母、失败/弃权状态、图或清楚文字流程及证据边界，不能用方案文本冒充新实验。

先审阅新的 [Optimize 生产前研究决策单](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/tasks/optimize_gaussian/protocol_decision_review_20261121.md)：其中D01–D19逐项区分现有证据已支持的约束与研究者仍须选择的TBD，提供建议/备选、估计量与成本后果、分母/失败状态和建议图表。所有建议仍是 `REVIEW_DRAFT_NOT_FROZEN`；r_star、reference、solver残差/预算、弃权与统计规则不得从旧pilot/toy回填。特别先明确fit/dev-only开发与sealed最终blind的RunSpec阶段表示，避免builder为全部test输入绑定hash时提前读取sealed bytes；当前fixed-W训练入口也不能仅因worker/preflight通过就宣称trainable-W。该决策单本轮只做文本/静态接口审阅，未运行测试、生产输入读取或训练。

1. 当前 CPU 通用 SPD 与项目 $H/B$ cross-link initializer 只显示短预算下对部分 zero/random SLSQP 的条件性改善；IID strong spectral 略优，高预算 SLSQP 更低。因此**尚无稳定超过强基线的证据**，不再以相同 synthetic labels 扩样。v9 GPU feature/系数重训、固定随机 $W$ 与 IDs70–99 的 held-out评估已经做过；不要把固定$W$说成训练所得$W$。先读[v9 provenance audit](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/v9_provenance_audit_v1/README.md)：可恢复的源码 hash 已闭合；仍缺v9运行/设备receipt与原始propagator bytes。D链0–99的Pion propagator与receipt现已核验且可复用，Optimize下一步先冻结 Optimize-specific RunSpec、objective/reference/预算和代码hash；因Pion reference gate失败，未经独立确认的新reference只能支持明确标注的开发诊断，不能产生物理分数。任何模型/solver盲测须在共享holdout锁定后另行解盲。
2. **冻结 P3 决策量与真实 RunSpec，再运行 GPU 实验。** [`doc/optimize_p3_design.md`](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/doc/optimize_p3_design.md) 已将实用改善阈值和样本量/功效标为 `TBD/NOT_RUN`。RunSpec 生成的 worker 命令指向 `train.main`；外层 `worker_runspec` 通过 `hard_timeout` 监督完整训练子进程，并在成功终止后核验产物 hash。CPU 侧现已把 `solver_contract` 接入 `train.optimize`：显式协议必须冻结调用数、walltime、KKT residual 定义/阈值及 solver-success 要求；RunSpec builder 在绑定输入前拒绝缺失策略，child 核验 protocol hash；最终点重算与 KKT probes 同样计入预算；拒绝时写 no-clobber `diagnostic_only` 收据并在 learned-candidate dev-val/test 评分前停止。确定性调用轨迹和文字说明见[CPU 接线报告](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/solver_contract_train_integration_cpu_v1/README.md)；相关四模块 **87 passed**。它只验证 CPU fixture，不代表已选好生产策略、GPU/QUDA 中断安全或物理 PASS。

   下一位 agent 的顺序：先请研究者冻结最小有意义效应、KKT residual 是否适用及其阈值、objective-call 口径、walltime 计时边界、失败/abstain、paired CI 层级、多重性、样本量与停止规则；不得从 CPU 玩具值回填。然后绑定真实每臂 configuration IDs、同一 links/probes/target 与双预算，生成 production RunSpec 并通过 hash preflight。reference FAIL 时所有风险仍标 `diagnostic_only`。获得 GPU 与数据 split 授权后，依序做短时启动/中断 smoke、七幂/嵌套$Q$ × SLSQP/Powell 2×2和real/uniform/shuffled消融；该2×2只分辨算符类与solver效应，不回答 trainable-$W$ 是否优于 SLSQP。每个诊断必须交付 paired CI、有效秩、calls/time best-so-far、成本拆分、KKT/residual、失败/弃权图或清楚文字解释。
3. 主候选固定为真正trainable-$W$，两个确认性contrast固定为同预算多起点SLSQP与固定强谱基（建议degree4 Chebyshev）；Gaussian/fixed-W为secondary/mechanism，initializer/restart单独定义不冒充learned smear。固定阶数来自偏谱类heat-kernel合成诊断，不能称物理最强或从blind调阶。匹配objective-call和walltime两种预算，分列setup/training、inference、solve、总成本、GPU memory、KKT/residual和失败率；输出paired-risk差/区间图、best-so-far收敛图、成本分解图和失败样本说明。每项实际对照同时报告planned/valid/failure/abstain分母和文字边界；软件预检交数据流/receipt与文字说明。CPU toy结果只支持有限预算算法可行性，不能替代真实smear结论。
4. D IDs140–199是Pion/Optimize/Wilson共同持有的60个未读配置；Wilson 已有封存该区间的 RunSpec，但 Pion/Optimize 的全部物理 candidates 与 score/endpoint 仍须在任何读取前共同冻结源码与分析 hash。一次 shared holdout 解盲后不可再用于调参或作为多份相互独立的验证。保存所有失败和预算截断；无GPU/生产输入或该holdout已被打开时，把相应实验标为`NOT_RUN`/已观察并说明边界，不得伪称blind。
5. 即使得到负结果，也保留结果图、区间、失败/弃权表和解释并关闭对应研究分支。production 2×2、真实trainable-$W$/initializer与新blind comparison完成前，本文件不得删除。

### 已完成：nested solver contract 的 CPU 固定样例与 Optimize 接线诊断

[`solver_contract.py`](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/tasks/optimize_gaussian/solver_contract.py) 与[`诊断报告`](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/cpu_solver_contract_v1/README.md)实现并检查 objective-call cap、协作式 walltime、非有限目标、solver success、显式 residual 阈值、最终参数独立重评估、逐臂有序 configuration ID 一致性，以及只把 accepted receipt 交给 `core.search`。原始独立 adapter 快照是 **11 passed**；加入 residual-probe 计数与 budget-exhaustion 传播用例后，`test_optimize_solver_contract.py` 当前为 **13 passed**。原报告中的六个确定性软件场景为1个accepted、5个拒绝；新增两个用例专门验证 KKT probes 不改善 best-so-far 以及 KKT probe 自身耗尽预算时 fail closed。这些是覆盖场景数，不是估计出的失败率。paired ID 检查只核对 ID/order/hash，没有实际 paired-risk 比较。

原独立 adapter 报告含 calls/time best-so-far 曲线、接受/拒绝计数图、原始逐例 receipt、CSV 失败表和边界说明。随后已将其接入 `train.optimize`、显式 hash-bound `train.main` 与 RunSpec preflight，并扩展定向 CPU 测试：接受样例为 1 次 solver 调用 + 1 次最终点重算 + 5 次二维 KKT 探测（共7次）；KKT 探测计入 calls/time 但不进入 optimizer best-so-far。预算仅容纳前两次调用时，第3次 KKT 请求触发拒绝，训练入口生成 no-clobber rejection receipt、`physical_score=null`，不写 `results.json` / `manifest.json`，也不对 learned candidate 继续 dev-val/test 评分。RunSpec 测试验证没有完整策略时在输入 hash 绑定之前拒绝。完整的固定样例、测试路径和范围限制见[训练接线 CPU 报告](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/solver_contract_train_integration_cpu_v1/README.md)；`test_optimize_solver_contract.py`、`test_optimize_runspec_builder.py`、`test_optimize_training_protocol.py`、`test_optimize_gaussian.py` 合计 **87 passed**。

walltime 仍是同步的 cooperative guard，无法中断正在执行的 objective/native/GPU call；慢调用可超时一次后拒绝，不能称 hard stop。CPU gate 接线已完成，但 production RunSpec、策略阈值、reference/统计决策、paired risk 与真实 GPU/QUDA receipt 均未完成，故保留本 TODO。下一步先冻结研究策略及真实文件输入/输出和 hash，再生成 production RunSpec；授权后先完成 GPU 短时启动/中断与设备回收诊断，最后才能运行2×2和learned-smear比较。报告需同时提供图或可复查文字流程、paired 区间、calls/time best-so-far、成本、KKT/residual与失败/弃权解释。

**Walltime 语义与建议（阈值仍待研究者冻结）：**当前 `run_solver_with_contract` 的 walltime 是协作式软上限：objective 调用开始前检查，返回后若超限便拒绝该次候选并停止后续调用，但无法中断正在运行的 Python/native/GPU 求解；实际用时可超过上限一个调用时长。由于主对照要求匹配 walltime 和 objective-call 两种预算，我建议正式比较采用可终止的 hard walltime cap；保留当前 wrapper 作为 CPU/诊断 soft-cap 路径，并明确报告超限时长，不能把它称为严格匹配的生产预算。研究者仍须在 RunSpec 中冻结 cap 数值、startup/training/inference/solve/teardown 是否计入总时长和超限状态口径。

**Hard-cap 的可行实现：**由 parent 进程持有 deadline 和冻结 RunSpec，启动独立、可杀除的 worker/process group；worker 在子进程内初始化求解器/GPU，从只读输入路径执行一个完整 arm 或预先冻结的配置批次。每个 objective 完成后，worker 原子写入包含 source/input/arm identity、calls、累计时间、best-so-far 和最后成功候选的 checkpoint/receipt。deadline 到达时 parent 发送 TERM，宽限期后 KILL，并原子写 `hard_timeout` receipt，绑定退出码、elapsed、最后 checkpoint hash、完成调用数及被中断的 in-flight 调用；该调用不得被记为成功或纳入 risk。禁止 fork 已初始化的 CUDA/QUDA context；GPU setup、模型训练、worker启动和收尾按冻结规则计入各臂成本。

**Standalone CPU hard-timeout smoke 已完成，production 接线未完成（2026-11-20）：**[`hard_timeout.py`](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/tasks/optimize_gaussian/hard_timeout.py) 现能为无 shell 的独立 worker 进程组实施 deadline、TERM/grace/SIGKILL、checkpoint/log 收据和 no-clobber 发布；清理未确认时返回 `termination_failed`，不对仍可能变化的 checkpoint/log 生成 hash。文本时间线和原始汇总见[hard-timeout CPU diagnostic v2](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/hard_timeout_cpu_v2/README.md)及其`summary.json`。四个合成场景分别为正常退出（4次durable calls/0.253s）、TERM响应型超时（2次/ cap 0.220s、总0.239s）、忽略TERM后SIGKILL（4次/总0.314s）和worker exit 7；所有输出的`physical_score=null`。额外的定向用例注入SIGKILL拒绝且保留一个TERM-resistant descendant，监督器在有界时间内标`termination_failed`、不哈希可变文件，并由延迟marker证明子进程仍存活；该失败被正确保留，而非记作hard-timeout成功。聚焦测试 **9 passed**，覆盖正常/超时/异常/启动失败/进程组子进程/TERM-resistant worker/KILL拒绝/no-clobber/输入冲突。早期v1报告保留了初版SIGKILL `PermissionError` 的诊断截面，v2已按失败闭锁逻辑重跑。

这只证明 standalone CPU 进程监督器在这些合成条件下的执行控制；不证明SciPy/native/GPU/QUDA训练可以被正确中断，也未计生产初始化、训练或启动摊销。原下一步“先接入 frozen RunSpec”现已推进到**执行层 CPU 闭环**：[`worker_runspec.py`](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/tasks/optimize_gaussian/worker_runspec.py) 在 worker 启动前解析无重复键的 RunSpec、校验协议/源码/输入文件 SHA-256、检查 split ID 不重叠及 no-clobber 输出，再把 RunSpec hash、文件身份、split 和 hard-timeout 结果写入执行收据。schema v2 还要求显式 child environment，避免继承未绑定的 parent environment；receipt 仅存环境 hash 和变量名。新增可选 `outputs.artifacts` 使 parent receipt 在子进程成功后记录训练产物 SHA-256；可声明位于 child 新建目录的产物路径，失败/超时/成功 leader 留下进程组子进程时不接受其产物，缺件和CLI非零退出也会显式标记。v3 的环境隔离与 preflight 图文仍见[worker RunSpec CPU 报告](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/worker_runspec_cpu_v3/README.md)，新 receipt 行为的合成文字诊断与 **24 passed** 见[worker RunSpec CPU v4](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/worker_runspec_cpu_v4/README.md)。v1、v2报告保留为历史快照，不覆盖。

流程级 launcher 已连接到 builder 生成的 `train.main` 命令，原 CPU builder→supervisor pytest fixture 仍用假 child 检验 parent artifact receipts；新增[synthetic CPU integration](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/cpu_train_main_supervisor_v2/README.md)则在独立 worker 子进程中真正调用 `train.main`，保留 SciPy Powell 与 solver contract，并验证4项训练产物 hash。solver 拒绝 JSON 仍列入 RunSpec optional diagnostics，父级只在子进程组稳定后记录其 hash；该字段与成功 artifacts 分开，非零 child 仍保持失败状态。实际 Optimize production RunSpec/策略阈值尚未冻结，也未验证 GPU/QUDA context、设备回收、真实 basis cache 和 solver-arm checkpoint。下一位先冻结 Optimize-specific protocol、允许输入/输出、reference/statistical endpoint 与策略，再让 builder 生成真实文件 hash 绑定的 RunSpec；确认 `results.json` / `manifest.json` 等列入 `outputs.artifacts` 并由 parent receipt 核对成功产物 hash，同时审阅 `outputs.diagnostics` 中的 rejection receipt hash。CPU相关 6 模块合计 113 passed；新 integration 是单独的直接执行诊断，不是额外 pytest 通过数。之后仅在获授权 GPU 环境做短时启动/中断 smoke，记录设备回收、显存、setup/training/inference/solve/teardown walltime。不得把 synthetic fixture 当 production GPU hard-cap 或物理比较。若研究者选择软预算，保留 cooperative solver wrapper，但报告每个超限调用的 overshoot，不宣称 hard cap。真实比较仍需交付 paired-risk 区间、calls/time best-so-far、成本图、残差/失败/弃权表与解释；不得把 CPU fixture 当统计结果，也不得扩样相同 synthetic labels。

硬件范围说明：真实 $W$、P3 的生产数据 2×2、metadata 复采与 blind test 依赖 GPU/生产输入；§3 中新增噪声结构或 independent links/probes 的统计诊断仍可在 CPU 做。若选择这些 CPU 支线，必须独立预注册并配 paired 区间图、失败样本和文字解释；它们不能替代真实 learned-smear 对照。

### 2026-11-21 补充：实际 hopping correlator 的 trainable-W 链式梯度（CPU）

此前解析梯度只验证 synthetic coefficient-MSE。现在[`trainable_w.correlator_and_gradient`](trainable_w.py)将链式梯度接到真实 `features.directional_features` 与 `kernels.apply_kernel` 的 paired forward/backward hopping correlator，并明确包含每个方向独立空间中心化的伴随投影。新增[有限差分收敛报告与图](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/results/optimize_gaussian/trainable_w_operator_gradient_20261121_v1/README.md)：H=2合成fixture的9个步长×7个W/v/c参数共63项，63 valid、0 failed、0 abstained；固定测试步长2e-6的最大绝对误差约`3.52e-8`，步长扫描最佳约`3.24e-9`（步长`1e-4`）。输入是确定性合成SU(3) links与复probe field，空间存储(z,y,x)、方向(x,y,z)；没有使用生产gauge/propagator、GPU/QUDA、reference或sealed IDs。

该梯度报告本身完成的是“参数→实际CPU hopping correlator”局部算子导数核验，**没有**验证ensemble risk/reference的导数、训练选择、SLSQP对照或生产`train.main`。另一个独立D11 CPU候选现已覆盖Gaussian sandwich/positivity代数，但仍未接入生产训练。后续只有D04/D06及其他研究合同获审阅后，才把parameterized measurement evaluator接入RunSpec-bound `train.main`，用独立CPU fixture核验数据流/hash/failure receipt，再另行申请GPU验证。不得把63个有限差分点当作样本或物理证据。

### 2026-11-21 补充：确认性基线与下一步决策交接

Optimize 主比较必须同时保留同预算 SLSQP 与一个预先固定的强谱基线：用户已将确认性族定为 `F=2`，分别比较 trainable-W 对 SLSQP、trainable-W 对强谱基。现有[生产前研究决策单](https://github.com/DianjunZhao/LatticeArena/blob/54f0e97ba85b7f8f724c31d9f9d3512f040583a9/tasks/optimize_gaussian/protocol_decision_review_20261121.md)已含一页文字流程和 D04/D06/D11 备选审阅表；其中 degree-4 Chebyshev 只是待批准的固定基候选，operator、谱尺度、系数域/正则、共享 objective 和 calls+walltime 双预算仍须解盲前冻结。CPU heat-kernel 结果不能证明该谱基对物理问题最强。

当前最先决的是 D04 reference 与 D06 最小有意义物理效应：Pion reference 仍 FAIL 时，任何新风险都只能标 `diagnostic_only`，不得填造 `r_star` 或解封盲测。研究者决策与策略冻结后，执行者才生成真实文件/source/environment hash 绑定的 RunSpec，接入真正更新 W 的训练入口，交 CPU 接口/梯度图文；另行授权后再做设备 smoke 与真实比较。交付须含两项 paired-risk 区间图、calls/time best-so-far、成本拆分、residual/KKT、planned/valid/failure/abstain 分母及失败解释。本轮没有训练、GPU 或 sealed split 数据读取；TODO继续保留。

### 2026-11-25：bounded block-risk 的非渐近校准候选与依赖负控

按用户接受的“bounded block-risk 是不同 estimand”方向，新增 [`bounded_block_risk.py`](bounded_block_risk.py) 与 [`d06_bounded_block_risk_diagnostic.py`](d06_bounded_block_risk_diagnostic.py)，报告为[图文 calibration v4](../../results/optimize_gaussian/d06_bounded_block_risk_cpu_20261125_v4/README.md)。RNG seed=`20261125`，协议 JSON 同时存入分方法 planned/valid/failed/abstained 总数与关键覆盖/功效摘要；源与 artifact SHA-256 均复核匹配。定义每个**配对宏块**的有界分数

`X[b,a]=(1/4) Σ_{t=2..5} min((q[b,a,t]-1)^2, 1) ∈ [0,1]`，`μ_a=E[X[b,a]]`，

与 `D04 R_a=mean_t{Var_boot(q_a(t))+|E_boot(q_a(t))-1|²}` 不同，不能替换或混报。保持用户选定 `F=2`、`r_star=10/9`。每个比较直接检验配对差 `D[b]=X[b,comparator]-r_star X[b,W]` 是否正，同时要求 `μ_W` 的单侧下界大于 0；总 `α=0.05` 平均分给两个比较和一个共享正分母界（每界 `δ=1/60`）。Hoeffding 有界界为保守基线；empirical-Bernstein 候选使用 Maurer–Pontil 2009 Theorem 4，并按观测区间宽度缩放。其数学前提是宏块独立；[原始定理](https://arxiv.org/abs/0907.3740)不支持把相关 HMC 配置块自动视为独立样本。

IID 校准设计覆盖 3 个已知真值有界 DGP、低于/恰在/高于最小效应的三组双比较风险比、`n=200/1,000/5,000/20,000`、每 cell 1,000 个独立 synthetic panels。每方法计划 `36,000` 面板；Hoeffding valid/failed/abstained=`26,371/0/9,629`，empirical-Bernstein=`27,000/0/9,000`。IID 网格中，两方法都在 24 个 null/boundary cells 的合计 `24,000` panels 中观察到 `0` 次家族发布。empirical-Bernstein 的联合下界覆盖在 `n≤5,000` 每格为 `100%`；`n=20,000` 最低格为 `999/1,000`，Wilson 95% 下界 `99.44%`。在两项真实比值都超过 `r_star` 的设计中，它等格平均家族发布率从 `n=5,000` 的 `35.0%`（cell range `1.5%–100%`）升到 `n=20,000` 的 `98.4%`（`95.2%–100%`）；Hoeffding 到 `20,000` 仍无家族发布。该结果表示经验 Bernstein 在 IID 有界 synthetic risk 下兼具高覆盖和一定功效，而 Hoeffding 明显保守；它不等于统计 reviewer 批准，也不表示物理链满足 IID。

依赖负控另外计划每个 lag-1 `rho=0/0.5/0.8/0.9/0.95/0.98` 各 `10,000` 个、每 panel 1,000 宏块。empirical-Bernstein 联合下界覆盖从 `rho=0.8` 的 `98.61%` 降至 `rho=0.9/0.95/0.98` 的 `92.66%/82.22%/77.09%`；最高相关度的 Wilson 95% 区间为 `[76.26%,77.90%]`。Hoeffding 在 `rho=0.98` 也仅 `88.47%` 并大量 abstain。这一负控明确显示依赖会破坏所用有限样本定理的覆盖解释；不能从“块长为 5”或轨迹间隔声称 independence。图同时展示覆盖、false release、功效与依赖敏感性；protocol/source/artifact hashes 全部核对通过。

新增 bounded-risk 专项测试 **9 passed**；本 checkout 的完整 `tests/test_optimize*.py` 为 **97 passed**。当前只交付了候选统计规则和 synthetic calibration，不改写 `research_contract_v1.json` 的 D04 主风险，不接入 production evaluator，不读取 gauge/reference/sealed split，不产生 `physical_score`。

后续顺序：先由独立统计 reviewer 复核“配对对比 + 分母正性 + 三界 Bonferroni”的覆盖逻辑和代码；再正式决定是否将 clipped block-risk 作为新版本 estimand（需保留旧 D04 结果并行报告）。无论该决定如何，真实执行前都需要可审计的独立宏块证明，或另推导并校准适用于 Markov/依赖宏块的修正界；否则 fail closed 为 `ABSTAIN`。之后还要冻结 block 构造、`q` 的 reference-fit uncertainty 传播与 confirmation split，再用独立授权的开发数据接 evaluator。文字流程：`frozen reference/candidates -> paired block q -> bounded score -> simultaneous contrast and positive-denominator bounds -> dependency receipt check -> release or abstain`。

### 2026-11-26：Optimize bounded block-risk 的 optimizer failure 计分与分母

合同已要求每臂失败 run 留在 assigned-arm 分母，但此前 bounded-risk helper 没有定义缺失 `q` 如何与真实 solver failure 区分。现将候选规则补为：若冻结的 optimizer 流程在某个配对宏块中对某臂没有返回 accepted finite candidate（包括预算耗尽或 frozen residual gate 失败），该臂整个宏块记 `X=1`，其它臂照常评分，宏块仍在 `n`；无法归类为 optimizer failure 的非有限/损坏输入不填补、不记作优化损失，runner 必须 `ABSTAIN`。因此 `failed_blocks=0` 表示没有丢弃不可评分观测；已发生的算法失败另按臂计数并作为最大 bounded loss 纳入风险。

[`bounded_block_risk.py`](bounded_block_risk.py) 的 `bounded_block_scores(..., arm_failure_mask=...)` 现在只允许在明确标记失败的整个 arm-block 上接受缺失 `q`，并强制分数为 1；`bounded_block_risk_decision` 报告每臂算法失败数、含任一算法失败的宏块数及仍参与统计的 scored denominator。无失败掩码的 NaN、错误 mask shape/type、或把失败标记为低于 1 的分数都 fail closed。定向代码回归 `17 passed`（包含该 helper 与 bounded-risk diagnostic tests）。

[精确混合敏感度图文](../../results/optimize_gaussian/d06_failure_penalty_cpu_20261126_v1/README.md)用成功风险 `(.04,.08,.075)` 和 303 组 failure-probability × pattern 公式格（planned/valid/failed/abstained=`303/303/0/0`）说明它如何改变两项主比较；公式为 `mu_a=p_fail,a+(1-p_fail,a)*mu_success,a`。例如 trainable-W 独有 `5%` 失败时，两个期望风险比为 `0.909/0.852`；两 comparator 独有 `5%` 失败时为 `3.150/3.031`；三臂同为 `5%` 时为 `1.432/1.378`。这些风险与概率都是图中声明的示意输入，不是训练测得的失败率或统计覆盖结论。此评分仍在 `[0,1]`，但只有独立配对宏块假设成立时才能套用当前浓缩界；HMC 依赖校正、D04 reference uncertainty 与 trainable-arm selection 仍未解决。

文字流程：`预分配 arm + paired macroblock -> solver accepted: four-time clipped loss; solver failed: score=1 -> 保留全臂/全 block denominator -> missing/corrupt source not explained by solver status: ABSTAIN -> apply calibrated bound only after dependence and reference receipts`。当前没有 production solver receipt，主 D04 不变，`physical_score=null`；进入 evaluator 前仍须冻结 residual/timeout 的 failure 判定并校准真实失败过程。

实现后完整 Optimize 测试文件集 `tests/test_optimize*.py` 为 **105 passed**（CPU）；failure-penalty 报告 artifact hashes **4/4**、源文件 hashes **2/2** 匹配。此验证只覆盖计分接口和精确示意公式，不构成 production solver 或失败率测量。

### 2026-11-26：D06 有界 block-risk 的 Markov 依赖备选

独立 macroblock 假设当前没有证据；为此补了一个严格条件化的 Markov 路线，而不是把 HMC 间隔、经验自相关时间或 synthetic coverage 当成独立性证明。实现见 [`bounded_block_risk_markov.py`](bounded_block_risk_markov.py)，解析情景与 known-gap stress runner 见 [`d06_markov_dependence_diagnostic.py`](d06_markov_dependence_diagnostic.py)，[图文报告](../../results/optimize_gaussian/d06_markov_dependence_cpu_20261126_v1/README.md)。

该路线对三项同时事件作 union bound：trainable-$W$ 均值的正分母界，以及相对同预算 SLSQP 与 degree-4 Chebyshev 的两个固定对比。设任一 summand 的 support width 为 `w`，则使用 `Var_pi(f)≤w²/4`、`|f−E_pi f|≤w`；风险均值的 support 是 `[0,1]`，配对 contrast 的 support 是 `[-r_star,1]`。按 Paulin corrected arXiv v5 Theorem 3.11 / Eq. (3.25) 反解两侧 Bernstein 尾概率，再保守地用于 lower bound。若 burn-in 后的**完整 paired-block state** 初始律与 stationary law 的 TV 距离有上界 `δ_TV`，将其对三个事件的并集只支付一次，剩余 `alpha_total−δ_TV` 再平分三界；这对应 Proposition 3.15 / Eq. (3.29)，并需要针对该链和该 burn-in 的可审阅 TV receipt。采用的数学来源：[Paulin corrected arXiv v5](https://arxiv.org/html/1212.2015v5)。

[诊断图](../../results/optimize_gaussian/d06_markov_dependence_cpu_20261126_v1/markov_dependence_feasibility.png)左栏显示 support-only 半径随块数和假设 `gamma_ps` 的变化；144 个公式 scenario 的 planned/valid/failed/abstained=`144/144/0/0`，其中仅 `69/144` 在分母和 contrast 的最乐观支持端点下仍有理论放行空间。对假设 contrast margin `0.01–0.10`，所需 scored blocks 跨声明 gap/TV 网格为 `6,290–24,732,070`；这只是条件样本量，不是项目效应估计。

中、右栏是独立 two-state Markov toy 的已知 gap 覆盖压力测试：`n=250/2,000/20,000`，每 cell 4,000 个 panels，共 `36,000`；已知 `gamma_ps=1/0.36/0.0975`。joint lower-bound coverage 九格全为 `1.000`（每格 Wilson MC 95% 下界 `0.999`）；SLSQP 真比值恰在 `r_star` 时，家族误放行计数 `0/4,000` 每格，Wilson 上界约 `0.10%`。这验证的是实现对给定合成链的保守行为；不验证真实 HMC 链的齐次性、stationarity、gap 或统计 reviewer 的结论。生成器种类和块数覆盖有限，因此这组全覆盖结果不能当成广泛 coverage guarantee。

本轮 `tests/test_optimize*.py` 为 **122 passed**；Markov 报告 payload hashes **7/7**、源文件 hashes **3/3** 匹配，`git diff --check` 通过。D04 原 estimand 不改；bounded block-risk 已被用户接受为并列的不同 estimand，继续校准但不替代 D04。未拿到独立 paired-block `gamma_ps`、burn-in/TV、score-map 与 D04 reference receipts 前，不接 confirmation evaluator。最简流程：`冻结三臂和 bounded score -> 构造配对宏块链 -> 审查 homogeneity + gamma_ps + stationary/TV receipts -> 三个同时下界 -> 两项都通过才可申请下一 gate，否则 ABSTAIN/no release`。当前无真实 solver receipt、无数据授权、无 GPU/QUDA 运行，`physical_score=null`；TODO 保留。

### 2026-11-26 方法选择状态核对与下一步

回查当前 [`research_contract_v1.json`](research_contract_v1.json) 后，旧文中“D04 reference、D06 `r_star`、D11 operator 尚待研究者选择”的话只适用于早期草案，不能再作为当前交接。当前选择是：

- **D04 reference：**独立 E-chain、candidate-blind 的 PP-even/A4P-odd 联合两态谱拟合；主窗 `[3,8]`，`tmin=2,3,4` 与 2/3-state 检查，固定 covariance shrinkage/eigenvalue floor，reference gate 和失败规则均写入合同。E-chain 数据/拟合 receipt 和独立 review 仍为 null；旧 D0–99 单态 gate 失败，故当前风险仍只能是 `diagnostic_only`。
- **D06 主比较：**保留 `F=2`，trainable-$W$ 分别对同预算 SLSQP 和固定 degree-4 Chebyshev；`r_star=10/9`。新增 bounded block-risk 是另一个并列 estimand，用户已接受其不同于 D04；它不能替换 D04，也无需再次询问是否保留。Markov 界只在其 paired-block chain、score map、`gamma_ps` 与 burn-in/TV 收据被独立审阅后才可使用。
- **D11 强谱对照：**合同已记录三维周期空间 `H=sum_i(T_i+T_i^dagger-2I)`、`[-12,0]` 谱域、degree-4 Bernstein controls 到 Chebyshev 的固定映射、`0<=b0<=...<=b4=1`、共同未归一化 `G^2 P4(H) G^2` sandwich；CPU helper 的 U(3) 数学条件与生产 SU(3) identity receipt 分开。该 adapter 仍是 diagnostic schema；生产 links、独立 operator review 和 evaluator 均未闭合。

因此不再让用户重复选择上述默认方法，也不把它们标成已认证。Optimize adapter 已在 PR #6 接入独立 D04/D11 stages 与原子 D06 bundle；为保持该 PR 基于 `main` 可检查，它携带共享 gate 的最小实现，Pion adapter/诊断仍单独在 PR #5。`core/source_integrity.py` 已提供只检查显式路径映射的通用 byte-hash preflight，但尚未连到 Optimize RunSpec/source reader。下一步由独立 reviewer 复核公式、单位与失败规则；owner 仍需提供身份绑定、获授权的 E-reference 与 paired solver receipts。通过 D04 gate 后，才可用固定 RunSpec 做配对训练与确认。160 objective/gradient-call cap、一次预设起点和所有训练/setup 成本计入单任务 walltime 已写入合同；真实硬件时间、设备回收和总成本仍待授权运行。方法和接口细节见[框架扩展图](../../doc/framework_extensibility.md)与本 TODO 文末的 adapter handoff。成功的 metadata gate 不认证任何数据或物理 claim。所有下一步图文都应分开报告诊断单位的 planned/valid/failed/abstained 与物理 lane 状态，保持 `NOT_READY/UNCERTIFIED`、`physical_score=null` 和 sealed split closed。

### 2026-11-26 D06 bounded-risk Markov 校准扩展：非可逆多态链

在早先的两态 reversible known-gap 压力测试之外，新增 [`d06_markov_multistate_diagnostic.py`](d06_markov_multistate_diagnostic.py) 与[图文报告](../../results/optimize_gaussian/d06_markov_multistate_cpu_20261126_v1/README.md)。它使用 19 态定向 cycle 加 uniform refresh：平稳分布为均匀分布，链非可逆；在零均值子空间上是缩放酉移位，因此 `gamma_ps=1-(1-eta)^2`，程序同时从转移矩阵 singular spectrum 数值复核。分数 map 取满 `[0,1]` 支持：trainable-W 风险精确为 `9/19`，SLSQP 为 `10/19`，其 paired contrast 恰好在 `r_star=10/9` 边界；Chebyshev 风险为 1，提供正对照 contrast `9/19`。TV=0.02 情景使用精确混合初始律 `nu=(1-rho)pi+rho*delta_0`，并把 TV 一次性从 family alpha 扣除。

12 个预设 cell（3 个 gap × 2 个 block 数 × 2 个初始 TV）共 14,400 个 panel、158,400,000 个 scored paired macroblocks，planned/valid/failed/abstained=`14400/14400/0/0`；没有排除链或 panel。所有 cell 的联合下界覆盖率均为 1.000，Wilson Monte Carlo 95% 区间均为 `[0.9968,1.0000]`；SLSQP 恰在阈值时没有 family release，最大 Wilson 上界为 0.0032。正对照的 Chebyshev 下界随 `gamma_ps`、样本量及 TV 变动，图中显示了误放行控制与可用性损失的权衡。报告还用同一实际路径对照公共 decision helper 与批量计算，数值一致；协议及图表产物 SHA-256 已记录。

这不是对真实 HMC 的校准证明，也不能给生产链填入 `gamma_ps` 或 TV；全支持界仍非常保守，高覆盖可以伴随低放行力。该结果只扩大了 bounded-risk 路线在一种新的非可逆多态合成族上的诊断覆盖，不替代 D04、candidate-selection 不确定度或独立统计 review。

**接下来的具体工作：**不再复跑两态或这组 19 态生成器的 seed/网格。由未参与实现的统计 reviewer 独立复核 pseudo-gap 推导、Paulin bound 反演、TV 并集校正及 boundary fixture；然后接收身份绑定的 paired-macroblock state/score map、生产链 `gamma_ps` lower-bound 与 burn-in/TV receipts、E-chain reference receipt 和配对 SLSQP/Chebyshev solver receipts。只有这些来源/分割/哈希均通过合同后，才固定真实数据 RunSpec、运行授权的 GPU dry-run 与比较。交付需继续包含风险下界图、calls/time 与成本分解、residual/KKT、failure/abstention 表和完整 denominator。任何输入缺失时保留 `NOT_READY/UNCERTIFIED`、`physical_score=null`，不读取 sealed IDs。

### 2026-11-26 共享 evidence schema v2 的 Optimize 接线要求

PR #5 的共享 gate 已升级到 evidence-envelope v2：review 必须同时带 `reviewer_role_claim="independent"`、review scope/receipt hash 和 `independence_claim_sha256`。这些是可审计声明的索引，不验证 reviewer 身份或独立性；Optimize adapter 仍须在 task 层核验 review artifact 与 source bytes。为使 PR #6 能基于 `main` 单独检查，本 PR 携带该 shared gate 的最小 runtime 和 schema tests；PR #5 仍保留 Pion task adapter/诊断。PR #5 合并后应核对共享文件一致性，不能重复解释成两份不同 gate。

适配器应至少构造并分别检查两类 envelope：

1. `D04_reference`：绑定独立 candidate-blind E-chain、PP-even/A4P-odd 两态主拟合与预定窗口/model checks、固定 covariance 规则、reference source/receipt hashes、失败分母和独立审阅 scope。未通过参考门不能生成风险比较的物理解释。
2. `D06_paired_block_risk`：绑定 D-chain 的 paired macroblock state/map、bounded estimand 与 `r_star=10/9`，分别保留 trainable-W vs same-budget SLSQP、trainable-W vs degree-4 Chebyshev 两项对照及各自 multiplicity allocation；训练/优化失败留在 assigned-arm denominator。D06 envelope 不可代替 D04 reference envelope。
3. `D11_operator`：绑定 degree-4 Bernstein-to-Chebyshev controls、生产 SU(3) link provenance、共同 `G^2 P4(H) G^2` 实现和 operator-review scope；CPU U(3) fixture 不满足生产 SU(3) gate。

负例必须覆盖 D04/D06 chain 或 split 串用、reference/score-map/source hash 过期、仅有 self-review、缺独立性声明 hash、160-call denominator 未闭合、任一 comparison 缺项以及 production link 仅有 U(3) unitary 而无 SU(3) receipt。每个 task gate 仍须由 `RunReceipt` 记录 planned/valid/failed/abstained；schema+分母完整最多到 `draft`，缺项进入 `abstained`。当前 E reference、paired solver/Markov receipts、独立审阅及获授权硬件成本均缺，因此本项不生成 score，不开放 sealed split，也不执行真实 W/GPU 工作。

文字流程：`独立 E-chain D04 拟合与 source/review 核验 -> D04 reference gate -> D06 paired-block map 分别比较 SLSQP/Chebyshev 并计入失败 -> D11 SU(3) operator receipt -> 两项独立比较的误差界与成本 -> RunState draft 或 abstained`。D04 不通过即停止物理解读；D06 任一 planned arm 缺失不删分母；D11 只有 CPU U(3) fixture 时拒绝生产门。

本轮联合复跑 `tests/test_optimize*.py tests/test_wilson*.py`：**184 passed**。仅为 CPU 软件/合同回归；不证明 E reference、真实 Markov gap/TV、GPU 成本或物理 score。

PR #6 当前的 `apply_evidence_bundle_gate` 把 D06 的 SLSQP、Chebyshev 与 trainable-W envelopes 及共同正分母作为一个 statistical bundle 原子校验；D04 reference 与 D11 operator 仍走各自 stage，避免一项通过后误表示全合同通过。只有合同规定不同链时才要求 chain identity 不碰撞；这仍不证明统计独立。

### 2026-11-26 Optimize task adapter 接线与剩余边界

当前 PR #6 携带共享 evidence schema v2 的最小实现，使本 PR 可直接从 `main` 检查；完整 Pion task adapter/诊断仍单独留在 PR #5。本任务新增 [`evidence_adapter.py`](evidence_adapter.py)：

- `apply_optimize_d04_reference_gate` 绑定独立 E-chain、D04 joint PP/A4P map、固定来源/审阅/授权收据与 6 个 planned fit variants（3 个 `tmin` × 2/3-state）。它逐项检查相对质量误差 `<=0.05`、最小 GOF `p>=0.01`、最大 channel/window/model shift `<=2.5` paired SE、无 boundary hit；任一未运行、失败或摘要 hash 不符即 abstain。
- `apply_optimize_bounded_block_risk_gate` 仅接收用户接受的**并列 bounded estimand**，原子绑定 SLSQP、degree-4 Chebyshev、trainable-W 三 envelopes 与一个 paired chain。它检查 clipped-score/contrast maps、共享预算计划/成本账本 hash、共享 `gamma_ps`、burn-in/TV、`F=2` 两个 contrast 加共享正分母的三项 alpha、宏块分母 `N` 和 assigned-arm 分母 `3N`。`N` 不在 adapter 中预填；统计结果仍只到 draft，D04 原风险比没有被它替换或宣称已校准。
- `apply_optimize_d11_operator_gate` 绑定生产 link manifest、SU(3) provenance、以单调 Bernstein controls 经固定 exact dyadic transform 表示的 degree-4 Chebyshev operator、`[-12,0]` support 与逐例 operator summary。只有 CPU U(3) fixture 或 synthetic split 会拒绝；metadata 完整仍只到 draft。

文字流程：`D04 E-chain manifest -> 六种预定 joint-fit variant + reference summary gate -> reference draft/abstain`；`D11 production link/SU(3) receipt -> operator cases + spectral/Clenshaw checks -> operator draft/abstain`；`D06 paired macroblock score matrix -> SLSQP/Chebyshev 对照 + trainable-W 分母 -> gamma_ps/TV 校正后的三项 Markov lower bounds -> statistical draft/abstain`。每条通过都不等于 source bytes 已验证、reviewer 身份已认证或物理 score 已通过。

本轮 task-adapter fixtures：D04 `reference_fit_variants=6/6/0/0`；D11 `operator_cases=2/2/0/0`；D06 `paired_macroblocks=4/4/0/0`、`assigned_arm_runs=12/12/0/0`。另有 p 值阈值、摘要 hash、缺 SU(3)、封存 split、TV 超 alpha、缺 comparison route 等 fail-closed 负例。`tests/test_optimize_evidence_adapter.py` 为 **5 passed**。这些是合成 schema/摘要 fixtures，数字不是 E-chain 拟合、真实 D06 样本或生产 link 测量。

通用字节身份前置现在由 [`core/source_integrity.py`](../../core/source_integrity.py) 的 `verify_source_bytes` 提供，并由 `core/evidence.py` 的 `apply_evidence_gate_with_source_bytes` / bundle variant 在 metadata gate 前调用：只散列调用方明确映射、位于指定 root 下的相对路径；planned denominator 保留未映射/缺文件为 abstained、hash mismatch/越界路径为 failed。专项临时文件测试 **6 passed**，与 RunState gate 集成测试另计于 shared evidence suite。通过字节 gate 也最多到 `draft`；它只证明本次读到的字节与给定 hash 一致，不证明 hash 的授权来源、数据独立性，也不锁定文件后续不变。尚未接入 Optimize RunSpec/真实 source reader，本轮未打开任何物理输入。

**继续工作：**独立统计/物理 reviewer 需复核 D04 六项 fit gating、D06 paired state + pseudo-gap/TV union bound、D11 operator 维度/边界/舍入与 SU(3) 证据范围；软件侧将 source-byte gate 接到实际 RunSpec/source reader，并验证 receipt 指向的 bytes 与文件清单一一对应。随后仍需经过授权的真实 E-reference、paired solver/失败/成本 receipts、实际 `train.main` evaluator 接线。D04 reference、D06 Markov gap/TV 与真实 solver 输入均未提供；当前不训练 W、不测 GPU、不打开 sealed IDs，保持 `NOT_READY/UNCERTIFIED` 和 `physical_score=null`。所有后续报告继续保留 planned/valid/failed/abstained 分母，并提供图或文字数据流/决策流程。


### 2026-11-26 Optimize task adapters 的 source-byte 默认门

本次把 source-byte 检查从 shared helper 推进到三个 Optimize adapter 的实际 gate 调用点：D04/D11 单 envelope 接受 `source_root + source_paths`，D06 接受同一 root 下按 route 分开的 `source_paths_by_envelope`。默认 `require_source_bytes=True`；没有 root 或路径映射即进入相应 stage 的 `abstained`。即便调用方显式给出 `require_source_bytes=False`，也只有在 root 和 paths 都未提供时才走 metadata-only 分支；这是 synthetic fixture 的显式选择。若路径映射提供不完整，shared verifier 会按所有注册 source hashes 计计划文件，并将缺路径/文件记为 abstained、字节不匹配或越界路径记为 failed。全量字节一致后仍只到 `draft`。

文字流程：`RunReceipt + task envelope/source hashes -> task-specific contract checks -> explicit root/path map -> 每个 source hash 逐文件 SHA-256（planned = 注册的 source-hash 项数；valid/failed/abstained 按实际文件计） -> shared schema gate -> draft 或 abstained`。D04 的 E-chain/reference/fit/review/authorization 文件、D11 的 link/operator/kernel/SU(3)/review/authorization 文件、D06 三路的 score/chain/gap/TV/budget/cost/review/authorization 文件均须由调用方明确给出相对路径；不会由 adapter 猜文件位置。

新增三项默认拒绝测试分别覆盖 D04、D11、D06；缺路径时对应 source-file P/V/F/A 分别为 D04 `8/0/0/8`、D11 `7/0/0/7`、D06 `30/0/0/30`。另有 D06 三路 bundle 的正向字节映射，30/30 通过；篡改共用文件后为 `30/3/27/0` 并原子 abstain。Optimize adapter suite 为 **9 passed**，与 Wilson adapter suite、共享 evidence 和 source-integrity suite合计 **44 passed**；整个 checkout CPU suite **276 passed**，Optimize/Wilson test 文件组合 **202 passed**。这些 task-level source 文件均为临时 synthetic bytes，不是实际 source-map 或生产收据。任何完整 byte map 也仅证明本次读取字节与调用方提供 digest 一致，不验证 hash authority、reviewer 身份/独立性、物理内容或读取后的不可变性。

仍未把实际 RunSpec/source reader 接到这些参数，因此当前不能声明 receipt 与程序真正读取的输入文件一一对应。下一步软件工作是由 task evaluator 在同一读取事务中导出真实文件清单、hash 和相对路径映射，再对新增接线做正例、丢失文件、hash mismatch、root escape 和 receipt/reader 不一致测试。独立 reviewer 还需复核 D04/D06/D11 的物理与统计门；E-reference、paired solver、Markov gap/TV、SU(3) production links 与授权成本 receipts 未到位前，不运行真实训练或 GPU、不读 sealed split。`NOT_READY/UNCERTIFIED`、`physical_score=null` 保持不变。


### 2026-11-26 实际读入器目标盘点

对当前 PR #6 checkout 的只读代码检索确认：`tasks/optimize_gaussian/draft_plan.py` 只生成 `DRAFT_NO_LAUNCH` 的泛型 TaskSpec/RunSpec，`data={}`、`launch_allowed=false`；其内嵌说明明确指出本 checkout 没有 `tasks/optimize_gaussian/runspec_builder.py`、task worker 或 `train.main`。因此当前没有可供 source-byte adapter 接入的 Optimize production reader/evaluator。前述新 adapter 已让 gate API 默认 fail closed，但并未把 envelope 哈希映射绑定到真实输入读取过程。

文字诊断：`当前合同/草案 + 源码 hashes -> 空 data map -> no worker/runner -> no source files opened -> D04/D06/D11 证据 gates remain abstained`。下一软件阶段不是把任意路径传给 adapter，而是先恢复/实现经 review 的 Optimize evaluator 和 RunSpec builder，再让同一个 reader 产出输入 bytes、manifest、逐文件 hashes 与 adapter paths；以 CPU fixtures 覆盖 reader/gate 清单一致、缺文件、内容变化与失败 arm denominator。此盘点只读代码和文件清单，物理读取分母 `0/0/0/0`，没有训练/GPU 或 sealed 数据访问。

### 2026-11-26 同一 source snapshot 供 reader 与 evidence gate 复用

为消除“先 hash 文件、随后 evaluator 再打开路径”之间的竞态，shared [`core/source_integrity.py`](../../core/source_integrity.py) 新增 `capture_source_snapshot` / `SourceByteSnapshot.read_bytes` / `verify_source_snapshot`。它逐个读取显式映射文件一次，只保留匹配 caller digest 的不可变 `bytes`；默认总量上限 64 MiB，超限记 failed，不把大测量数组悄悄复制进内存。Optimize D04、D11 单 envelope 与 D06 三路 bundle gate 现在都可接收这个 snapshot；source verification 显示 `read_mode=immutable_snapshot`，bundle 仍一次性转移或 abstain。

文字流程：`expected source hashes + 显式 root/path -> capture each file once -> task reader parses snapshot.read_bytes(name) -> 同一 snapshot 交给 task gate 重算 byte hashes -> 全匹配最多 draft / 任一缺失、hash mismatch 或超限则 abstain/failed`。CPU 负控在 capture 后覆写原路径，D06 三路仍为 `30/30/0/0`；D04 与 D11 不完整 snapshot 分别 `8/0/0/8`、`7/0/0/7` 并 abstain。Optimize/Wilson adapter 与 shared evidence/source-integrity 聚焦测试 **52 passed**，PR #6 完整 CPU suite **284 passed**；均为合成 byte fixtures，未读真实输入。

范围边界：目前 adapter 仍接受独立传入的 summary，path-only 兼容入口也仍存在；故即使 snapshot gate 通过，也不能单独证明真实 Optimize parser 确实用 `snapshot.read_bytes` 计算了该 summary。此接口降低未来接线的 TOCTOU 风险，但没有补出当前 PR #6 checkout 缺少的生产 reader、D04 E-reference、D06 配对 solver receipts 或 D11 production links。源码调查发现 `fork/dianjun@54f0e97` 的另一个宽范围历史分支包含 `runspec_builder.py`、`worker_runspec.py`、`train.py` 等实现；它与当前 PR #6 分支大量分叉，未直接 cherry-pick，以免一并引入不相关的框架/任务状态。旧段落中“reader 已接到 train.main”的记录指向该分支历史，不代表当前 PR #6 HEAD。

下一步：在当前合同审阅后，按模块审计兼容依赖，只移植最小的 CPU RunSpec/source reader；reader 必须从同一 `SourceByteSnapshot` 解析它实际使用的 manifest、score matrix、solver receipts 和 operator inputs，并由 adapter 测试反向核对解析对象与 snapshot bytes。先补 content mismatch、路径映射/manifest 不一致、snapshot 64 MiB 边界、失败 arm 分母和 no-clobber 测试，再重跑完整 CPU suite。现无授权更改数据访问与 `NOT_READY/UNCERTIFIED`，`physical_score=null`，不读 sealed split、不运行 GPU/QUDA。

### 2026-11-26 D06 bounded-risk q-matrix snapshot reader

新增 [`source_reader.py`](source_reader.py)，给用户接受的 bounded block-risk estimand 增加一个有限的、可运行 CPU source reader：从 caller 已捕获的 `SourceByteSnapshot` 读取唯一命名 NPZ，不回退到 path；文件只接受精确 schema，split 必须显式是 calibration 且不能含 gap/validation/holdout/sealed/confirmation；宏块 ID 严格递增且唯一，arm 顺序固定为 trainable-W / same-budget SLSQP / degree-4 Chebyshev，时间顺序固定为 `(2,3,4,5)`。逐块调用既有 `bounded_block_scores`，只有已声明 solver failure 的 arm 可含非有限 q，且按合同计损失 1；其它坏 q 保留为失败 macroblock，不能静默删行。完整结果接入既有 bounded-risk reducer，并回报 source、macroblock 与 assigned-arm 三种分母。

文字流程：`显式 D06 calibration snapshot -> NPZ schema/split/ID/arm/time preflight -> q[block,arm,t] -> failure arm 记损失 1、其它无效块计 failed -> bounded-block reducer -> diagnostic only`。合成捕获后改写原文件仍读 snapshot：source `1/1/0/0`，macroblock `12/12/0/0`，assigned arm-run `36/35/1/0`；损坏非失败臂使分母变为 `12/11/1/0` 并整块弃权，错 arm 顺序也拒绝。`tests/test_optimize_snapshot_reader.py`、D06 adapter 与 bounded-risk tests 定向套件 **27 passed**。这些是合成 bytes；没有 E/D 物理输入、生产求解器或独立链读入。

本段记录 q-matrix reader 初版时的边界：当时它只读已生成矩阵，reader 的 reducer 结果尚未由 adapter 重算；这些软件项随后已由文末“D06 同快照 reader 到原子 evidence gate 的接线核验”完成。当前分支仍没有 production Optimize RunSpec/worker、D04 原始 E-reference evaluator 或 D11 production-link evaluator，因此此历史 reader 仍不能生成真实配对 solver outcomes、reference draws、成本表或物理 evidence。后续只可在 reviewed evaluator contract 下完成这些 reader/RunSpec 接线，并通过同源 snapshot fixtures 验证；真实 E-reference、paired solver/cost/gap-TV/SU(3) receipts、独立审阅和授权仍是生产门。保持 `NOT_READY/UNCERTIFIED`、`physical_score=null`，不读 sealed split、不运行 GPU/QUDA。

### 2026-11-26 D06 reader 与统计合同方法一致性修正

复核上述 reader 后发现原实现调用 i.i.d. `bounded_block_risk_decision` 的 empirical-Bernstein helper，而 D06 envelope/合同冻结的是 corrected Markov pseudo-gap Bernstein；即使两者对同一个 bounded score 计算，适用假设也不同。现已改为直接调用 [`bounded_block_risk_markov.py`](bounded_block_risk_markov.py) 的 `bounded_block_risk_markov_decision`，并将三 route envelope 的共同 `gamma_ps_lower`、`tv_bound_after_burn_in` 与 paired-macroblock burn-in 输入计算。方法名只能是合同规定的 `corrected_markov_pseudo_gap_bernstein`；依赖参数无效、缺失或三条 route 不一致会弃权。仅当输出为 `CONDITIONAL_MARKOV_PASS_BOTH_COMPARISONS`（正分母下界与两个 `r_star=10/9` 对比同时通过）才调用后续 byte/envelope gate；条件 no-release 或 gap/TV 不可用时不再推进到 draft。

reader 现在还从同一 snapshots 严格解析 `optimize-paired-macroblock-chain-receipt-v1`、`optimize-markov-gap-receipt-v1` 与 `optimize-initial-law-tv-receipt-v1` 三种 JSON 收据；要求 paired routes 捕获完全相同的 bytes，核对 chain/split/review hash、完整 block IDs、arm/time order、5-config block length、gamma、TV 和 burn-in，再把该 receipt denominator 单列为 planned/valid/failed/abstained。合成 positive 的 receipt 分母=`3/3/0/0`；改写且重哈希 gamma 收据而保留 envelope 数值的负控为 `3/2/1/0`，拒绝 draft。

测试正控为 synthetic `N=2,000`、`gamma_ps=1`、`TV=0.001` 的支持界算术，arm-run planned/valid/failed/abstained=`6000/5999/1/0`；负控覆盖共同风险下 effect no-release、三个 route 的 Markov 参数不一致和收据内容/envelope 不一致。reader/evidence adapter 聚焦套件 **20 passed**；本轮 PR #6 checkout 完整 CPU suite **295 passed**。上述测试不测链、不证明 gap/TV receipt 正确，也不证明 Markov 假设成立；独立 review hash 与 byte equality 不认证 receipt 的数学推导、签署者身份或独立性。文字流程：`q-matrix snapshot + chain/gap/TV JSON receipts -> bytes/schema/identity/denominator reconciliation -> contract-selected Markov support bound -> both effect LCBs + positive denominator -> conditional pass 才进入 byte/envelope draft；否则 ABSTAIN`。

本修正避免把 i.i.d. 结论误标成 Markov，但并未修复 receipt semantics：真实 gap/TV receipt 的 schema/parser、独立 reviewer、E-reference uncertainty、生产 paired solver/RunSpec 和 source-reader authorisation 仍待完成。D04 squared-risk-ratio 保持独立，真实项目继续 `NOT_READY/UNCERTIFIED`、`physical_score=null`、sealed splits closed，不运行 GPU/QUDA。

### 2026-11-26 历史 Optimize runner 的兼容性审计

只读检查历史对象 `fork/dianjun@54f0e97` 的 `runspec_builder.py`、`worker_runspec.py` 和 `train.py` 后，不将旧 runner 直接恢复到当前 PR #6。旧 RunSpec 是 `optgauss-worker-runspec-v2`，入口固定为 `python -m tasks.optimize_gaussian.train`；builder 按旧 `fit/dev_val/test` ID ranges、seed 列表、cache/validation manifests、`reference_bootstrap`、可选 `reference_path` 及 Pion propagator receipt 绑定整套文件。旧训练入口仍通过路径打开 protocol、gauge/cache、reference/bootstrap 并写 `results.json`、`manifest.json`、`data.npz`；worker 在启动前核对路径哈希并提供进程组 hard-timeout，但对训练子进程实际读取输入没有共享 `SourceByteSnapshot`，因此旧文件哈希 preflight 与 parser 读取之间仍存在内容竞态。

与当前合同不兼容的关键处：当前 D06 的输入单位是明确 calibration split 上的 paired macroblock `q[block, arm, t=(2,3,4,5)]`、三臂固定顺序、已声明失败臂记损失 1、corrected Markov pseudo-gap Bernstein 及同一 captured bytes 内的 gap/TV/burn-in 收据；旧 runner 的 `dev_val/test`、bootstrap-risk 与 learned-smear训练/测试路径没有实现这套同一 snapshot 的分母与 Markov gate。当前 D04 E-chain candidate-blind PP/A4P 联合 reference 和 D11 degree-4 SU(3) operator证据也没有由旧 RunSpec/训练器闭合。旧 `SOURCE_PATHS` 未绑定当前 `source_reader.py`、`bounded_block_risk_markov.py`、`evidence_adapter.py` 与研究合同。因此整段移植会造成“旧程序成功运行”被误读成满足新合同；只可借鉴其进程监督设计，不能复用旧 task protocol 或声称它是当前 evaluator。

文字流程：`历史 RunSpec hashes -> 路径式 train.main -> 旧 fit/dev_val/test 与 bootstrap objective` 与当前 `D06 immutable snapshot q-matrix + chain/gap/TV receipts -> Markov bound` 在读取事务、split 语义和统计 estimand 上不相接；本轮静态审计 planned/valid/failed/abstained=`3/3/0/0`（三个旧模块 inspected），物理文件读取=`0/0/0/0`。继续软件接线应实现新协议的 evaluator/parser并从同一 bytes snapshot 派生 q 矩阵、失败分母、receipt 摘要和证据摘要；旧 runner 仅作 hard-timeout 参考。不得为满足“有 runner”而重建旧 `train.py` 或访问真实/封存输入。

本次 Wilson reader 的全仓 CPU 回归为 **299 passed**；这验证软件行为，不解决 E-reference、真实 gap/TV receipt、paired solver/cost、D11 SU(3) 审阅、production RunSpec 或授权。状态仍为 `NOT_READY/UNCERTIFIED`、`physical_score=null`。

### 2026-11-26 immutable NPZ 的解压预算预检

`SourceByteSnapshot` 限制的是被捕获的压缩文件总字节，但 `.npz` 是 ZIP 容器，单靠压缩字节上限不能限制 NumPy 解包/分配的数组大小。新增 [`core/npz_safety.py`](../../core/npz_safety.py)：在 `numpy.load` 前检查 ZIP 成员数/重复名/路径、每个 NPY header 的 shape 与 dtype、header 推出的 payload 长度和总 uncompressed bytes；拒绝 object array，默认解压预算 64 MiB。当前 Optimize D06 q-matrix、Wilson C-calibration loop matrix 与 retrospective C-development snapshot parser 均先走此检查。超预算数组不进入 NumPy 分配路径。

文字流程：`压缩 snapshot bytes -> ZIP central-directory / NPY header 预检 -> 64 MiB 解压预算与实际成员长度相符 -> 才允许 numpy.load -> task-specific schema/hash/split gate`。新 synthetic regression 覆盖正常压缩 NPZ、低压缩率膨胀拒绝、header 声称巨大 shape 但 payload 不足、object dtype 和重复成员；Optimize reader 与 Wilson reader 的 immutable snapshot 路径仍复用同一 captured bytes。当前 PR #6 定向 source/Optimize/Wilson suite 为 **33 passed**，全仓 CPU suite **305 passed**。物理输入读取仍 `0/0/0/0`，没有读取 loop/GEVP/thermal 或 sealed data，也没有 GPU/QUDA。

限制：64 MiB 是 parser 的内存保护默认，不是物理数据完整性或来源证明；高于该预算的未来谱/相关函数输入应另做有界流式 reader 与显式审阅，不应调大上限绕过。JSON summary 和非 NPZ source 仍由各自 parser 负责；此改动不补 C/F production exporter、q/h/x 推导、D04 reference 或 Optimize solver receipts。所有 lane 继续 `NOT_READY/UNCERTIFIED`、`physical_score=null`。

### 2026-11-26 DRAFT 计划的读取依赖闭合

复核发现 `source_reader.py` 已 import `core.npz_safety`、`core.source_integrity`、bounded-risk reducer 与 Markov 界，但 no-launch draft 的 `source_hashes` 没有完整列出这些可改变读入/风险结果的模块，报告快照也没有随附全部依赖。现已扩展 `draft_plan.py` 的依赖表和报告快照，纳入 core I/O/receipt/spec/state/source-integrity/NPZ-safety，以及 bounded-risk/Markov、candidate、Chebyshev、risk-statistics 和 q-matrix reader。

文字流程：`D04/D06/D11 contract + reducer/reader/integrity/parser/core schema sources -> SHA-256 dependency map + source snapshots -> DRAFT_NO_LAUNCH`。新增报告测试逐项校验 artifact hashes，并确认 `core/npz_safety.py` 与 `core/source_integrity.py` 的 snapshot 存在；focused Optimize + snapshot + integrity suite **19 passed**，PR #6 全仓 CPU suite **306 passed**。这改善源码可追溯性，不产生新的 D06 scores、gap/TV receipts、D04 reference draws、SLSQP/learned-W 比较或 GPU evidence；planned physical gates 仍 `5/0/0/5`，sealed split 不读，`physical_score=null`。

### 2026-11-26 Optimize no-launch draft 绑定 learned-W 原型身份

复核 PR #6 的 `draft_plan.py` 时发现一个来源图缺项：TaskSpec 已声明 `trainable_W` 为主臂，却未在 `source_hashes` 或复制的报告源码快照中绑定其实现依赖。现将 [`trainable_w.py`](trainable_w.py) 与 [`features.py`](features.py) 同时加入哈希依赖表和 no-launch 报告快照，报告文案明确说明这只是 CPU coefficient/operator prototype，其 coefficient-MSE 目标不是选定物理 risk，也没有接到生产训练臂。随后又将改变 D04/D06/D11 gate 行为的 `evidence_adapter.py`、`core/evidence.py` 和 `core/receipts.py` 加入同一依赖图与快照。这样后续审阅者能区分“方法合同里列了 trainable-W”与“已绑定某个可审查的参数化和证据门版本”，同时避免把原型写成已完成的公平风险对照。

文字流程：`selected F=2 + D04/D06/D11 contract -> hash trainable-W/features + reducers/readers + task/shared evidence gates -> copy exact sources into DRAFT_NO_LAUNCH report -> no production objective/solver/data => all physical gates remain abstained`。`tests/test_optimize_draft_plan.py` **2 passed**，覆盖依赖哈希键与生成报告快照；包含 gate-dependency 补丁后的完整仓库 CPU suite **315 passed**、Wilson 定向 suite **88 passed**。这一改动不运行模型、不生成新 paired score，不改变同预算 SLSQP 和 degree-4 Chebyshev 两项对照，不打开 sealed IDs，不启动 GPU/QUDA；D04/D06/D11 与 physical score 仍未就绪。

### 2026-11-26 D06 同快照 reader 到原子 evidence gate 的接线核验

前面“reader 结果还未被 adapter 用来重算/比对 envelope 统计字段”的软件缺口现已在当前分支闭合：`apply_bounded_block_risk_snapshot_gate` 从三个 route 的同一 captured NPZ bytes 重新计算 paired block scores、Markov support bounds 和 arm-failure 分母；同一快照里的 chain/gap/TV JSON receipts 会逐字节跨 route 对齐，再检查 chain/split/review identity、全部 block IDs、固定 arm/time 顺序、block length、gamma、TV 与 burn-in。receipt macroblock/arm-run 分母必须与解析矩阵一致，且 corrected Markov pseudo-gap Bernstein 的 trainable-W 下界、两个既定对照下界和正分母门全部通过后，才调用同一 snapshots 上的 source-byte bundle gate。输入损坏、分母错配、任何条件界不足或 source byte 缺失均进入 `RunState.abstained`；成功 synthetic fixture 也只到 `draft`。

文字流程：`同一 score-matrix + chain/gap/TV snapshot -> NPZ/schema/split/ID/arm/time 检查 -> 三臂 bounded scores 与 Markov 界重算 -> receipts/分母逐项交叉核对 -> 同一 snapshots 原子 source-byte gate -> draft 或 ABSTAIN`。集成正控为 **2,000** synthetic paired macroblocks：每 route reader=`1/1/0/0`，每 route blocks=`2000/2000/0/0`，assigned arm-runs=`6000/5999/1/0`，Markov receipts=`3/3/0/0`，source files=`30/30/0/0`；其中一个预先声明失败的 SLSQP arm 仍按 loss 1 计入完整分母。错 receipt denominator、route-specific gamma、gamma receipt 与 envelope 不一致、以及无法越过效应门的负控均 fail closed。当前 checkout 定向 `tests/test_optimize_evidence_adapter.py tests/test_optimize_snapshot_reader.py` **21 passed**。这些值全部来自合成 NPZ/JSON；`DRAFT_PENDING_TASK_VALIDATION`、`certified=false`、`physical_score=null`。

这项接线取代本 TODO 较早的“reader 输出尚未由 adapter 重算”段落，但不等于真实 Optimize evaluator 已接通：当前分支仍无 production RunSpec builder/worker 或训练入口；D04 summary、D11 operator summary 仍是调用方传入且没有基于生产原始输入的同快照 evaluator。下一步只可在独立审阅后的新 evaluator 协议下补这些 parser/receipt 接线；真实 E-reference、paired solver/cost、gap/TV 与 SU(3) production receipts 以及数据/GPU 授权未到前，不训练、不启动 GPU、不读 sealed splits。trainable-W vs 同预算 SLSQP 与 trainable-W vs degree-4 Chebyshev 两项比较继续并列保留。

### 2026-11-26 D04/D11 summary 的同快照解析与 digest 自引用修正

本轮推进了上一段遗留的软件边界：D04 `reference_summary` 与 D11 `operator_summary` 现在可由 gate 直接从传入的 `SourceByteSnapshot` 读取，不再信任旁路传入的 summary 对象；若调用者同时传入的 summary 与捕获 bytes 解析结果不一致，gate 会 abstain。新 JSON reader 拒绝重复 key、NaN/Inf 和非 object 根。另修正一个此前无法满足的合同：summary 不可能在自身文件中保存“等于完整文件 SHA-256”的字段，因为添加该字段会改变文件 hash。现改用 `summary_content_sha256`，对去掉此字段后的 canonical JSON 内容计算；源文件的完整 SHA-256 仍独立由 source map 和 immutable snapshot 检查。

文字流程：`同一 SourceByteSnapshot -> strict JSON decode -> 读取 D04/D11 summary -> canonical content digest/schema 检查 -> task-specific fit/operator gate -> 原始文件 SHA-256 gate -> draft 或 abstain`。真实合成快照中 D04 source files `8/8/0/0`、D11 `7/7/0/0`，均只进入 `DRAFT_PENDING_TASK_VALIDATION`；缺文件仍 `ABSTAIN_SOURCE_BYTES`，summary 冲突或内容 digest 错误 fail closed。定向 `tests/test_optimize_evidence_adapter.py tests/test_optimize_snapshot_reader.py` **27 passed**；Optimize 全集 **150 passed**；PR #6 全仓 CPU suite **344 passed**。这些是 synthetic JSON/source fixture，summary content hash 不证明签发者身份或物理内容。

与这次 reader 代码同步生成了新的[data-free RunSpec 草案报告 v7](../../results/optimize_gaussian/draft_plan_20261126_v7/README.md)：20 个 artifact/source hashes 全匹配，schema plan=`1/1/0/0`，physical gates=`5/0/0/5`，data inputs=`0/0/0/0`，`launch_allowed=false`。报告的文字流程把 D04/D11 summary-byte parser 标为 metadata 接线，并明确没有原始 correlator fit 或生产 link/operator evaluator。

此项关闭的是 D04/D11 summary 的路径旁路与自哈希不可能性，不等于实际 evaluator：manifest/envelope/run receipt 仍未由当前 production RunSpec 从读取事务生成；没有真实 E-chain correlator parser/六种 joint fit、D11 production SU(3) link/operator evaluator、独立审阅或 source authorization。下一步仍需按 reviewed contract 实现真实同快照 evaluator，并用输入清单一致、content mismatch、缺文件、hash mismatch、root escape 与 denominator 边界 fixtures 验证；在真实 E-reference、paired solver/cost、Markov gap/TV、SU(3) receipt 与授权到位前，不读 sealed IDs、不启动 GPU，保持 `NOT_READY/UNCERTIFIED`、`physical_score=null`。两项 Optimize 比较仍并列为 trainable-W vs 同预算 SLSQP 和 trainable-W vs degree-4 Chebyshev。

### 2026-11-26 GEVP diagnostic 之前的跨 PR 兼容性复核

在 PR #6 文档收尾提交 `a838eda` 上重新运行 `PYTHONPATH=. pytest -q`：**317 passed**（完整 CPU suite）。另将 PR #5 当前 tracked HEAD 放入临时归档，仅覆盖成 PR #6 的 `core/specs.py` 后运行完整 Pion suite：**257 passed**；临时目录已删除，Pion 工作树未改写。两个分支选定共享模块中，`core/evidence.py`、`core/receipts.py`、`core/source_integrity.py`、`core/npz_safety.py` 与 evidence/receipt tests 内容逐字一致；`core/state.py` 的差异仅为相同状态转移集合的排版；`core/specs.py` 的差异是 PR #6 新增 split-range 校验与 Optimize operator-contract 名称，Pion overlay suite 验证了向后兼容。`test_source_integrity.py` 的跨分支差异仅是 import/空行排版。

这只降低当前 shared TaskSpec/gate 的代码整合风险，不代表 PR 已合并或 GitHub 报告 mergeable；PR #5/#6 当前均无 checks，mergeability 为 `UNKNOWN`。此次 suite 是 CPU 软件回归，没有读取真实 E/D/solver/operator 输入，也没有改变任何物理 gate；生产 RunSpec/evaluator、独立 review、授权 receipts 和 `physical_score` 仍未完成。

### 2026-11-26 PR #6 后续全仓回归快照

在新增 Wilson 固定 `L={6,8,10}` MBB 敏感度图文包后，PR #6 当前提交 `773468acf1f1ce6fce824dd8aa207383a3a7d3e1` 的全仓 CPU suite 为 **419 passed**；Optimize 源文件与方法合同未在该提交中改变。该回归不关闭本 TODO 中 D04 E-chain evaluator/reference、D06 获准 paired solver/gap-TV receipts、D11 production SU(3) link/operator evaluator/独立审阅等事项；仍保持两项既定比较、`NOT_READY/UNCERTIFIED`、`physical_score=null` 与 sealed data closed。

### 2026-11-27 D04 reader 来源指纹和报告快照一一对应

检查 no-launch 草案报告时发现 `d04_raw_reference.py` 已提供同快照的 raw PP/A4P parser 与六个拟合变体，但此前不在 `TaskSpec.dependencies` 哈希表/报告快照中；并且 source hash map 与复制到报告中的文件清单由两份独立映射维护，`contract.py` 与 `kernels.py` 也有已登记 hash、未保留 snapshot 的缺口。现用一个 `_source_paths()` 作为哈希和快照的共同清单，报告构建前从同一映射捕获所有依赖 bytes，并检查每个捕获字节的 SHA-256 与计划 hash 一致；不匹配时在创建输出目录前停止。D04 raw reader 加入该清单，报告正文与机器可读 `text_flow` 标明它只输出 fit diagnostic，不产生 dependence-aware error 或 gate-ready reference。

新版[no-launch 草案报告 v8](../../results/optimize_gaussian/draft_plan_20261127_v8/README.md)包含 source dependencies/snapshots `21/21` 一一对应，artifact hashes `23/23` 匹配；schema plan `1/1/0/0`，物理 stages `5/0/0/5`，data inputs `0/0/0/0`，`launch_allowed=false`。`tests/test_optimize_draft_plan.py tests/test_optimize_d04_raw_reference.py` **9 passed**；本轮 PR #6 全仓 CPU suite **436 passed**，`git diff --check` 通过。新增验证只确保哈希清单与随附源码快照一致，不证明 D04 数据来源、fit coverage 或 E-chain 参考有效。

下一步保持范围在同快照 D04 evaluator：review fit window/channel/sign conventions，补依赖感知 mass error、PP/A4P/window/model 的 paired-SE shifts 与完整拟合/失败 summary，再由 adapter 从同一 source bytes 检查 summary 与 receipt；需要独立 reviewer 和获准的非封存 E-chain 输入后才能尝试 reference gate。没有这些输入时保留 D04 `NOT_READY/UNCERTIFIED`，不读 sealed IDs、不启动 GPU，两个 D06 对照与 bounded block-risk estimand 不变。

### 2026-11-27 D04 条件性 moving-block bootstrap 与成对拟合位移

为推进上节明确的软件缺口，`d04_raw_reference.py` 现在可选择从同一个 `SourceByteSnapshot` 做 circular moving-block bootstrap；reader 将配置顺序作为输入，但不会把严格递增 ID 当成独立性证明。拟合仍用共享全协方差、固定 PP-even/A4P-odd 符号和三组确定性 multistart。每个重抽样本对六个合同 joint fits、PP-only、A4P-only 共八个拟合使用相同配置索引；成对位移按同一 replicate ordinal 计算，任一拟合失败就记在 pair denominator，未按有效项重新排序。主质量经验 percentile interval 至少要求 90% draw 有效，否则区间级 abstain；`L=1` 是 iid 重抽基线，`L=4,8` 是本轮预选的敏感度值，不是从链估出的 decorrelation scale。

新增[`d04_block_bootstrap_diagnostic.py`](d04_block_bootstrap_diagnostic.py)与[固定 synthetic AR(1) 输入的图文报告](d04_mbb_cpu_20261127_v1/README.md)。输入为 240 个合成两态 correlator rows，真值 `a_t m_0=0.32`、`a_t m_1=0.91`，振幅扰动来自 `rho=0.65` stationary AR(1)；每个 `L=1,4,8` 做 24 draws。三种长度的 primary joint mass interval 分别为 `[0.3199794,0.3199983]`、`[0.3199798,0.3200014]`、`[0.3199821,0.3199964]`，bootstrap SE 为 `5.72e-6, 6.24e-6, 4.41e-6`；每个长度主 fit 为 `24/24/0/0`。除 `L=1, tmin=2, nstates=3` 外，各 channel/window/model paired comparison 均为 `24/24/0/0`；该 fit 只有 `21/24` 个成功 draw，区间按冻结的 90% reporting rule 记为 `1/0/0/1` abstain。所有可报告对照中最大 `|point shift|/paired SE=2.04`（`L=8, tmin=2, nstates=2`），低于合同里的 2.5 *诊断门*，但不能消掉另一个模型的失败/弃权；该 toy 点值结果不是实际 reference pass，也不测试多数据集覆盖率。图左画主质量 percentile interval 与注入真值，右画全部七个成对对照并以 x 标识弃权；完整每项 P/V/F/A 在 `summary.json` / `summary.csv`。

文字流程：`same snapshot bytes -> PP/A4P folds + shared covariance -> L=1/4/8 circular block draws -> same draw refit joint/PP-only/A4P-only/window/model -> fit/pair failures remain in denominator -> interval and paired-SE plot -> diagnostic-only`。专项 [`tests/test_optimize_d04_raw_reference.py`](../../tests/test_optimize_d04_raw_reference.py) **9 passed**；artifact hashes `8/8`、reader/generator/test/contract source hashes `4/4` 匹配，report 的 derived-only refresh 先验核验旧 hashes 且不重拟合；`py_compile` 与 `git diff --check` 通过。该 pilot 的 block lengths 不能认证真实 E-chain stationarity/dependence、percentile interval coverage、channel conventions 或 reviewer/source authority。仍须独立复核 fold/sign/channel 与 resampling 定义，并从获准、未封存、candidate-blind E-chain snapshot 得到 provenance/dependence/reference receipts；没有这些证据不计算 gate-ready D04 summary，保持 `NOT_READY/UNCERTIFIED`、`physical_score=null`，不读 sealed split、不运行 GPU/QUDA。

### 2026-11-27 D04 fold、协方差与成对重抽样源码自审

对照当前合同和 `d04_raw_reference.py` 做了方法实现自审：PP 用 `0.5*(C(t)+C(16-t))`，A4P 用 `0.5*(C(t)-C(16-t))`，A4P 中点因反对称性固定为零且不进入拟合；全部 fit variants 共用由原始折叠样本构成的 `t=2..8` 联合协方差，含 25% diagonal shrinkage 与相对特征值 floor `1e-10`。每个 MBB draw 都用同一组配置索引重新计算折叠量、均值和协方差，再拟合六个 joint variants 与 PP-only/A4P-only；配对位移按 draw ordinal 对齐，任一侧拟合失败就不进入该 paired interval，失败仍留在分母。

文字流程：`raw PP/A4P rows -> even/odd fold -> shared covariance -> one ordered circular-MBB index set per draw -> 8 fits on that same draw -> paired shift or retained fit failure -> 90% valid-draw reporting threshold -> diagnostic only`。代码中 chi-square p 值仍是近似 GLS 诊断；percentile MBB 区间未做有限样本覆盖校准。自审也不能决定 exporter 的 A4P source/sink 符号、轨迹顺序是否构成平稳链、L={4,8} 是否覆盖依赖尺度或 circular wrap 是否合适。这里没有新增/重跑测试，也没有独立 reviewer；它确认代码与本地拟定合同的对应关系，不确认物理约定。

因此，本项关闭的是“明确代码实际折叠/重抽了什么”的审计缺口，不是 D04 reference gate。继续所需的最小外部包仍是获准、未封存、candidate-blind E-chain 原始 PP/A4P bytes、来源和依赖收据、production exporter/source-sink 约定，以及独立统计/物理 review。拿到前不生成 gate-ready summary、不打开 sealed split、不训练或运行 GPU；D04/D06/D11 保持 `NOT_READY/UNCERTIFIED`、`physical_score=null`。

### 2026-11-27 shared NPZ 解析失败归一化与整仓复核

NPZ 共享预检器现把 ZIP 不支持的压缩方式、加密成员和损坏 DEFLATE 流产生的解码异常归一为 `ValueError`。Optimize 的 D04/D06 readers 因而沿用已有的异常处理，进入其明确的 failed/abstained 记录，而不是把格式异常作为未捕获异常漏出 task adapter。source snapshot 也按剩余聚合预算读取，最多多读 1 字节用于超限判定。流程：`captured NPZ bytes -> member/header/size preflight -> ZIP/codec parse error -> ValueError -> task-specific fail-closed denominator`；`source root -> bounded snapshot read -> same captured bytes -> reader + evidence gate`。`tests/test_source_integrity.py` **17 passed**；PR #6 修正历史报告快照测试后整仓 CPU suite **447 passed in 72.47 s**。这验证解析失败和读取预算的软件行为，不验证 reader 输入的来源或物理内容；没有读取 E/solver/operator receipts，D04/D06/D11 仍 `NOT_READY/UNCERTIFIED`、`physical_score=null`，不启动 GPU/QUDA。

### 2026-11-27 Optimize D04 报告补齐共享 snapshot helper 身份

跨任务 source-map audit 发现，D04 MBB generator 使用 `capture_source_snapshot`，其 raw reader 还使用共享 NPZ 预检，但当时 `_SOURCE_PATHS` 都没有绑定这两个实现。现将 `core/source_integrity.py` 与 `core/npz_safety.py` 加入 report source map，并在 README 明确它们属于生成报告所用源码。回归用例确认 source map 指向当前 tracked helpers；全仓套件覆盖此测试。

文字流程：`D04 raw reader + MBB generator + contract + test + SourceByteSnapshot + NPZ preflight helpers -> source SHA map -> report README -> artifact hashes`。审计只修正未来报告的源码依赖闭包；没有重建已有 D04 MBB 产物或改写其历史 hashes。PR #6 全仓 CPU suite **448 passed in 72.46 s**。该源码身份记录不验证 E-reference 来源、D04 拟合约定、MBB 覆盖率或 candidate-blindness；真实 D04/D06/D11 gates 仍 `NOT_READY/UNCERTIFIED`、`physical_score=null`，sealed inputs closed，不运行 GPU/QUDA。

### 2026-11-27 D04 近似 chi-square GOF 的有限样本校准诊断

在已选的 `tmin=3`、PP-even/A4P-odd 联合两态 fit（13 nominal dof）上，新增 CPU 合成 pilot，检查当前 `gof_p_approx` 是否在 null 下接近均匀、检验 covariance shrinkage，并测试漏掉第三态及配置序列依赖/厚尾。共 16 个 fit cells，每格计划 300 个 fit；由于 covariance 对照复用同一批原始样本，实际为 2400 个独立生成的 configuration panels、4800 行 fit 结果，所有 fit 均留在计划分母。前 10 格包括 Ncfg=64/128 Gaussian null、Ncfg=128 同样本 covariance alpha 扫描、mild/strong omitted-state 及其 full-covariance 配对。新增 6 格比较 iid Gaussian、rho=0.30 AR(1) Gaussian、iid Student-t(ν=5)、rho=0.30 AR(1) Student-t(ν=5)；后两种 AR(1) 过程各有 estimated-covariance 与 oracle long-run covariance 配对。每个配对 panel 检查 SHA-256 相同。单配置特征仍为 27 维，注入 temporal-feature correlation `0.65`、channel correlation `0.2`、相对噪声 `0.03`；不读取生产数据。

结果摘要：

- null `Ncfg=128` 的 estimated+25%-shrinkage p 值均值/中位数为 `0.761/0.848`，`p<0.05` 为 `0/300`（Wilson 95% 上界 `0.0126`），经验 CDF 明显低于均匀参考；`Ncfg=64` 也是保守方向（均值 `0.746`，`p<0.05` 为 `2/300`）。
- 同样的 `Ncfg=128` null 样本下，known-covariance shrinkage alpha 扫描为：

  | covariance metric | mean p | median p | `p<0.05` | Wilson 95% interval |
  | --- | ---: | ---: | ---: | ---: |
  | known full, `α=0` | 0.519 | 0.507 | 12/300 = 0.040 | [0.023, 0.069] |
  | known diagonal shrinkage, `α=0.05` | 0.595 | 0.620 | 6/300 = 0.020 | [0.009, 0.043] |
  | known diagonal shrinkage, `α=0.10` | 0.656 | 0.699 | 3/300 = 0.010 | [0.003, 0.029] |
  | known diagonal shrinkage, `α=0.25` | 0.783 | 0.856 | 0/300 = 0 | [0, 0.013] |
  | known diagonal shrinkage, `α=0.50` | 0.889 | 0.947 | 0/300 = 0 | [0, 0.013] |

  同样本配对比较中，alpha 从 `0.05→0.10` 和 `0.25→0.50` 时没有一例出现更高 alpha p 值更低（`fraction_right_pvalue_lower=0`）；平均 p 位移分别为 `+0.062` 和 `+0.106`。estimated covariance 再加合同当前 `α=0.25` 时，平均 p 为 `0.761`、`p<0.05=0/300`；其保守方向与 known `α=0.25` 一致。该 DGP 下的受控结果支持“收缩 metric 会使近似 p 值更保守”这一机制解释，但不同 alpha 未作多重性校正，不能据此挑选一个校准阈值或生产 alpha；拟合有限样本及 covariance estimation 仍可能共同作用。
- strong 三态遗漏的 estimated+shrinkage fit 只有 `2/300=0.0067` 达到 `p<0.05`（CI `[0.0018,0.0240]`）；对同一批 300 个样本用 known full covariance 后为 `24/300=0.080`（CI `[0.054,0.116]`），每个有效 panel 的 oracle p 值都低于对应 estimated+shrinkage p 值，平均差（oracle−estimated）`−0.241`。这表明本合成设定下 power 对协方差度量敏感；alternative 是探索性强压力格，不是预注册确认检验。
- 配置序列相关/厚尾压力格显示，按 iid 规则用 `sample covariance / Ncfg` 会在正相关下严重低估均值方差。`Ncfg=128, rho=0.30` 的精确 AR(1) 有限样本膨胀因子为 `1.8476`。p 值均值和 `p<0.05` 拒绝率为：iid Gaussian estimated covariance `0.738 / 0/300`；AR(1) Gaussian estimated `0.314 / 65/300=0.217`；iid Student-t(5) estimated `0.769 / 0/300`；AR(1) t(5) estimated `0.323 / 49/300=0.163`。因此相同的 shrinkage 方案可从 iid 的保守行为变成有明显过拒绝的行为，具体取决于链相关。
- 同一 AR(1) 样本改用已知精确 long-run covariance（`Var(mean)=Σ/Ncfg × 1.8476`）后，Gaussian 的 `p<0.05` 为 `19/300=0.063`（Wilson 95% CI `[0.041,0.097]`），Student-t(5) 为 `20/300=0.067`（CI `[0.044,0.101]`）。两者比 estimated-iid-covariance 的 `21.7%/16.3%` 接近标称 5%，但不是生产估计方法；1%尾部仍粗，其中 AR(1) t(5) oracle 有 `7/300=2.3%` 达到 `p<0.01`。这组配对证据支持“需要 dependence-aware mean covariance/误差”的方法方向，却不能认证当前 bounded block-risk receipt、Markov gap/TV 或任何真实 HMC chain。

图文报告：[D04 GOF 校准图与完整分母/配对摘要](d04_gof_calibration_cpu_20261127_v1/README.md)。第一张图含 covariance-alpha null ECDF、strong-alternative 同样本配对散点和前十格拒绝率；第二张图单列六个 iid/AR(1)/Student-t stress cells 的 p-value ECDF 与 Wilson 拒绝区间。流程为：`固定 generator / seeds -> 2400 个 raw panel samples -> covariance / AR(1) / Student-t paired panels -> PP-even/A4P-odd + 三组确定性 multistart 两态 fit -> 13-dof approximate chi-square p -> 完整 4800-fit 分母 / ECDF / Wilson / paired contrasts -> diagnostic-only`。本报告完整记录 per-cell seed map 与 4800 行 panel-fit 结果；artifact hashes `13/13` 与 6 个 source snapshot hashes `6/6` 一致。

解释边界：每个阈值在 `n=300` 下的 MCSE 是 `0.00574`（1%）和 `0.01258`（5%），因此 1% 尾部尤其粗；DKW 带是单格描述参考，图上多格比较不是多重性控制。已知 stationary covariance 与 exact finite-N long-run covariance 是 generator oracle，不是可部署估计器；只有一个 `rho=0.30`、一个 ν=5、一个 Gaussian feature covariance 和当前 optimizer/fit 流程，未覆盖真实 HMC chain、非平稳/多时间尺度依赖、其他 tail weight、窗口/态数选择或真实 exporter 约定。不能将相同 panels 用于调整阈值后再宣称独立通过。

下一步对更宽的 rho/ν 网格做独立预冻结 null calibration，并为 covariance-of-mean 与 block/HAC 估计建立独立训练/校准/验证 panel；比较 approximate chi-square 和每个 parametric-bootstrap panel 都重估 covariance/spectrum 的 p 值，预先冻结窗口/模型选择处理、校准误差与尾部区间，并保留完整失败分母。由独立统计审阅者复核 fold/sign、long-run covariance 与 nominal dof。合成压力结果不能替代获准、candidate-blind E-chain snapshot、来源/依赖 receipts 和独立 review；在这些输入到位前维持 D04 `NOT_READY/UNCERTIFIED`、`physical_score=null`，不读取 sealed split、不启动 GPU/QUDA。

### 2026-11-27 D04 iid Gaussian nested parametric-bootstrap pilot

为具体推进上一节提出的校准候选，新增独立脚本 [`d04_parametric_bootstrap_diagnostic.py`](d04_parametric_bootstrap_diagnostic.py) 与[图文报告](d04_parametric_bootstrap_cpu_20261127_v1/README.md)。预先固定 60 个外层 iid Gaussian 两态 null panels、每 panel `Ncfg=128` 和 199 个内层 parametric replicates。先按当前 D04 `tmin=3` 联合 fit 拟合两态质量，再在固定质量下用同一 GLS covariance 解振幅以重建完整 27 维 fitted null mean；外层原始 panel 的未收缩 covariance 用作 plug-in generator，最小特征值 floor 为最大 eigenvalue 的 `1e-12`。每个内层 panel 重新做 PP-even/A4P-odd folding、估计 covariance、25% diagonal shrinkage、eigenvalue floor 与三起点谱拟合。当前 runner 的保守上尾 p 值为 `(1 + #{成功 draws 中 χ²_boot >= χ²_observed} + 失败 refits)/(planned draws + 1)`；至少 180/199 个 refit 成功才报告，否则该 panel abstain。原 v1 报告的 11,940 次内层 refit 全部成功，因此本次失败计为超越的修正不改变该历史报告的数值。

所有外层结果和内层 refit 均有效：approximate-chi-square outer denominator `60/60/0/0`，bootstrap outer p-value denominator `60/60/0/0`，bootstrap draw denominator `11940/11940/0/0`（均为 P/V/F/A）。当前这 60 个 outer panels 下，approximate chi-square p 值均值/中位数为 `0.752/0.839`，KS distance-to-uniform 为 `0.419`，`p<0.05` 是 `1/60=0.0167`（Wilson 95% CI `[0.003,0.089]`）；parametric-bootstrap p 值均值/中位数为 `0.485/0.530`，KS distance 为 `0.065`，`p<0.05` 是 `3/60=0.050`（Wilson 95% CI `[0.017,0.137]`）。在相同 60 个 outer panels 上，bootstrap p 值每例均低于 approximate chi-square p 值，平均配对差（bootstrap−approximate）`−0.268`。图左显示 bootstrap ECDF 接近 uniform reference、而 approximate chi-square curve 偏向大 p 值；图右给出 1%/5% size 与 Wilson 区间。[图表](d04_parametric_bootstrap_cpu_20261127_v1/bootstrap_calibration.png)

解读边界：外层 `n=60` 的拒绝率 MCSE 为 `0.0128`（1%）和 `0.0281`（5%）；内层 `B=199` 的 p-value 网格步长为 `0.005`，还存在嵌套模拟噪声。5% bootstrap 拒绝率的区间很宽，因此本结果只是证明“重拟合 covariance 与 spectrum 的 plug-in bootstrap 候选”值得继续校准，不是有限样本 size/coverage 认证。当前 pilot 仅 iid Gaussian；plug-in raw covariance、拟合均值重建和 multistart 规则均需独立审阅，AR(1)/厚尾/拟合选择尚未进入 bootstrap。此报告的 14/14 artifact hashes 与 7/7 source snapshot hashes 匹配；`physical_score=null`，D04 reference gate 仍 abstained。

下一步以更大的独立外层 panel set 比较 approximate chi-square 与 parametric bootstrap，并把 bootstrap 纳入先前的 rho/ν stress grid；冻结 independent calibration 与 untouched validation split，预先登记 p-value 失败分母及多重性处理，再由独立统计/物理 reviewer 检查拟合均值 reconstruction 和 covariance plug-in 方案。没有获准、candidate-blind E-chain 与来源/依赖收据时继续保持 `NOT_READY/UNCERTIFIED`，不将 synthetic bootstrap 写成物理 reference、不打开 sealed split、不启动 GPU/QUDA。

### 2026-11-27 D04 AR(1) null 下的依赖感知 bootstrap 复核

为检验上一节 iid parametric bootstrap 是否能面对已知配置序列相关，新增 [`d04_ar1_bootstrap_diagnostic.py`](d04_ar1_bootstrap_diagnostic.py) 和[完整图文报告](d04_ar1_bootstrap_cpu_20261127_v1/README.md)。冻结 `Ncfg=128`、外层 `160` 个独立 stationary Gaussian AR(1) 两态 null panels、真值 `rho=0.30`、每外层 panel/每 bootstrap 方法 `B=199`；相同 panel 配对比较当前 approximate chi-square、iid plug-in bootstrap、以及按该 panel 的 27 个原始 feature series pooled lag-1 OLS 估计 rho 的 AR(1) plug-in bootstrap。两 bootstrap 方法使用同一外层 fit 得到的 fitted null mean 与未收缩样本协方差；每个内层 replicate 都重做 fold、25% diagonal covariance shrinkage、特征值 floor 与三起点两态谱 fit。`Var(mean)` 对 iid 估计的精确 AR(1) 膨胀因子为 `1.8475765`。

结果分母全部闭合且无失败/abstention：outer fit `160/160/0/0`；rho estimate `160/160/0/0`；两 bootstrap outer p-value 各 `160/160/0/0`；每臂内层 draw 各 `31840/31840/0/0`（P/V/F/A）。SHA-256 样本 hash 有 160 个唯一值；报告 artifact hashes `13/13`、source snapshot hashes `6/6` 全部匹配。旧 v1 结果中的 31,840 个 draws/arm 全部成功，因此新加的失败最不利计数规则不改其 p 值或拒绝率。5% rejection 的配对结果是：

| p-value method | rejection | Wilson 95% CI | empirical p-value KS distance | mean p |
| --- | ---: | ---: | ---: | ---: |
| approximate chi-square | `32/160 = 20.0%` | `[14.5%, 26.9%]` | `0.246` | `0.349` |
| iid parametric bootstrap | `72/160 = 45.0%` | `[37.5%, 52.7%]` | `0.564` | `0.146` |
| estimated-AR(1) parametric bootstrap | `10/160 = 6.25%` | `[3.4%, 11.1%]` | `0.093` | `0.463` |

在 1% threshold，三者 rejection 分别是 `11/160=6.88%`、`33/160=20.63%`、`2/160=1.25%`；AR(1) bootstrap Wilson 95% CI 为 `[0.34%,4.44%]`，但尾部区间仍很宽。rho-hat 均值/中位数是 `0.286/0.288`，均值 bias `−0.014`，标准差 `0.041`，5–95% 分位 `[0.225,0.351]`。图文诊断把三种 p-value 的经验 CDF、1%/5% Wilson 区间以及 rho recovery histogram 并列展示：[AR(1) bootstrap 图](d04_ar1_bootstrap_cpu_20261127_v1/ar1_bootstrap_calibration.png)。文字流程：`独立 AR(1) null panel -> paired outer fit -> iid / pooled-rho AR(1) parametric draws -> 每 draw 重估 covariance 并重 fit -> p-value / failure denominator / Wilson + ECDF -> synthetic-only report`。

**诊断结论：**该单一 Gaussian `rho=0.30` 合成机制下，朴素 iid bootstrap 破坏了原始序列相关，误拒率比 approximate chi-square 更高；估计 rho 的 AR(1) bootstrap 把 5% rejection 从 `45%` 降到 `6.25%`，其 Wilson 区间包含标称 5%，且 p-value KS distance 从 `0.564` 降至 `0.093`。这是对依赖感知校准方向的支持，不是校准认证：外层仅 160 例，5% MCSE 为 `0.0172`；1% 更粗；rho、创新分布、特征协方差、stationarity、fit window 都只有一个设定。 pooled rho、plug-in raw covariance 与 fitted-mean reconstruction 尚未独立 review，图中的单方法 DKW 带也不作跨方法多重性校正。报告运行约 35 分钟，后续多格 refit 应按预冻结 cell 分批并提供运行进度，不能在看过结果后减少/改动分母。

**接手的下一步：**先由独立审阅者逐式检查 pooled lag-1 OLS 的适用假设、AR(1) bootstrap 的 stationary initialization、raw marginal covariance plug-in、fitted mean reconstruction 和 p-value ties/失败口径；审阅结论须附一页文字流程或图并逐项指向源码。之后预注册一个小而有判别力的 held-out 网格（至少 `rho={0.0,0.3,0.6}`，另含厚尾/创新分布 stress），同 panel 配对比较 approximate、iid bootstrap、estimated-AR(1)、circular-block/HAC bootstrap；校准/方法选择 panels 与 untouched validation panels 必须用不同 seed manifest。先用较小 screening cell 估算每 cell 成本，再冻结确认性样本数和 1%/5% MC 误差目标；所有失败、abstain 和 MC 区间完整保留。若 AR(1) 在非 AR(1) 或多尺度压力下偏离，报告适用边界，优先比较 block/HAC 而不是增加 rho 调参。没有授权的 candidate-blind E-chain bytes、来源/依赖 receipts、production source/sink convention 和独立统计/物理 review 前，不生成 gate-ready D04 reference，不读 sealed split、不运行 GPU/QUDA，继续 `NOT_READY/UNCERTIFIED`、`physical_score=null`。

### 2026-11-27 D04 bootstrap 失败值的保守上尾计入

复核上面 iid/AR(1) nested-bootstrap 实现后发现：旧公式把 p 值分母设为 `valid+1`，只要成功 refit 达到 `90%` 门槛就丢开其余失败值；虽然失败数在完整分母另有记录，但这会条件化在“拟合成功”，且不能声称是 planned-draw 下的完整上尾估计。现将共用 helper `_bootstrap_upper_tail_pvalue` 接入 iid 与 estimated-AR(1) 两条路径：有效拟合中 `chi2_boot >= chi2_observed` 的次数照常计，失败 refit 按最不利的上尾超越计，分母固定为 `planned+1`；若成功数低于冻结的 `90%`，panel 整体 abstain。形式为 `(1 + valid_exceedances + failed_refits)/(planned_draws+1)`，它对未知失败统计量给出上尾保守 completion；这不是额外 Monte Carlo 样本，也不把 failure 伪装成观测值。

图文流程：`每 panel 预定 B 次 inner refit -> 分成 valid / failed 并闭合 planned denominator -> 若 valid < 0.90B 则 ABSTAIN -> 否则把每个 failed refit 当作 exceedance -> 以 B+1 固定分母算保守上尾 p -> 外层校准仅汇总可报告 panel`。回归测试覆盖无失败时与标准 rank formula 一致、失败增加上尾 p、有效数不足弃权、分母不闭合时报错；`tests/test_optimize_d04_bootstrap_pvalue.py` **4 passed**，两个 runner `py_compile` 通过。历史 iid 与 AR(1) v1 报告的所有内层拟合都成功，故旧点值/拒绝率维持不变；历史 artifact hashes/source snapshots仍是其原始实现快照，不重写、不重复运行旧的 35 分钟 AR(1) 网格。此次只修复小样本失败分母策略，未解决 rho 估计误差、厚尾/多时间尺度依赖、finite-sample GOF 校准、模型选择或生产 E-reference；这些仍需独立审查和新 frozen held-out panel。保持两项 D06 对比、bounded block-risk estimand 与 `NOT_READY/UNCERTIFIED`，不读 sealed split、不启动 GPU/QUDA。

### 2026-11-28 Optimize no-launch source closure 自审

按不新增 D04 synthetic seed/rho/block-length 扫描的边界，只对 `draft_plan.py` 的 no-launch source map 做本地 import 闭包与摘要核对。21 个已声明源码/合同文件都存在，AST 遍历其本地 import 边后仍是 21 个，无未列出的本地 Python 依赖；`build_plan()` 生成的每个 source digest 与当下文件字节 `21/21` 匹配。该清单覆盖 D04 raw reader、D06 snapshot/Markov reducer、D11 metadata adapter、trainable-W 原型以及共享 evidence/source/NPZ helpers；它证明的是计划文件的源码身份，不是生产 evaluator 的运行依赖、输入来源或物理结果。

文字流程：`no-launch _source_paths -> 捕获 21 个源码/合同路径 -> AST 递归本地 import closure -> 与 source_hashes 逐字节核对 -> 无缺项 -> launch_allowed=false`。此次没有写入新结果目录、读 measurement/calibration/sealed bytes、重跑 synthetic panels 或启动 GPU/QUDA。当前物理认证门仍 `5/0/0/5`，数据输入 `0/0/0/0`，`NOT_READY/UNCERTIFIED`、`physical_score=null` 不变。下一软件工作应等待获准 development/calibration package 后，让 reviewed production evaluator 从同一 immutable snapshot 派生 D04/D11 summaries 和三臂 paired receipts；不得把 source closure 解释成 reference/effect/cost/coverage pass。

### 2026-11-27 D04 bootstrap 源码假设审计与校准精度门槛

本轮逐段检查了 [`d04_ar1_bootstrap_diagnostic.py`](d04_ar1_bootstrap_diagnostic.py) 的 `_sample_stationary_ar1`、`_estimate_common_rho` 与 outer/inner 调用链，以及 [`d04_parametric_bootstrap_diagnostic.py`](d04_parametric_bootstrap_diagnostic.py) 的拟合均值重建、raw covariance plug-in 和失败值计数；并与 [`d04_raw_reference.py`](d04_raw_reference.py) 的 covariance-of-mean、GLS objective、nominal degrees of freedom 对照。此项是实现作者对当前源码的自审，不是独立统计审查，也没有新读入数据或运行模拟。

**核实的实现性质与适用边界：**

- `_sample_stationary_ar1` 先取 `z_0 ~ N(0, Σ)`，随后按 `z_t = ρ z_(t-1) + sqrt(1-ρ²) ε_t`、`ε_t ~ N(0, Σ)` 递推，最后加上拟合均值。对 `|ρ|<1`、共同 scalar rho、Gaussian innovation 和 stationary marginal covariance `Σ`，该构造的有限段每个时点 marginal covariance 正确；它没有表示多个相关时间尺度、feature-specific rho、非平稳链、热化漂移或厚尾创新。
- `_estimate_common_rho` 把 27 个 feature 各自减 panel sample mean 后，汇总相邻乘积与平方和得到一个 pooled lag-1 OLS rho，并截断到 `[-0.95, 0.95]`。在“所有 feature 共用 rho”的 DGP 下，历史面板显示轻微向下偏差；对不同分量有不同相关时间、负/长记忆结构或短链热化效应时，它只是一个加权摘要，不能当作逐分量 Markov 参数。估计误差也没有在单个 outer panel 的 inner draw 中重新抽取；这是 plug-in bootstrap 的条件近似，需由独立 calibration 决定是否足够。
- 可在获准 calibration bytes 上与 pooled OLS 比较一个 stationary matrix-normal profile-likelihood 候选，而非把任何候选预先宣布为优胜者。令 `X∈R^(N×27)`、`R_ij(ρ)=ρ^|i-j|`、`X~MN(1 μᵀ,R(ρ),Σ)`；对每个 `ρ∈[-0.95,0.95]`，先算 `μ_hat(ρ)=(1ᵀR⁻¹1)⁻¹1ᵀR⁻¹X`、`S(ρ)=(X-1μ_hatᵀ)ᵀR⁻¹(X-1μ_hatᵀ)`，再最小化 profile deviance `27 log|R(ρ)| + N log|S(ρ)/N|`。这会同时 profile mean 与未知 feature covariance，避免把 arithmetic-centering OLS 当有限样本无偏；仍假定平稳共同-rho Gaussian、Σ full-rank，不能代表 feature-specific/多尺度/非高斯链。实现可用 Cholesky，或 AR(1) 精确三对角 precision 与解析 `log|R|`，并对 profile feature covariance 求 `logdet`；不得悄悄加 ridge。若 `S` 不可识别、profile 区间触碰 `±0.95` 或 ACF/innovation 检查不符 AR(1)，该 panel 应 abstain/转入 block-HAC lane，而非使用被截断的 rho 点值。比较图应并列 OLS 与 profile-likelihood rho、profile 曲线/区间、逐 feature lag-1 残差及覆盖率；是否采用由独立 held-out calibration 决定。
- `_regularize_sample_covariance` 使用 outer panel 的原始 configuration covariance（`ddof=1`）和极小正定 floor 生成 bootstrap configurations；每个内层样本再由 reader 重算 `cov(sample)/Ncfg`、25% diagonal shrinkage 和 `1e-10` relative eigenvalue floor。该顺序实现上前后一致，但依赖 raw sample covariance 与共同 AR(1) 模型能代表真实 marginal/long-run covariance；不能把 shrinkage covariance-of-mean 误当 long-run covariance。
- `_fitted_null_mean` 从主窗 `t=3..8` 的选中列及同一拟合协方差出发，在拟合 masses 固定时解 GLS amplitudes，再按 feature descriptors 重建 27 项 null mean；列标签顺序有显式一致性检查。`_fit_variant` 的三组 deterministic starts、质量坐标界、目标函数与 `dof = nobs - fitted_parameter_count` 在源码上闭合，当前主拟合 nominal dof 为 13。近边界、多态未识别、按结果选择 window/model，以及 covariance 估计与拟合残差的有限样本相关，仍会破坏把 `chi2.sf(..., 13)` 当校准 p-value 的条件。
- 当前共享 p-value helper 用 `1 + valid_exceedances + failed_refits` 除以 `planned_draws + 1`；`chi2_boot >= chi2_observed` 将 ties 放入上尾，失败 refit 作为最不利上尾 exceedance，低于 90% 成功拟合则 abstain。源码逻辑闭合了 planned denominator；历史 AR(1) v1 每个方法 `31,840/31,840` 内层拟合成功，所以后来的失败计数修复不改该报告点值，但旧 artifact 的 source snapshot 仍忠实记录当时版本。

**历史数据对“校准”的实际支持：**[`AR(1) v1 报告`](d04_ar1_bootstrap_cpu_20261127_v1/README.md)只有 `160` 个 outer panels、每方法 `B=199`。estimated-AR(1) bootstrap 在 1% 阈值拒绝 `2/160=1.25%`，Wilson 95% CI 为 `[0.34%, 4.44%]`；5% 拒绝 `10/160=6.25%`，区间 `[3.43%, 11.12%]`。同批 pooled rho 均值 `0.28623`，真值 `0.30`，偏差 `-0.01377`、标准差 `0.04079`。所以结果显示它比 iid plug-in bootstrap 在这个单一平稳 Gaussian toy 下少过拒绝，但样本量不足以判断 1%/5% calibration；不能把“区间包含标称值”写成已校准。`B=199` 的 rank p-value 最小步长为 `1/200=0.005`，对合同 `p>=0.01` 门槛过于粗。

**获准独立 calibration package 到位后的精度规则（规划值，不授权或触发本轮新合成扫描）：**

- 对单个 panel 的 bootstrap exceedance probability，在 `p≈0.01` 附近要求 pointwise 95% 正态近似 MC half-width 不大于 `0.002`（标称 1% 的 20%）；公式 `B ≈ z_(0.975)^2 p(1-p)/0.002^2` 给出约 `9,508` draws。最终区间改用 exact binomial interval；该数是内层模拟精度规划，不是物理数据量或通过证据。
- 外层校准若预注册 `K=6` 个 null strata、同时查看 `α={0.01,0.05}` 共 12 个 rejection rates，并要求 familywise 95% MC half-width ≤ `0.002`，Bonferroni 正态规划 `N ≈ z_(1-0.05/(2×12))^2 × 0.01×0.99 / 0.002^2` 得每 strata 约 `20,320` 个独立 panel。若实际 strata 数不同，必须按实际 `K` 重算；最终报告 Wilson/Clopper–Pearson 区间和完整计划分母。这个预算明显高于 v1，故应先评估实际 fit 成本，并使用预注册的 sequential Monte Carlo confidence sequence：每 panel 先做 199 draws，再按固定批量追加；只有 99% anytime-valid interval 已完全落在 `0.01` 与 `0.05` 同侧时才提前停，达到 9,999 draws 仍跨门槛则记为 MC-indeterminate/abstain。外层固定独立 panel ID；若要提前停止，另用对 12 个 family outcomes 作 alpha-spending 的 simultaneous confidence sequence。不可用重复查看普通 95% CI 的方式偷看停止。
- 这些数只为未来经授权、独立于方法开发集的 calibration 输入规划计算预算。当前冻结状态仍禁止重跑同一 synthetic generator 的 seed/rho/block 扫描；没有获准 bytes、来源 manifest 和 independent review 时，不创建 panel、不用合成数字补门槛，D04 继续 `NOT_READY/UNCERTIFIED`。

**图文交付和后续 agent handoff：**

```text
获准且与开发集独立的 calibration bytes + provenance/hash
  -> 固定 feature order、链 ID 与预注册 null strata
  -> outer fit (tmin=3, nstates=2) / rho_hat / raw covariance
  -> inner bootstrap: B=199 -> anytime-valid MC interval -> 批量扩到最多 9,999
       ├─ valid < 90% planned: ABSTAIN
       ├─ MC interval 跨 0.01 或 0.05: 追加或记 MC_INDETERMINATE
       └─ 不跨阈值: 保存 p-value、rho、协方差、所有失败与 denominator
  -> 独立 held-out strata 的 rejection rate + 同时 exact CI
  -> 仅当 reviewer 接受校准误差和拟合约定后，才考虑关闭 D04 GOF 子门
```

下一位执行 agent 必须先获得 package 的显式来源/使用授权，并由未参与该实现的人复核 pooled rho、profile-likelihood 推导、协方差估计与拟合窗/符号约定；再从同一 immutable snapshot 生成三张图：① `p`-value ECDF 对 Uniform(0,1)，展示同时置信带；② `α=1%/5%` 的拒绝率点与精确区间，标 nominal line、分母及失败/abstain；③ pooled-OLS/profile-likelihood rho 对已知/可审参考 rho 的恢复图，并展示 profile 曲线、逐 feature lag-1 residual 和 strata。图下同时给出文本 denominator 流程和 CPU/precision/cost。此自审只关闭“源码假设尚未逐项核对”的文档缺口，不关闭独立审阅、D04 reference、production convention 或统计校准门；不读 sealed data，不启动 GPU/QUDA，`physical_score=null` 不变。

### 2026-11-27 D04 stationary matrix-normal rho 候选的纯软件实现

为把上一节公式从纸面推进到可审查代码，在 [`d04_ar1_bootstrap_diagnostic.py`](d04_ar1_bootstrap_diagnostic.py) 新增 `_profile_matrix_normal_deviance` 与 `_profile_common_rho_candidate`。给定已有的 `N×d` feature panel 和候选 rho，前者应用 stationary AR(1) 精确三对角 precision、profile GLS feature mean，再以 residual scatter 的 `S/N` profile full feature covariance；row covariance 用解析 `log|R|=(N-1)log(1-rho²)`，feature covariance 用 `slogdet`，目标为 `d log|R| + N log|S/N|`（省略与 rho 无关常数）。后者在 `[-0.95,0.95]` 评估固定 81 点网格，精化每个网格已分辨的局部极小并比较两端点，返回 candidate、完整 profile grid/deviance、局部精化数和 `boundary_hit`。该过程降低单次 evaluation 的矩阵成本并减少单起点局部陷阱，但有限网格不构成任意窄极小的数学全局证明；profiled covariance 不正定、输入非有限或优化失败时 fail closed。

文字流程：`固定 N×d panel -> 精确 AR(1) 三对角 precision + row logdet -> GLS profile mean -> profile full feature covariance -> 81 点 profile 曲线 -> 精化已分辨局部极小 + endpoint check -> 保存 profile/boundary/failure`。新增 [`test_optimize_d04_ar1_profile_rho.py`](../../tests/test_optimize_d04_ar1_profile_rho.py) 用确定性代数 fixture 核对正/负 rho 的直接矩阵公式、feature 平移/缩放重参数下 rho 候选不变、奇异 profile covariance 被拒绝；与 p-value 分母回归合跑 `python -m pytest -q tests/test_optimize_d04_ar1_profile_rho.py tests/test_optimize_d04_bootstrap_pvalue.py` 为 **8 passed**，`git diff --check` 通过。

这只实现了一个待校准 estimator 候选：81 点 profile 加局部精化不能排除网格间任意窄的极小；未给 rho interval、未接入现有 AR(1) runner、未改变冻结 p-value 路径，也没有对报告 panel、随机 seed 或新 synthetic generator 运行它。最新代码上的 `python -m pytest -q tests/test_optimize*.py tests/test_wilson*.py` 为 **372 passed in 75.75 s**；这是 Optimize/Wilson 定向 CPU 回归，不是完整仓库测试或物理校准。下一步须在获准独立 calibration bytes 上检查整条 profile 曲线/多峰与边界、同 pooled OLS 对照 rho 恢复和 tail calibration，并画 rho/feature residual 诊断；独立统计审阅同意之前，现行估计器保持不变，D04 仍 `NOT_READY/UNCERTIFIED`、`physical_score=null`。

### 2026-11-28 Optimize/shared evidence 的超范围有限数 fail-closed 补齐

复核 D06 Markov envelope 路径发现，`evidence_adapter._finite` 与共享 `core.evidence._finite` 接受 `numbers.Real` 后直接做 `float(value)`；合法 Python `int` 但超出 double 范围时会抛 `OverflowError`，打断 adapter，而非返回 schema error/abstain。现让两个 validator 把转换越界、类型转换异常统一转成普通有限数校验失败。对已构造 envelope 再注入 `gamma_ps_lower=10**400` 的负控现在按既有 D06 失败路径 abstain，不提升阶段、不吞掉样本；Wilson q-reader 同轮的数值溢出负控见 [`tasks/wilson_loop/todo.md`](../wilson_loop/todo.md)。

文字流程：`D06 envelope gamma -> shared evidence finite-real validation -> task adapter finite-real validation -> gamma 不在 (0,1] -> statistical stage abstained`；超 double 范围整数不再越界抛错。定向测试 **43 passed**；随后当前 PR #6 worktree 的完整 `python -m pytest -q tests` **464 passed in 76.45 s**，`git diff --check` 通过。这里只关闭 metadata 数值异常逃逸；不改变 D04 reference、D06 estimand/Markov 假设、SLSQP/degree-4 对照或 D11 谱算符结论，也不产生 chain gap、TV、reference、solver-cost 或 SU(3) receipt。未访问 sealed bytes、未新增 synthetic panel/GPU 运行；三个 Optimize gate 仍 `NOT_READY/UNCERTIFIED`、`physical_score=null`。

### 2026-11-28 D04 独立合成依赖压力筛查（小样本，只用于方法分流）

新增 [`d04_heldout_dependence_diagnostic.py`](d04_heldout_dependence_diagnostic.py) 与 [筛查报告/图表](d04_dependence_screening_cpu_20261128_v1/README.md)，在预先冻结的五个合成 null strata 上配对比较现行 13-dof approximate chi-square、iid plug-in parametric bootstrap、pooled-OLS estimated-AR(1) bootstrap 和 `L=8` circular moving-block bootstrap；另记录 pooled OLS 与 matrix-normal profile rho 的恢复。strata 为 iid Gaussian、Gaussian AR(1) `rho=0.30/0.60`、`t(5)` innovation AR(1) `rho=0.30`，以及 `rho_fast=0.15/rho_slow=0.75`、fast weight 0.65 的两尺度 Gaussian mixture。流程：`冻结 DGP/fit/方法与 disjoint seed streams -> 每格生成 12 个 Ncfg=128 有序面板 -> 每面板作共同 PP-even/A4P-odd 两态 tmin=3 fit -> approximate / 三种 bootstrap -> failures 计入计划分母或触发 abstain -> 画 ECDF、1%/5% Wilson 区间、rho recovery -> synthetic screening only`。

筛查总计 60 个 outer panels；每种 bootstrap 计划 2,340 个 inner refits。各方法 inner fit 均为 `2340/2340/0/0`（planned/valid/failed/abstained）。当前 seed 下 5% 拒绝计数（每 stratum 12 个 outer panels）为：iid Gaussian 四法均 `0/12`；Gaussian AR(1) `rho=0.30` 的 approximate/iid/estimated-AR(1)/block 分别 `4/12, 6/12, 1/12, 1/12`；`rho=0.60` 分别 `10/12, 11/12, 1/12, 1/12`；`t(5)` AR(1) stress 中仅 iid bootstrap 为 `5/12`，其余为 `0/12`；两尺度 mixture 为 `6/12, 8/12, 4/12, 3/12`。这些计数的 Wilson 区间很宽，是描述性筛查，不能据此宣布任一方法通过校准。inner `B=39` 的最小可报告 rank p-value 为 `1/40=0.025`，因此 1% rejection 结构性为零，完全不能作为 1% size evidence。

报告保存 panel CSV、seed manifest、source snapshots、三张图与 hashes；18/18 artifacts、10/10 source snapshots 已核对。独立粗略计时为 20 次 nonlinear refit 用时 0.7295 s（约 0.0365 s/refit）。原拟议的 `200 outer × 199 inner × 5 strata × 3 bootstrap methods` 约需 597,000 次 inner nonlinear refit，粗略单线程估计约 6.1 小时，尚未计入 outer fit/profile 与绘图成本。因此先保留本小规模分流筛查，不把昂贵的 full grid 伪装成已完成；如将来决定运行确认设计，应先明确所需外层 MC 精度/多重性和算力预算，并在独立审阅后冻结分母。注意这张已冻结的筛查 ECDF 图使用逐曲线 pointwise 95% DKW band，并非跨 20 条曲线的同时带；当前 runner 已改成未来报告使用 Bonferroni simultaneous DKW band。拒绝率图显示逐格 Wilson 95% 区间，没有多重性校正。

此项仅检查已声明 Gaussian/AR(1)/厚尾/多尺度合成机制对四种校准候选的方向性影响；profile-rho 只作 nuisance recovery 诊断，bootstrap 仍使用冻结 pooled OLS 估计值。它不验证真实 Markov chain 的 stationarity、HMC dependence、production source/sink、E-chain provenance、D04 reference、coverage 或 candidate blindness。未读取 sealed bytes、未启动 GPU/QUDA，`physical_score=null`；D04/D06/D11 仍 `NOT_READY/UNCERTIFIED`。下一步先由独立统计审阅者复核 DGP、bootstrap fitted-mean/covariance plug-in、失败计数和图示，再决定是否值得按预注册 MC 目标扩大纯合成 panel；真实 D04 gate 仍需获准 candidate-blind E-chain package、来源/依赖 receipts、production convention 与独立物理 review。

### 2026-11-28 D04 calibration helpers 的作者自审（不是独立统计 review）

为在等待独立审阅与授权 package 时继续推进，逐式核对当前 `d04_ar1_bootstrap_diagnostic.py`、`d04_parametric_bootstrap_diagnostic.py`、`d04_heldout_dependence_diagnostic.py` 与其失败计数测试。**这是实现者自审，不满足此前要求的 independent reviewer。**未读取报告以外的 E-chain/solver/operator bytes，也没有再抽 synthetic panel 或改变冻结 DGP。

公式与软件路径复核如下：

1. `_sample_stationary_ar1` 先抽 `z_0~N(0,Σ)`，随后按 `z_t=ρ z_(t-1)+sqrt(1-ρ²) ε_t`、`ε_t~N(0,Σ)` 递推，故每步 stationary marginal covariance 都是 `Σ`，滞后协方差为 `ρ^k Σ`。`_finite_n_mean_variance_factor` 的 `1+2 Σ_(k=1)^(N-1)(1-k/N)ρ_k` 是有限长度样本均值方差对 iid `Σ/N` 的精确比值；多尺度生成器将独立 fast/slow AR 残差按平方根权重相加，故线性相关函数为权重平均。这里假设 stationarity 与生成器给定的相关核，不能证明实测链满足它。
2. `_estimate_common_rho` 对所有 feature 的去均值 lag-1 乘积与 lag-0 平方求一个 pooled 比值，再 clip 到 `[-0.95,0.95]`。它估计的是“各 feature 共享同一 scalar AR(1)”下的 nuisance 候选；有限样本偏差、特征异质 rho、慢快混合、厚尾与平衡性均未由公式消失。现有 bootstrap 将 `rho_hat` 和原面板 sample covariance 代入 stationary Gaussian generator，内层 refit 虽重估谱 fit，却不重估/传播 rho 与 generator covariance 的估计误差，因此是 plug-in 条件程序，不自动具有名义 size。
3. `_profile_matrix_normal_deviance` 实作 separable matrix-normal 工作模型 `Y~MN(1 μᵀ,R_ρ,Σ)`：用 stationary AR(1) 精确三对角 `R_ρ⁻¹` 求 GLS `μ_hat`，以 `S/N` profile feature covariance，再比较 `d log|R_ρ| + N log|S/N|`，其中 `log|R_ρ|=(N−1)log(1−ρ²)`。目标只省略与 rho 无关常数，不是已校准 likelihood-ratio statistic。profile helper 的 81 点网格 + 已分辨局部极小精化不是全局最优证明；`boundary_hit` 应使调用方 abstain。该 likelihood 仍只适用于共同 rho、stationary Gaussian、separable row/feature covariance 假设。
4. `_circular_moving_block_sample` 对每个完整 feature vector 去样本均值，按预定 `L=8` 循环抽连续 residual chunks，并加回 fitted null mean；它保留 chunk 内时间邻接和 feature vector 联合结构，但 wraparound 会把首尾接成邻居，且单一固定 L 不能覆盖未知链相关尺度。旧报告上的 L/B 敏感度并没有验证物理链 dependence。
5. `_bootstrap_upper_tail_pvalue` 将 tie (`chi2_boot >= chi2_obs`) 放入上尾，失败 refit 作为最不利 exceedance，用 `B+1` 分母；有效拟合不足 `ceil(0.90B)` 则 abstain。这对缺失统计量给出保守上尾计数，但不修正 plug-in generator 失配，也不等价于外层 type-I error 已校准。当前 `B=39` 筛查的最小非零 bootstrap rank p 为 `1/40=0.025`，1% tail 不可识别。

文字流程：`独立 outer panel -> fold/共同 tmin=3 两态 fit -> pooled OLS rho 与 profile rho 只作 nuisance recovery -> 用同一 fitted mean / sample covariance 生成 iid、plug-in AR(1) 或 L=8 MBB panel -> 每个 draw 重跑 fit -> tie 计上尾、失败计 conservative exceedance、有效率<90% 即 abstain -> 12 panel/stratum 的拒绝率只作分流`。检查结论与既有观察一致：estimated-AR(1) bootstrap 在某些 Gaussian AR(1) 格优于 iid/approximate，厚尾与多尺度格表现不同；该方向性结果不足以选定通用方法。全仓 suite **482 passed**（含本轮 Wilson 新增回归）；该 suite 证明实现和 fail-closed contract 行为，不证明这些渐近/plug-in假设或独立校准。

**尚缺项和交接：**由未参与该实现的统计 reviewer 复核 pooled rho finite-sample 行为、matrix-normal profile 适用性、同一数据估计 rho/covariance 后 plug-in bootstrap 的 null size、MBB wraparound/L=8 风险及 1%/5% MC 精度；审阅需附逐式图文与脚本/测试路径。若 reviewer 同意再决定是否需要新、独立于开发集的 null DGP/seed manifest 与确认样本量；不要继续沿用这五格 12-panel generator 调参。即使 synthetic calibration 通过，D04 reference 仍需显式获准、candidate-blind 的 E-chain immutable snapshot、真实 PP/A4P source/sink 与 t/window convention、依赖/来源收据和独立物理 review。D06 的 bounded block-risk estimand 与 SLSQP/degree-4 对照、D11 degree-4 算符合同亦尚不能用本审查关闭；三项保持 `NOT_READY/UNCERTIFIED`、`physical_score=null`、sealed inputs closed，不启动 GPU/QUDA。

### 2026-11-28 Optimize 采用共享 core 实数输入校验

此前 D06 iid 与 Markov 两个 risk decision 的复数 cast、bool/string coercion、超范围整数异常，现统一调用 [`core/numerics.py`](../../core/numerics.py) 的 `as_real_float_array`。共享 helper 还拒绝超出 float64 精确整数范围 `[-2^53,2^53]` 的整数，随后仍由 Optimize 原有 `[0,1]` score support、失败 mask 和风险 contrast 检查负责领域判定。Markov metadata scalar finite check 同样复用 `as_finite_real`，schema/evidence 的阶段语义未改。

文字流程：`shared exact-real conversion -> Optimize iid/Markov [0,1] validation -> failure=maximum loss -> separate contrast/support and gamma-TV checks -> decision or ABSTAIN`。核心数值 helper 与 Optimize D06 定向用例合计 **86 passed**；PR #6 全套 CPU tests **513 passed**。同一 core helper 也被 Pion、Wilson 调用，并已写入 [`doc/framework_extensibility.md`](../../doc/framework_extensibility.md) 作为可复用扩展点。此项只关闭数值类型旁路；不校准 bounded block-risk coverage、不认证 macroblock chain、D04 reference、paired production solver 或 D11 SU(3) operator。两个用户选定比较臂、`r_star=10/9` 与 D04 风险比 estimand 均保持不变；`NOT_READY/UNCERTIFIED`、`physical_score=null`、sealed inputs closed。

### 2026-11-28 no-launch source map 纳入共享数值依赖

共享 helper 被 D06 iid/Markov reader 导入后，Optimize 的 `draft_plan._source_paths()` 曾漏掉 `core/numerics.py`；因此 no-launch 报告虽含两个调用方源码，却不记录决定其转换边界的依赖。现把 helper 加入 `TaskSpec.dependencies`/`source_hashes` 共用清单和逐字节 retained snapshot，测试也要求 `snapshot_core_numerics.py` 与计划哈希一致。静态 AST 本地 import 闭包核对为 22 项声明（21 个 Python 文件加合同 JSON）、21 个 Python 依赖全部在清单内、无未列出的本地 Python import、缺失文件为 0；报告仍是 `DRAFT_NO_LAUNCH`、`data={}`、`launch_allowed=false`。

文字流程：`_source_paths -> SHA-256 TaskSpec dependencies -> retain same source bytes -> AST local-import closure -> compare every digest -> DRAFT_NO_LAUNCH`。`tests/test_optimize_draft_plan.py` 与 PR #6 当前全仓 CPU suite **515 passed**，`git diff --check` 通过。该检查证明 no-launch 草案所声明源码的身份与闭包，不证明 D04 数据有效、D06 覆盖率/链依赖、solver 成本公平、D11 生产 SU(3) 算符，亦不创建运行器。

**下一步交接：**任何真实 runner/evaluator 建立后，复用这张来源依赖图并从同一 immutable input snapshot 产生 D04/D06/D11 summary 和三臂 paired receipt；不可只 hash entrypoint 而漏掉本地 import。仍需独立获准 E-chain reference、dependence/calibration receipt、trainable-W/SLSQP/Chebyshev 配对 solver 结果与分阶段成本，以及 reviewed SU(3) link/operator receipt。缺任一项则不读 sealed IDs、不启动 GPU/QUDA，保持两个比较臂、bounded block-risk estimand、`NOT_READY/UNCERTIFIED` 与 `physical_score=null`。

### 2026-11-28 D04 profile-rho 多峰候选诊断记录

继续补强上述 stationary matrix-normal profile 候选的可审计性：`_profile_common_rho_candidate` 现在在返回值中记录固定 81 点网格的间距、两端点候选、每个被网格分辨的局部极小的括区/精化结果，以及最终候选来自端点还是局部精化。返回的 interpretation 明确说明：被选择的是“端点与网格分辨极小中的最低者”，有限网格无法排除网格间未采到的窄极小，也不构成全局最优证明。

新增 `tests/test_optimize_d04_ar1_profile_rho.py::test_profile_candidate_records_and_compares_all_grid_resolved_minima`，将 profile objective 替换为有两个分辨极小的确定性双井函数，核对两个极小均被精化并保留、最终选中较低者、网格间距正确，且返回文本带有未采样窄极小的限制说明；另以单调 objective 核对端点最优会标记为 endpoint 和 `boundary_hit=true`。文字诊断流程：`[-0.95,0.95] 固定网格 -> 检测网格局部极小 -> 在相邻网格括区内 bounded refine -> 与两个端点比较 -> 返回每个候选及来源标签 -> 明示有限网格局限/边界`。这是一项优化器/报告契约单元测试，不是新增合成 calibration panel，也没有重跑/调参任何历史 DGP、seed 或物理输入。定向测试 `python -m pytest -q tests/test_optimize_d04_ar1_profile_rho.py` 为 **6 passed**。

仍不把此候选接入冻结的 D04 bootstrap runner，也不把 `boundary_hit=false` 解释为模型适用或校准通过。后续如要进入数值比较，必须在批准且独立于方法开发的 calibration snapshot 上，与 pooled OLS 同 panel 比较 rho recovery、profile 多峰/边界及 tail calibration，并由独立统计 reviewer 审查 profile 假设和不确定度；真实 E-chain reference、source/sink convention 与依赖凭据仍未提供。D04 保持 `NOT_READY/UNCERTIFIED`、`physical_score=null`，sealed input 不读取，GPU/QUDA 不启动。

### 2026-11-28 D11 Chebyshev candidate 的数值入口与来源闭包

对已选 degree-4 Bernstein/Chebyshev CPU candidate 做入口审计后，发现 `lambda_fraction` 与 Gaussian `alpha` 等 scalar 经裸 `float`/`np.asarray(dtype=float)` 转换，会接受数字字符串/布尔值；同时 D11 报告虽保存 candidate/kernel 源码快照，却没有保存其新调用的共享 `core/numerics.py`。现将 control points、scaled spectral coordinate、Gaussian `alpha`、`ftol` 与 operator tolerances 接到共享 exact-real/finiteness validator；所有合法值域与 `b4=1, 0≤b0≤…≤b4` 合同不变，非法输入早于算子计算拒绝。

测试新增 control/coordinate/alpha/tolerance 的 bool、numeric-string、complex negative controls。D11 spectral-response report、framework candidate registration 与 source-math audit 现在都把 `core/numerics.py` hash 入 contract/source map，并分别保留 `snapshot_core_numerics.py`；旧报告不改写。文字流程：`controls/fraction/scalar tolerance -> exact-real conversion -> Bernstein/range/positive-tolerance checks -> degree-4 Chebyshev CPU matvec -> report manifest covers baseline + kernel + core numeric helper -> D11 physical gate still abstains`。这是接口一致性和来源闭包改进，不是 GPU 执行、SU(3) determinant/provenance 证明或 production objective 结果。

专项 `tests/test_optimize_chebyshev_baseline.py tests/test_optimize_chebyshev_baseline_report.py tests/test_optimize_candidate_contract.py tests/test_optimize_d11_operator_contract_audit.py` **36 passed**；当前 PR #6 全仓 CPU suite **525 passed in 83.43 s**，`git diff --check` 通过。独立 production-link/operator review、SU(3) receipt、D04 reference、paired D06 trainable-W/SLSQP/Chebyshev 风险与成本、GPU/QUDA authorization 仍缺；继续 `NOT_READY/UNCERTIFIED`、`physical_score=null`，sealed inputs closed。

### 2026-11-28 PR #6 跨任务 CPU 回归更新

Wilson C-lane compatibility adapter 与 shared C snapshot reader 的最新改动后，PR #6 当前整仓 CPU suite 为 **532 passed**。Optimize 的 D04/D06/D11 代码和研究合同未被本轮 Wilson adapter 修改；本次全套回归覆盖了它们与 shared core 的联合导入/测试状态，但仍只是软件回归。文字流程：

~~~text
D11 strict numeric/source closure + D04/D06 frozen CPU contracts
  -> shared repository suite including Wilson C reader
  -> 532 tests pass
  -> confirm software compatibility only; no E-reference, paired solver, SU(3) production receipt, or physical score
~~~

Optimize 继续保持 NOT_READY/UNCERTIFIED、physical_score=null；D04 E-reference/GOF independent review，D06 获准 paired solver 与 dependence receipts，D11 SU(3) source/operator review 和真实设备执行仍未闭合。

### 2026-11-28 D04 bootstrap resolution 的自批运行预算

我按用户“自行给出输入、自己批准”的要求，对现有 D04 synthetic screen 作出本地继续/停止决策，并冻结下一次新 calibration candidate 的 Monte Carlo 分辨率预算。当前屏查的 inner `B=39` 最小 rank p-value 是 `1/(B+1)=0.025`，因此不能检查 D04 合同中的 `p>=0.01` 边界，也不能产生有效的 1% size 结论；12 个 outer panels/stratum 的 1%/5% Wilson intervals 仅用于方向筛查。**不重跑同一 generator/seeds，也不把现有 60 panels 重新解释为确认性校准。**

若之后另有预先冻结的新 DGP/seed plan、拟合规则和候选方法，inner bootstrap 固定 `B=999`（rank resolution `0.001`），避免 p 值网格本身压在 1% 门槛以上。按二项比例正态近似，真实 rejection rate `p=0.01` 时要让 95% Monte Carlo 半宽不超过 `0.005`，outer 独立 panels 至少需 `ceil(1.96² p(1-p)/0.005²)=1522`；我取整到 **2000 panels/stratum**，使 1% 率的近似半宽约 `0.0044`。同样精度若还要求 5% 率半宽 `0.005`，约需 **7299** panels/stratum；因此该 2000-panel 方案只把 1% tail 设为主要 MC 精度目标，5% 结果仍附精确区间且不作为同精度声明。

按当前五个压力 strata、三种 bootstrap 方法计，新方案要约 `5×2000×3×999=29,970,000` 个内层拟合，未计 outer 拟合和失败；这是规划量而非已运行结果。现阶段**不批准**把该量直接投到当前 `B=39` 方法/模型上：先冻结并审计可进入下一轮的 null calibration protocol，再用新 seed 做小型无调参 pilot 估时；若仍无法解释 D04 reference-fit 的校准目标或资源，则停在 software/synthetic evidence，不开 GPU/QUDA、不读 sealed input。这里的本地自审和自批只决定试验预算与方法分流，不能替代独立统计 review、candidate-blind E-chain、PP/A4P source/sink 与依赖来源收据。

文字流程：`现有 B=39 screen -> 1% p-resolution 不足 -> 冻结 B=999 / 2000 outer per stratum 的新候选预算 -> 显示约 29.97M inner fits 的成本 -> 只有候选方法/DGP/seed 再冻结后才测时和运行 -> 物理 D04 仍等独立 E-chain`。该决定不改动 D04 reference contract、D06 `r_star=10/9` 与 trainable-W 对 same-budget SLSQP / degree-4 Chebyshev 两条比较，也不升级 Optimize 的 `NOT_READY/UNCERTIFIED` 或 `physical_score=null`。

### 2026-11-28 Optimize/Wilson 冻结报告哈希复核

为核实本地待审 PR #6 所依据的既有诊断没有在连续迭代中被静默改写，我按每个报告自己的清单重新读取 D04 dependence screen 与 Wilson F 五-strata screen。D04 的 10/10 captured source snapshot、18/18 report artifacts 均匹配；Wilson 的 root manifest 264/264、S1–S5 各 16/16 captured sources 与 49/49 分层 artifacts 全部匹配，合计 **617/617**。文字流程：`冻结报告 manifests -> 读取每条声明路径的当前 bytes -> SHA-256 重算 -> 与保存的 digest 逐项比较 -> 617/617 matched`。该复核只证明当前 checkout 内快照和已保存文件与其 manifest 一致；它不证明 snapshot 是独立/获准的生产源，不重跑 DGP，不验证 chain 或 SU(3) physics，也不增加 D04 reference/Optimize baseline 或 Wilson force receipts。

PR #6 当前 GitHub 状态仍为 OPEN，复核时 head `a06ecd91e3a83fc19b4ece6b2e66f82d6cde2f41`，没有 review 或 conversation comments。本轮在同一工作树执行 `PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider tests`，结果 **532 passed in 75.65 s**；这是当前源码的软件回归快照，不是 D04 校准、真实 E-reference、paired solver 或 Wilson force 的物理证据。不把该哈希检查或软件回归当作 gate closure。Optimize 仍保持两条用户要求的比较臂、bounded block-risk estimand、`NOT_READY/UNCERTIFIED`、`physical_score=null` 与 sealed data closed。

### 2026-11-28 PR #6 跨任务 CPU 回归：有限时间基态力反例后

在 Wilson F-lane 新增正谱有限时间非唯一性构造及 positivity-only 宽区间诊断后，重新运行当前 PR #6 全仓 CPU suite：**543 passed in 76.98 s**。这比 532 项快照多了 4 项 Wilson 定向测试；Optimize 的 solver / D04 / D06 / D11 实现与合同保持原样。新 Wilson 报告只证明正谱有限矩可有不同 ground force，并给出带 `a_t E_0>=0` 前提的宽 positivity interval；不产生任何 Optimize 的 reference、solver-risk、成本或 GPU 证据。

文字流程：`Optimize 双比较臂与 D04/D06/D11 合同 + Wilson finite-time positive-spectrum witness -> 全仓 CPU regression -> 543 passed -> 仍需独立 E-reference、获准 paired solver receipt、SU(3) 生产算符 review`。因此此项只更新软件兼容性快照；Optimize 三个 gate 继续 `NOT_READY/UNCERTIFIED`、`physical_score=null`、sealed inputs closed。下一步若没有开放的 Optimize development package，不重复 CPU toy labels 或旧 seeds，也不以这次跨任务回归宣称研究完成。

### 2026-11-28 PR #6 跨任务 CPU 回归：条件 Wilson Q/gap screen 后

Wilson 新增条件正谱 effective-energy 区间、四角 `Qmax/Delta_min` bias bound 与图文报告后，PR #6 全仓 CPU suite 为 **551 passed in 78.10 s**；相较上一节 543 项快照新增 8 个聚焦 Wilson 检查，其中含对“绝对力反例不改变同一 transfer matrix 下 stout-minus-identity 基态差”的直接回归。新结果仍只是 synthetic CPU evidence，不增加 Optimize 的 D04 reference、D06 学习型 smear/SLSQP/degree-4 Chebyshev 配对风险或 D11 SU(3) 算符审阅证据。文字流程与边界见[Wilson 条件力区间报告](../wilson_loop/f_lane_gap_overlap_force_screen_20261128_v1/README.md)。Optimize 继续保留两项比较和 bounded block-risk estimand，维持 `NOT_READY/UNCERTIFIED`、`physical_score=null`、sealed inputs closed；不要把跨任务测试通过计作 Optimize gate closure。


### 2026-11-28 跨任务回归：Wilson 绝对 stout 基态力合同修正后

PR #6 全仓 CPU suite **557 passed in 83.44 s**，包含 Wilson F-lane 将 primary 改为绝对 stout ground force、identity 改作 shared-spectrum consistency diagnostic、4 角主预算传播、schema-v3 summary reader 与 joint-fit output field 的回归。该变化没有修改 Optimize 的 D04 reference、D06 `r_star` 或学习型 smear / SLSQP / degree-4 Chebyshev 比较规则；该测试结果也不提供 Optimize 的 reference、solver-risk、预算差或 SU(3) 算符证据。

文字流程：`Optimize 合同保持冻结 + Wilson estimand/schema 修正 -> 全仓 CPU regression -> 557 passed -> Optimize gates 仍需各自的研究证据`。Optimize 仍保留用户要求的两条比较臂与 bounded block-risk estimand，维持 `NOT_READY/UNCERTIFIED`、`physical_score=null`、sealed inputs closed；不得把跨任务软件回归计作 Optimize gate closure。

### 2026-11-28 跨任务回归：Wilson 绝对力 delta interval screen 后

Wilson F-lane 新增绝对 ground-force 的解析 delta 梯度与未校准区间，并用 24 个独立 omitted-state synthetic panel 发现该近似仅 4/24 覆盖真值；图、输入/源码快照与 hashes 已保存在 [`f_lane_absolute_force_coverage_20261128_v1`](../wilson_loop/f_lane_absolute_force_coverage_20261128_v1/README.md)。本轮当前 PR #6 checkout 全仓 CPU suite **559 passed in 81.39 s**。文字流程：`新增 Wilson absolute-force diagnostic + 两项回归 -> 559 CPU tests passed -> Optimize contracts unchanged`。这只是跨任务软件兼容性和 Wilson synthetic 方法诊断，不产生 Optimize 的独立 E-reference、D06 配对 solver-risk/成本、SU(3) 算符审阅或生产授权证据。Optimize 仍保留 trainable-W 对同预算 SLSQP 和 degree-4 Chebyshev 两项比较、bounded block-risk estimand，状态 `NOT_READY/UNCERTIFIED`、`physical_score=null`、sealed inputs closed。

### 2026-11-28 PR #6 跨任务回归：Wilson 绝对力 full-refit MBB screen 后

Wilson F-lane 新增 6 个 AR(1) omitted-state panel 的 absolute-force full-refit MBB 屏查：234/234 bootstrap fits 有效，MBB interval 6/6 覆盖 toy truth，但 interval median width 是 delta 的约 5.2 倍且 B=39 只能给 sample range，不能称校准。PR #6 当前 checkout 全仓 CPU suite **563 passed in 81.74 s**。这仅验证软件回归和 Wilson 一套 synthetic DGP；不提供 Optimize 的 candidate-blind E-reference、D06 配对 solver-risk/成本、SU(3) 生产算符审阅或 GPU/数据授权证据。文字流程：`Wilson absolute-force MBB 234 full refits -> 6/6 覆盖但区间约宽 5.2x -> 563 CPU tests -> Optimize contracts/gates unchanged`。Optimize 仍保留 trainable-W 对同预算 SLSQP 与 degree-4 Chebyshev 两项比较、bounded block-risk estimand，状态 `NOT_READY/UNCERTIFIED`、`physical_score=null`、sealed inputs closed。

### 2026-11-28 PR #6 cross-task regression after Wilson state-count sensitivity screen

Wilson's 12-panel matched-`N_t=32` 2/3-static-state screen and four helper regressions were added while Optimize's contracts stayed frozen. The updated full PR #6 CPU suite passes **567 tests in 82.19 s**. This only checks software compatibility: it adds no D04 candidate-blind E-reference, D06 paired trainable-W/SLSQP/Chebyshev risk or cost evidence, D11 production SU(3) operator review, or authorization. Optimize remains `NOT_READY/UNCERTIFIED`, `physical_score=null`, sealed inputs closed.

### 2026-11-28 PR #6 cross-task regression after estimator-matched three-state pilot

Wilson's finite-`N_t` 2-vacuum × 3-static-state candidate, noiseless forward identity checks, one-panel synthetic fit report and four new focused tests leave Optimize contracts unchanged. The full PR #6 CPU suite now passes **571 tests in 84.00 s**. This is software compatibility only; it supplies no D04 E-reference, D06 paired solver risk/cost, D11 production operator review, or data/GPU authorization. Optimize remains `NOT_READY/UNCERTIFIED`, `physical_score=null`, and sealed inputs remain closed.

### 2026-11-28 PR #6 cross-task regression after three-state identifiability screen

Wilson added a five-stratum deterministic synthetic identifiability screen and three generator-contract tests. Full PR #6 CPU suite passes **574 tests in 78.07 s**. The screen found a full-rank weak-ground-overlap fit whose force point is displaced by about 5.1% of toy truth; this is a Wilson model-sensitivity counterexample, not Optimize evidence. Optimize's D04/D06/D11 contracts remain unchanged and `NOT_READY/UNCERTIFIED`; no E-reference, paired risk/cost, production operator review, sealed input or GPU authorization was added.

### 2026-11-28 PR #6 cross-task regression after conditional Wilson overlap-profile screen

Wilson tested a local Gauss-Newton force profile under three hypothetical ground-overlap floors. Only 2/18 local grid points passed the nonlinear-fidelity screen, none of the three floor grids closed an interval, and absence of an independent floor receipt correctly abstained. The updated full PR #6 CPU suite passes **576 tests in 78.85 s**. This is a Wilson synthetic-method result only; Optimize receives no D04 reference, D06 paired solver-risk/cost, D11 production operator evidence, or authorization. Its two comparisons and bounded block-risk estimand remain unchanged, with `NOT_READY/UNCERTIFIED`, `physical_score=null`, and sealed data closed.

### 2026-11-28 PR #6 cross-task regression after Wilson analytic-Jacobian profile pilot

Wilson added an analytic Jacobian for the finite-`N_t` 2-vacuum × 3-static model and an exact-force nonlinear conditional profile. Its Jacobian matches central differences at `8.82e-12`; the one-panel nominal profile still excludes toy truth for floors 0 and 0.045, and the 0.075 case is grid-clipped. The full PR #6 CPU suite passes **578 tests in 76.96 s**. This adds no Optimize D04 reference, D06 paired risk/cost, D11 production operator evidence, or authorization; Optimize's comparisons, bounded block-risk estimand and `NOT_READY/UNCERTIFIED` state are unchanged.

### 2026-11-28 PR #6 cross-task regression after six-panel Wilson profile screen

Wilson's exact nonlinear force profile was screened over six fresh weak-overlap synthetic panels: 72/72 cells and 144/144 starts were valid, but empirical truth containment was 2/5 for no floor and 3/6 for an oracle toy floor of 0.045; Wilson 95% binomial intervals are broad, so the result is not calibrated. The updated full PR #6 CPU suite passes **582 tests in 76.52 s**. Optimize's D04/D06/D11 contracts and comparisons are unchanged; it gains no reference, paired-risk/cost, production operator or authorization evidence.

### 2026-11-28 PR #6 cross-task regression after Wilson multi-stratum profile abstention map

Wilson added 12 new profile panels across baseline, near-degenerate, omitted-vacuum and dependent/unequal-variance toy strata, reusing the six weak-overlap panels. It found 195/216 valid profile cells and 380/432 valid starts; the near-degenerate and omitted-vacuum strata had **0/3 complete profiles** under the generator-truth oracle floor, so this is an abstention/numerical-stability result rather than a coverage estimate. The full PR #6 CPU suite passes **584 tests in 78.57 s**; the focused Wilson profile/Jacobian tests pass **8 tests in 1.76 s**. This is synthetic Wilson evidence and does not change Optimize's D04 reference, trainable-W vs same-budget SLSQP and degree-4 Chebyshev contrasts, bounded block-risk estimand, or any D11 production-operator evidence. Optimize remains `NOT_READY/UNCERTIFIED`, `physical_score=null`, and sealed inputs closed.

### 2026-11-28 PR #6 cross-task regression after Wilson continuation retry

Wilson retried 13 failed near-degenerate/omitted-vacuum oracle-floor cells using the nearest valid force neighbor as a second start. Three cells were recovered, all in the near-degenerate stratum; only 3/26 continuation starts were valid. The near-degenerate complete-profile count rose from 0/3 to 1/3, but its one eligible interval still missed toy truth; omitted-vacuum remained 0/3 complete. A read-only receipt audit found all 26 cold-invalid and all 23 failed continuation starts hit the `nfev` cap while retaining local rank 66/66; worst Jacobian condition number was `1.10e10`. Full PR #6 CPU suite: **588 passed in 78.37 s**; focused Wilson profile/continuation checks: **12 passed in 1.14 s**. This is a solver-abstention diagnostic on reused Wilson toys; Optimize's D04/D06/D11 contracts, both requested comparison arms and bounded block-risk estimand remain unchanged. No Optimize reference, paired risk/cost, operator review or authorization was added; status remains `NOT_READY/UNCERTIFIED`, `physical_score=null`, sealed inputs closed.

### 2026-11-28 PR #6 cross-task regression after Wilson stationarity and dogbox comparison

Wilson recomputed scaled projected gradients from 49 saved failed profile endpoints, then compared TRF and dogbox on the same six synthetic panels. Dogbox raised valid starts from 0/26 to 13/26 for cold starts and from 3/26 to 13/26 for neighbor starts, but complete profiles stayed 1/3 in near-degenerate and 0/3 in omitted-vacuum; the only interval still missed toy truth. The updated full PR #6 CPU suite passes **588 tests in 80.02 s**; both new diagnostic scripts compile and report artifact hashes verify. This is synthetic Wilson solver evidence only. It adds no Optimize D04 reference, D06 paired trainable-W/SLSQP/Chebyshev risk or cost evidence, D11 production operator review, or authorization. Optimize keeps both requested comparison arms and bounded block-risk estimand; status remains `NOT_READY/UNCERTIFIED`, `physical_score=null`, and sealed inputs closed.

### 2026-11-28 paired D06 cost-ledger reducer

为填补“未来三臂结果虽有 score reader、但 setup/training 与每块求解成本还没有统一账本”的软件缺口，新增 [`paired_cost_accounting.py`](paired_cost_accounting.py)，并把它纳入 no-launch RunSpec draft 的 SHA-256 dependency/snapshot map。账本要求 trainable-W、同预算 SLSQP、degree-4 Chebyshev 三臂共享一个 `input_manifest_sha256` 与 `seed_plan_sha256`；逐臂记录协议 hash、device、precision、thread count、software versions 与 clock policy；每臂都必须逐个覆盖冻结 `planned_block_ids`，状态保留 `valid/failed/abstained`，每个阶段记录 objective/gradient calls 与秒数。`setup/training` 等 shared costs 单独列出并完整计入单个 task-batch total；每块 `solve/inference/I/O/failure_recovery` 成本单独聚合，失败恢复仍计时，不把训练成本摊到未来未批准的多个任务。只有全部 planned block rows 存在且三臂环境字段相同，才给 full-batch walltime ratio；否则返回明确的 incomplete/environment/invalid abstention，ratio 置空。hash 字段只作格式校验，不认证签发身份、来源或物理有效性。

文字流程：

~~~text
shared input-manifest + seed-plan hashes
  -> trainable-W / same-budget SLSQP / degree-4 Chebyshev arm receipts
  -> exact planned-block IDs + per-block valid/failed/abstained + calls/timing
  -> matched device / precision / threads / software / clock policy
  -> full setup + training + solve + inference + I/O + failure-recovery batch costs
  -> complete & comparable: expose paired cost ratios; otherwise abstain with no ratio
~~~

四项新增 cost-ledger 测试覆盖完整批次（含失败臂成本和 shared training charge）、漏块弃权、运行环境不匹配、负时间/布尔调用数/重复 block ID 负控；no-launch draft 测试也确认新 reducer 被纳入源码快照。PR #6 当前 Optimize 专项 `python -m pytest -q tests/test_optimize*.py` 为 **203 passed in 53.46 s**；`py_compile` 与 `git diff --check` 通过。该项只提供严格汇总接口和 synthetic schema fixtures；没有运行真实 solver、读取 measurement/sealed 数据或测硬件成本。后续必须由实际 paired-results producer 从同一授权 snapshot 填入完整三臂账本，并同时交付 D04 E-reference、D06 risk/依赖 receipts、D11 SU(3) operator review 与必要授权。Optimize 仍 `NOT_READY/UNCERTIFIED`、`physical_score=null`，sealed inputs closed。

### 2026-11-28 paired cost ledger 的 aggregate overflow fail-closed 修正

对已加入的 [`paired_cost_accounting.py`](paired_cost_accounting.py) 做极端合法输入审计：每行 seconds 即使各自有限，跨 block 的 phase sum、全任务六阶段总和仍可能溢出 float64；一个极小正 trainable-W denominator 也可能让有限 numerator 的 cost ratio 变成 `inf`。现以 `math.fsum` 检查每次累加及最终 batch sum，任何 aggregate overflow/non-finite 均返回 `ABSTAIN_COST_LEDGER_INVALID`；比例若非有限则置空并返回 `ABSTAIN_COST_RATIO_UNDEFINED`，不把 `inf` 当作巨大但有效的比较值。

文字流程：`finite per-row phase seconds -> guarded aggregate by phase -> guarded all-phase batch total -> complete comparable arms -> finite full-batch ratio; any overflow/non-finite -> abstain with null ratio`。新增两项负控分别让两个 `1e308` block costs 在 aggregate sum 溢出、以及 `1e308/1e-308` cost ratio 溢出；连同之前四项账本测试共 **6 passed**。本轮与 Wilson pivot-chart 定向测试合计 **11 passed**；`py_compile` 与 `git diff --check` 通过。此修正仅加固账本归约，不含真实 paired solver receipt、设备时间或 Optimize 物理结果；D04 E-chain reference、D06 三臂配对风险/依赖收据、D11 SU(3) production review 与必要授权仍未提交，Optimize 继续 `NOT_READY/UNCERTIFIED`、`physical_score=null`、sealed inputs closed。

### 2026-11-28 PR #6 全仓 CPU 回归：cost-ledger overflow 与 Wilson pivot chart 后

本轮新增 paired cost ledger aggregate-overflow/ratio-overflow 两个负控，并将 5 项 Wilson largest-component phase-chart 方法检查纳入当前仓库；PR #6 全仓 `python -m pytest -q tests` 为 **599 passed in 83.58 s**。这是 CPU 软件兼容性回归，不是 D04 reference-fit calibration、D06 trainable-W/SLSQP/Chebyshev 风险或真实成本证据，也不提供 D11 SU(3) production review。`physical_score=null`，sealed inputs closed。Wilson 相位图详细结果见 [`../wilson_loop/f_lane_three_static_pivot_chart_20261128_v1/README.md`](../wilson_loop/f_lane_three_static_pivot_chart_20261128_v1/README.md)。

### 2026-11-28 PR #6 cross-task regression after Wilson paired phase-chart fit screen

Wilson added an analytic-Jacobian paired point-fit screen on 12 reused synthetic panels; the profile-advance gate did not pass because the largest-pivot chart lost valid starts in two strata. One regression also checks exact fixed-operator-0 / zero-pivot equivalence and its analytic Jacobian. The current full PR #6 CPU suite passes **600 tests in 89.08 s**; the focused Wilson chart file passes **6 tests in 1.08 s**. This adds no Optimize D04 reference fit, D06 paired trainable-W/SLSQP/Chebyshev risk or hardware cost, D11 production SU(3) review, or authorization. Optimize's two requested comparisons and bounded block-risk estimand are unchanged; status stays `NOT_READY/UNCERTIFIED`, `physical_score=null`, and sealed inputs closed.
