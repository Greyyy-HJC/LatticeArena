# RSI Problem: 格点 QCD 中强子插值算符的优化构造

## 1. Scientific Motivation

在格点量子色动力学（Lattice QCD）中，强子插值算符（hadron interpolating operator）的构造直接影响关联函数的统计精度及激发态污染。

传统方法通常采用 Gaussian smearing、momentum smearing 等具有预设空间结构的算符。然而，这些算符并不一定能够在不同强子及不同动量条件下实现最优的基态投影。

本项目旨在利用 AI agent 自主探索和构造更优的强子插值算符，使其在保持正确物理量子数的前提下，同时实现：

- 更大的目标强子基态相对重叠；
- 更低的激发态污染（excited-state contamination）；
- 更高的信噪比（signal-to-noise ratio, SNR）；
- 更早且更可靠的基态主导区域。

本项目仅使用强子两点关联函数（hadron two-point correlation functions）作为评价依据，不要求进行三点函数或强子矩阵元的计算。

## 2. Physical Problems

考虑 pion 和 proton 两种代表性强子，分别研究三种动量条件：

| Hadron | 零动量 | 小动量 | 大动量 |
|---|---|---|---|
| Pion | $P_z=0$ | $P_z=P_{\rm low}$ | $P_z=P_{\rm high}$ |
| Proton | $P_z=0$ | $P_z=P_{\rm low}$ | $P_z=P_{\rm high}$ |

具体动量由所提供格点组态的空间尺寸和格距确定：

$$
P_z=\frac{2\pi n_z}{aL_s}.
$$

例如，可选取 $n_z=0,1,4$ 作为初始测试条件。

Agent 可以针对不同强子和动量分别构造最优算符，也可以提出适用于多个物理条件的统一算符构造方法。

## 3. Operator Construction and Physical Constraints

允许 agent 在一般化的算符空间内进行探索：

$$
\mathcal O_h(\mathbf P,t)
=
\sum_{\mathbf x}e^{-i\mathbf P\cdot\mathbf x}
\sum_{\alpha,\{\mathbf r_i\}}
W_{h,\alpha}(\{\mathbf r_i\},\mathbf P;U_t)
\mathcal B_{h,\alpha}
(\mathbf x,\{\mathbf r_i\},t).
$$

其中，$\mathcal B_{h,\alpha}$ 表示具有相应夸克场、颜色收缩、Dirac 结构及空间位移的算符基元，$W_{h,\alpha}$ 表示相应权重或由同一时间切片上的规范场决定的权重函数。

算符构造不限定于 Gaussian smearing，也不要求具有球对称性、可分离性或固定的解析形式。允许：

- 任意空间形状及复数权重；
- 不同夸克之间具有相关性的空间分布；
- 各向异性的 smearing kernel；
- 动量依赖的算符结构；
- 多种允许的 Dirac 结构的线性组合；
- Gauge-equivariant neural networks 或其他机器学习方法。

所有算符必须满足以下物理约束：

1. 保持正确的 flavor、baryon number、颜色单态及相关格点对称性。
2. 确保目标基态为所要求的 pion 或 proton 态。
3. 在非规范固定组态上，空间分离的夸克场必须通过适当的 gauge transport 或 gauge-covariant construction 实现规范不变性。
4. 在非零动量下，必须考虑剩余格点对称性（little group）以及不同 Dirac 结构之间允许的混合。对于 proton，还需要处理可能的 opposite-parity contamination。
5. 算符应定义在单一欧氏时间切片上，不能通过显式依赖 source-sink separation 的权重人为改变关联函数的时间行为。

优化过程中不允许修改 QCD action、规范场组态或夸克传播子本身的物理定义。

## 4. Data and Computational Resources

项目方应提供以下两类数据之一。

### Option A: Dynamical-fermion gauge ensembles

提供一套固定的包含 dynamical fermions 的格点 QCD gauge ensemble，以及：

- 格点尺寸、格距、夸克质量和费米子 action；
- 边界条件及组态编号；
- 计算夸克传播子所需的软件和参数；
- 固定的计算资源及 Dirac inversion budget。

Agent 可以在规定的计算预算内自主构造 source 和 sink 算符并计算传播子及两点函数。

### Option B: Precomputed point-to-all propagators

提供在一套固定 dynamical-fermion gauge ensemble 上预先计算的 point-to-all propagators，包括必要的 source positions、spin/color components、quark flavors 和 gauge links。

Agent 使用已有传播子构造并优化强子算符。

若传播子集合不能支持任意 source kernel，则任务限制为固定 source 下的 sink operator 优化，或已有 source basis 的线性组合，并使用对应的 baseline。

正式评分需要多个 gauge configurations，以可靠估计 ensemble-level statistical uncertainties 和 SNR。单个 gauge configuration 仅可用于算符构造及代码测试。

## 5. Evaluation and Quantitative Score

定义动量投影后的两点关联函数：

$$
C_{\mathcal O}(t,\mathbf P)
=
\left\langle
\mathcal O_h(\mathbf P,t)
\mathcal O_h^\dagger(\mathbf P,0)
\right\rangle.
$$

以下时间和能量均使用 lattice units。评价主要考虑：

$$
0<t=t_{\rm sep}<\frac{L_t}{2}.
$$

考虑到统计噪声、有限时间边界效应等因素，实际有效时间窗口可能更小：

$$
\mathcal T=
\{t_{\min},\ldots,t_{\max}\}.
$$

每个物理条件下的评价窗口及权重由项目方根据开发数据预先确定。所有候选算符和 baseline 必须在相同窗口内进行比较。

### 5.1 Excited-State Contamination

参考 [arXiv:2602.02436](https://arxiv.org/abs/2602.02436) Section III.C 中基于归一化关联函数的优化方法。

对于具有正定前向谱权重的相同 source-sink 算符，在处理 backward-propagating contributions 后：

$$
C_{\mathcal O}^{+}(t,\mathbf P)
=
\sum_{n=0}^{\infty}
|Z_n|^2 e^{-E_n t}.
$$

理想的纯基态关联函数满足：

$$
\frac{C_{\mathcal O}^{+}(t)}
{C_{\mathcal O}^{+}(t_0)}
=
e^{-E_0(t-t_0)}.
$$

定义无量纲的激发态污染指标：

$$
D_{\mathcal O}(t)
=
1-
e^{E_0^{\rm ref}(t-t_0)}
\frac{C_{\mathcal O}^{+}(t)}
{C_{\mathcal O}^{+}(t_0)},
$$

其中 $E_0^{\rm ref}$ 为独立确定的目标强子基态能量，$t_0$ 为预先固定的参考时间，且 $0<t_0<t_{\min}$。

综合评价指标为：

$$
D_{\rm ES}(\mathcal O)
=
\sum_{t\in\mathcal T}
w_{\rm ES}(t)D_{\mathcal O}(t),
$$

其中：

$$
\sum_{t\in\mathcal T}w_{\rm ES}(t)=1.
$$

默认采用均匀权重。

在正定谱分解及正确基态能量的条件下，$D_{\rm ES}$ 越小，表明算符具有越好的基态投影能力。

该指标应通过统一的谱分析方法估计，并考虑统计误差、有限时间效应以及基态能量的不确定性。

对于非对称 source-sink 算符，应采用独立评分分支，通过多态拟合中激发态贡献绝对值的占比评价污染，以避免不同激发态贡献相互抵消造成的虚假改善。

### 5.2 Signal-to-Noise Ratio

定义每个时间点的信噪比：

$$
\mathrm{SNR}_{\mathcal O}(t)
=
\frac{|\langle C_{\mathcal O}(t)\rangle|}
{\sigma_{\langle C_{\mathcal O}(t)\rangle}},
$$

其中 $\sigma_{\langle C_{\mathcal O}(t)\rangle}$ 表示 ensemble mean 的统计误差。

在相同统计样本和计算预算下，定义相对于 baseline 算符 $\mathcal O_B$ 的时间加权信噪比改善指标：

$$
R_{\rm SNR}(\mathcal O)
=
\sum_{t\in\mathcal T}
w_{\rm SNR}(t)
\frac{1+\mathrm{SNR}_{\mathcal O}(t)}
{1+\mathrm{SNR}_{\mathcal O_B}(t)}.
$$

默认采用：

$$
w_{\rm SNR}(t)
=
\frac{t}{\sum_{t'\in\mathcal T}t'}.
$$

该权重使中、大时间间隔的信号质量获得更高的评价权重，并避免短时间处的大信噪比主导整个评分。

### 5.3 Final RSI Score

定义单个物理条件下的综合评分：

$$
\boxed{
S(\mathcal O)
=
100
\left[R_{\rm SNR}(\mathcal O)\right]^\alpha
\left[
\frac{\epsilon+D_{\rm ES}(\mathcal O_B)}
{\epsilon+D_{\rm ES}(\mathcal O)}
\right]^{1-\alpha}
}
$$

建议初始设置：

$$
\alpha=0.5,\qquad \epsilon=0.01.
$$

其中 $\epsilon$ 用于避免激发态污染接近零时评分发散。

- Baseline score = 100。
- Score > 100 表示综合性能优于 baseline。
- Score 越高，代表更优的综合性能。
- 未通过物理量子数、基态能量或统计可靠性检查的算符，该项得分为零。

最终综合分数为六种物理条件下得分的等权平均。同时，应分别报告每种条件下的 SNR 和 excited-state contamination 指标。

评分参数、谱分析方法和有效时间窗口应在正式测试前冻结。

## 6. Baseline Reference

采用以下传统算符作为 baseline：

| Momentum regime | Baseline operator |
|---|---|
| Zero momentum | Gauge-covariant Gaussian smearing |
| Low momentum | Gaussian or momentum smearing |
| High momentum | Momentum smearing |
| Coulomb gauge | Coulomb-gauge Gaussian or momentum smearing |

Baseline 的参数应在开发数据上进行合理优化，并与候选算符使用相同的数据、统计样本和计算预算。

参考文献：

- G. S. Bali et al., *Novel quark smearing for hadrons with high momenta in lattice QCD*, [arXiv:1602.05525](https://arxiv.org/abs/1602.05525).
- V. Bellscheidt et al., *Wilson loops with neural networks*, [arXiv:2602.02436](https://arxiv.org/abs/2602.02436).

## 7. Optional Extension: Coulomb-Gauge Operator Optimization

在固定到 Coulomb gauge 的 gauge ensemble 上重复上述问题。

在该变体中，agent 可以直接探索空间分离的夸克场组合，无须显式引入空间 Wilson lines，但仍需保持残余规范对称性、颜色单态及正确的强子量子数。

需要研究：

- Coulomb gauge fixing 是否能够简化最优算符的空间结构；
- 大动量下是否存在更优的非 Gaussian 或各向异性算符；
- Coulomb-gauge operators 与 gauge-invariant operators 之间的性能差异。

该变体使用一致的 gauge-fixing prescription，并采用独立的 baseline 和评分。

## 8. Pilot Deliverables

Agent 应提交：

1. 每种强子及动量条件下优化后的算符构造，包括空间权重、Dirac 结构和实现方式。
2. 优化算符及 baseline 的两点关联函数。
3. Effective energy、SNR 和 excited-state contamination 随时间变化的结果。
4. 六种物理条件下的独立评分和综合 RSI score。
5. 对所得算符空间结构、动量依赖性及物理性质的分析。
6. 完整可运行的源代码、模型参数及计算资源使用记录。

## 9. Reproducibility and Contributor Credits

- Gauge configurations 按组态编号划分为 training、validation 和 held-out test sets。
- 算符优化、超参数选择及 baseline tuning 不得使用测试集。
- 最终评分应基于冻结后的算符在独立测试组态上的表现。
- 使用统一的 bootstrap 或 jackknife 方法估计统计误差，并考虑组态间的自相关。
- 公开或向项目方提供完整的代码、运行参数、随机种子和数据说明。
- 明确记录 ensemble providers、propagator providers、task designers、baseline developers 和其他 contributors 的贡献及署名信息。