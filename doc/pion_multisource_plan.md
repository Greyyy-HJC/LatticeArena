# pion_projection：多 source GEVP 的预注册方案（未执行）

C2 的 2×2 和 3×3 Hankel GEVP 都失败了（`doc/pion_gevp_design.md` §2.5–2.6）：powers-of-$H$
家族彼此高度相关，补一个 $H^{24}$ 也换不来独立的算符方向。本文把“多 source 测量”写成
可执行的预注册方案，**尚未提交任何 GPU 任务**。

## 1. 为什么需要多 source

Hankel 构造依赖

$$
C_{ij}(t)=\langle K_iS,K_jS\rangle=\langle S,H^{n_i+n_j}S\rangle,
$$

所有通道都来自**同一个点源**，只是 sink 不同；矩阵的秩由 $S$ 的谱决定，而不是由独立的算符
方向决定。要得到真正的 GEVP 矩阵，source 侧也必须是不同的算符：

$$
C_{ij}(t)=\langle O_i(t)\,O_j^\dagger(0)\rangle,
\qquad O_i=\bar q\,\Gamma K_i q .
$$

## 2. 测量方案

### 2.1 source 集合

| 编号 | source | PyQUDA 入口（待核对签名） |
| --- | --- | --- |
| 0 | point | `source.source(latt, 'point', ...)`（现有） |
| 1 | Gaussian, $H^4$ 等效 | `source.gaussian12` / `source.gaussianSmear` |
| 2 | Gaussian, $H^{16}$ 等效 | 同上，steps 更大 |
| 3（可选） | wall | `source.wall` |

建议先做 $N_s=2$（point + Gaussian）验证可行性，再扩到 $N_s=3$。

### 2.2 每个配置的流程

1. 对每个 source $j$：构造源场 $\rho_j$（保存其空间剖面），做一次反演得到 $S_j$；
2. 对每个 sink $i$：把 $K_i$ 作用到 $S_j$ 上，与 $\rho_j$ 收缩，得到 $C_{ij}(t)$；
3. 存成新的数组 `correlators_ij.npz`，shape `(nsrc, nsink, T)`，**不改动现有 NPZ**；
4. 记录每个 source 的 hash、反演残差、耗时。

### 2.3 成本

- 反演：$N_s$ 次/配置（现在是 1 次）→ $N_s=2$ 约 2×，$N_s=3$ 约 3×；
- 收缩：$N_s\times N_k$ 次 kernel 应用（现在 $N_k$ 次）；
- 60 开发 + 30 留出、$N_k=7$、$N_s=3$：约 **50–70 分钟** GPU（现有单 source 约 20 分钟）。

## 3. 预注册项

- source 集合与 $N_s$、sink 集合与 $N_k$；
- $t_0$（建议 2 或 3）与 $E_0$ 平台窗口（建议 $t=4\dots8$，取最后三点）；
- bootstrap：按 configuration block（长度 5）重采样，GEVP 在每个 draw 内重解；
- 失败判据：
  - $C(t_0)$ 非正定或条件数 > $10^{10}$；
  - 本征值在 draw 间排序交换超过 5%；
  - $t_0$ 从 2 改到 3 使 $E_0$ 移动 > 2σ；
  - $E_0(t)$ 在平台窗口内漂移 > 2σ。
- 若失败：输出“GEVP 不可识别”，不发布能量。

## 4. 与现有评分的关系

- 这个测量只增加通道，不改变 $\kappa$、几何、splits 或统计；
- 新的 GEVP 参考若通过预注册的门，可以替换 `reference.py` 里的单 cosh/两态，成为
  `references` 集合的第三个模型；
- 在它通过之前，pion 的正式 score 保持 `null`。

## 5. 执行结果（2026-11-18，已跑）

`measure_multisource.py` 实现了 3 个 source（point、`gaussian12` 的 gauss4/gauss16）的测量，
在 60 开发 + 30 留出组态上跑完（每配置约 2.5 秒，远低于原估的 50–70 分钟）：

- 产物：`results/pion_projection/s16t16_ms/`（`correlators` shape `(ncfg, 3, 8, 16)`）；
- 矩阵 $M_{ab}(t)=C[\text{source }a,\text{sink }H^{n_b}](t)$，$a,b\in\{$point, $H^4$, $H^{16}\}$；
- $C(t_0=2)$ 有 1.6% 的负本征值（噪声），用相对 $10^{-6}$ 的本征值 floor 正则化后求解。

结果**仍然不稳定**：

| 数据 | $E_0$ | 每时间点 $E_0(t=3,4,5,6)$ |
| --- | ---: | --- |
| development | $0.832\pm0.169$ | 0.676 / 0.780 / 0.947 / 0.924 |
| validation | $0.818\pm0.123$ | 0.698 / 0.817 / 0.939 / 0.820 |

### 诊断：source 与 sink 的基不一致

`gaussian12` 用的是 Wuppertal smearing（$\alpha=1/(4n/\rho^2-6)$），不是我们的
$H=(I+0.2\,\mathrm{hop})/2.2$。source 剖面和 sink 核不匹配，$M_{ab}$ 不是同一组算符的
关联矩阵，GEVP 因此没有物理意义。

**修复方向**：用与 sink 完全相同的 $H^n$ 作用到 source 场（`kernels.apply_kernel` 可以直接
作用在 source12 的每一列上），再做反演。这需要把 source12 的列转成时间片场格式、应用核、
写回——代码量不大，但要先在小样本上验证 $C_{ab}$ 的对称性。

## 6. 结果与诊断（2026-11-18）

把 source smearing 调成与 sink 相同的 $H^n$（$\rho=0.7146\sqrt{n}$，使
$\alpha=0.2\cdot6/2.2=0.5454$，每一步正好是 $H$）后重测，3×3 GEVP 的结论如下。

### 6.1 构造是对的

测量给的是

$$
C[\text{source }a,\text{sink }b]=\langle S K_a,\,K_b\,S K_a\rangle ,
$$

即 source 为 $K_a$、sink 为 $K_b$ 的关联函数，也就是 $M_{ba}$。GEVP 矩阵是它的转置，
而 $\det(C(t)^T-\lambda C(t_0)^T)=\det(C(t)-\lambda C(t_0))$，**广义本征值不变**。
所以这个矩阵是合法的 GEVP 矩阵（上一版“构造错误”的说法不成立，已撤回）。

### 6.2 真正的困难：条件数与正则化依赖

$C(t_0=2)$ 有约 1.6% 的负本征值，必须做本征值 floor。结果强烈依赖 floor：

| floor | dev $E_0$ | val $E_0$ |
| --- | ---: | ---: |
| $10^{-8}$ | 0.911 | 0.875 |
| $10^{-6}$ | 0.911 | 0.875 |
| $10^{-4}$ | 0.927 | 0.899 |
| $10^{-2}$ | **1.012**（$t=3$–$5$ 约 1.03） | **0.998**（约 1.03） |

- 小 floor（$\le10^{-4}$）时 $E_0\approx0.9$、没有平台；
- 大 floor（$10^{-2}$）时 $E_0\approx1.0$、$t=3$–$5$ 有部分平台，但 $t=6$ 掉到 0.90–0.94；
- 也就是说“平台”是**靠正则化调出来的**，不是数据本身的稳定结果。

### 6.3 结论

$H^0/H^4/H^{16}$ 这组算符彼此太相关、矩阵条件数太差；$C(t_0)$ 的近奇异让 GEVP 结果
依赖 floor，无法给出与正则化无关的平台。要突破，需要**真正独立的算符**（不同 $\gamma$
结构、wall、或另一套 smearing 家族）或显著更高的统计量，而不是继续调 floor 或增加 $H^n$
的幂次。

## 7. 状态

多 source 测量工具已可用（`measure_multisource.py`，3 sources、90 组态、约 4 分钟）；
3×3 GEVP 的失败已定位为**条件数/正则化依赖**，不是构造错误。C2 的下一步需要新的算符设计，
仍然 NOT_RUN。

## 8. 还剩下的算符选项

既然 2×2/3×3、两族/三族 Gram GEVP 都只在 $t\le5$ 有平台，下一类候选是真正改变算符基：

| 选项 | 是否改变矩阵 | 成本 | 备注 |
| --- | --- | --- | --- |
| wall source | 只改 source 剖面，不增加 sink 算符维度 | 低（一次反演/配置） | 不能单独构成更大的 GEVP；可作为交叉检查 |
| APE/stout + 不同 $\alpha$ | 已做（两/三族 Gram） | 中 | 平台仍限 $t\le5$ |
| 不同 $\gamma$ 结构 | 改变量子数，**不是同一个 pion 通道** | 低 | 不能用于本任务的 A1+ 参考 |
| 蒸馏/低秩（Peardon 0905.2160） | 是，换成低秩算符基 | 高（需要蒸馏向量与更多收缩） | 真正可能给出独立算符，但是一个新项目 |
| 多 source 位置 | 只增加统计量，不增加算符维度 | 低 | 不能解决条件数 |

结论：**能便宜地做的都做了**；要再有实质突破，需要蒸馏/低秩算符（新项目）或换可观测量（C3 + 独立
anchor）。两者都不在“完成现有 todo”的范围内，需要新的资源与设计决定。
