# GEVP 的统计估计（P0.2/P0.3）

实现：[`tasks/pion_projection/gevp_stats.py`](../tasks/pion_projection/gevp_stats.py)；
运行：[`gevp_bootstrap.py`](../tasks/pion_projection/gevp_bootstrap.py)；
产物：[`results/pion_projection/gram_stats/`](../results/pion_projection/gram_stats/)。

## 1. 估计顺序

1. **去重**：剔除矩阵完全相同的算符（$H^0$ 在每个族里都出现，旧代码把结构零模 floor 后当新自由度）；
2. **block bootstrap**：按 configuration block（长度 5）重采样，得到 $B=200$ 个 replica；
3. **replica 平均矩阵**：每个 replica 内先平均 $C_{ij}(t)$，**不在配置层面解 GEVP 再平均能量**；
4. **rank 截断**：对 $C(t_0)$ 做本征分解，只保留预声明的秩（已知谱用真实秩，真实数据用谱的有效秩），
   在保留子空间里白化后解 $C(t)v=\lambda C(t_0)v$；
5. **有效能量**：forward 估计 $E=-\log\lambda/(t-t_0)$；cosh 估计解
   $\cosh(E(t-T/2))/\cosh(E(t_0-T/2))=\lambda$（含有限 $T$ 回绕）；
6. **相关平台**：每个 replica 先在平台上平均 $E(t)$，再对 replica 分布取标准误——不再用“逐时间标准差 / √时间点数”。

失败 replica（非正本征值、空秩、cosh 求根失败、平台时间缺失）单独计数，不静默丢弃；
`summarise` 使用**固定平台时间集合**，任一所需时间不是有限值就把该 replica 整体记为失败，
并分别输出局部失败数、整 replica 失败数与每 replica 的有效时间数。

## 2. 已知谱验收

合成模型：三算符、两态谱 $E=(0.80,1.30)$、$T=16$、60 组态、2% 逐元素噪声、forward-only。
100 个独立数据集、每数据集 200 个 block replica：

| 量 | 结果 |
| --- | --- |
| $E_0$ 偏差 | $-2.4\times10^{-4}$（约 2 个 MC 标准误） |
| 95% 区间覆盖率 | **0.95**（100 次模拟，二项 MC 误差约 ±0.022） |
| 失败 replica | 0 |

即：估计器在已知谱上无偏、覆盖率与名义值一致；增加 draw 数只减少模拟误差。

## 3. 真实数据（Gram，探索性）

`s16t16_gram3` 开发集、12 blocks、200 replicas、秩截断 4：

| 估计器 | $E_0$ | $E_0(t=3,4,5,6)$ |
| --- | ---: | --- |
| forward | $1.0629\pm0.0075$ | 1.078 / 1.067 / 1.059 / 1.049 |
| cosh | $1.0640\pm0.0075$ | 1.078 / 1.067 / 1.059 / 1.052 |

- 平台从 1.078 漂到 1.049（约 4 个 bootstrap SE），不是纯噪声；$E_0\approx1.06$，介于两态 1.057 与单 cosh 1.086 之间、偏向两态；
- **但这仍是 sink–sink Gram 矩阵**（见 [`gram_operator_identity.md`](gram_operator_identity.md)），不是标准 source–sink 变分矩阵，
  所以只能标为探索性，不能作为“偏向两态”的独立证据；
- 标准矩阵（multisource）需要先把源端核与 sink 对齐（$C-C^\dagger<5\%$），再重跑本估计器。

## 4. 图

![gram rank and bootstrap](../results/pion_projection/gram_stats/gevp_bootstrap.png)

左图：$t_0=2$ 的 metric 谱（原始 9 个、去重后 8 个）；右图：forward/cosh 的 $E_0(t)$ 及 bootstrap 误差棒。
两张图都只支持“估计器行为”，不支持物理认证。
