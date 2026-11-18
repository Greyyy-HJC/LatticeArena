# GEVP 矩阵到底是什么（P0.1）

本文件回答：`measure_gram.py` 的 $M_{ij}(t)=\langle K_iS,K_jS\rangle$ 是不是标准变分矩阵
$C_{ij}(t)=\langle O_i(t)O_j^\dagger(0)\rangle$？结论：**只有在平移不变、彼此对易的核族上才近似成立**；
真实数据上两者差约 30%，所以旧 Gram GEVP 只能算探索性估计。

## 1. 两种矩阵

设 $O_i(t)=\sum_x \bar q(x,t)\gamma_5 K_i q(x,t)$，$S(x;0)$ 是点源传播子。

**标准 source–sink 矩阵**（sink 核 $K_i$ 在两条腿上，source 核 $K_j$ 在源端）：

$$
C_{ij}(t)=-\sum_x \operatorname{tr}\!\Big[K_i S(x;0)K_j\,\big(S(x;0)K_j\big)^\dagger\Big]
=\big\langle K_i\,S K_j,\;S K_j\big\rangle .
$$

它需要**源端涂抹后的传播子** $S K_j$（`measure_multisource.py` 测的就是这个）。

**sink–sink Gram 矩阵**（`measure_gram.py`，两个核都作用在 sink 空间）：

$$
M_{ij}(t)=\sum_x\operatorname{tr}\!\Big[(K_iS)(x)^\dagger (K_jS)(x)\Big]
=\big\langle S,\,K_iK_jS\big\rangle .
$$

$M$ 对任意 Hermitian $K$ 都严格 Hermitian、半正定；但这不等于它就是标准矩阵。

## 2. 什么时候 $M=C$

若 $K_i,K_j$ 是平移不变的（卷积核）且对易，则 $S K_j=K_jS$、$K_iK_j=K_iSK_j$，在无限体积/周期边界
（忽略边界）下 $M_{ij}=C_{ij}$。$H^n$ 恰好属于这一类：$K_iK_j=K_{i+j}$，于是 $M$ 就是 Hankel 矩阵
$C_{0,n_i+n_j}$——这正是旧 Hankel 构造成立的原因。

对**不同族**（$H^n$ 与 APE-$H^n$）或源端与 sink 端核不一致时，$M\ne C$。

## 3. 真实数据上的数值检查

用同一批 60 个开发组态比较两套测量（`results/pion_projection/gram_identity_check.json`，
脚本 `tasks/pion_projection/gram_identity_check.py`）：

| $t$ | $\max|M-C|/\max|C|$ | $M$ 的反对称部分 | $C$ 的 anti-Hermitian 部分 |
| --- | ---: | ---: | ---: |
| 3 | 0.294 | 0 | 0.294 |
| 4 | 0.301 | 0 | 0.301 |
| 5 | 0.303 | 0 | 0.303 |
| 6 | 0.304 | 0 | 0.304 |

两点结论：

1. $M$ 与 $C$ 差约 **30%**，不是边界效应能解释的；$M$ 本身严格对称（反对称部分为 0）。
2. $C$ **本身就不是 Hermitian**：$C-C^\dagger$ 的大小与 $M-C$ 相同。说明源端 `gaussian12` 的
   $K_j$ 与 sink 端的 $H^n$ 并没有精确对应（参数/约定/标签之一不一致），不能先用 $(C+C^\dagger)/2$
   把问题藏起来。

## 4. 对旧结果的影响

- `s16t16_gram`/`gram3` 的 GEVP 用的是 $M$，即 **sink–sink Gram**，不是标准变分矩阵；
  其 $E_0\approx1.02$–$1.05$ 只能标为探索性，不能作为“偏向两态参考”的独立证据（与 todo 一致）。
- `s16t16_ms` 的 source–sink 矩阵是**正确的标准矩阵**，但必须先把源端涂抹核与 sink 核对齐
  （验证 $C_{ij}=C_{ji}^*$ 到几个百分点以内），才能用于变分定理。
- 旧的 2×2/3×3 Hankel 是 $M$ 的对易特例（$K_iK_j=K_{i+j}$），它的“无平台”结论仍然成立。

## 5. 下一步（P0.2/P0.3）

1. 用和 sink 完全相同的 $H^n$ 构造源（而不是 `gaussian12` 的近似），重测 source–sink 矩阵；
2. 验收：$C-C^\dagger$ 的范数 $<5\%$，否则拒绝该矩阵；
3. 然后按 P0.2 的顺序（共享 block 重采样 → replica 平均 → 开发冻结子空间 → GEVP → mode matching →
   相关平台）重做，并用已知谱生成器验证覆盖。
