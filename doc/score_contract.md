# pion 评分契约（P0.4）

本文件固定“证书到底认证了什么”，以及什么情况下必须 abstain。实现见
[`tasks/pion_projection/scoring.py`](../tasks/pion_projection/scoring.py)，示例 spec 见
[`score_spec.example.json`](../tasks/pion_projection/score_spec.example.json)。

## 1. 目标量（estimand）

原指标的 departure 项是

$$
q(t)=\frac{\bar C(t)}{A f_0(t)},\qquad
R=\frac{1}{|W|}\sum_{t\in W}\left[\operatorname{Var}_{\rm boot}(q(t))+\big(q(t)-1\big)^2\right].
$$

它是一个**条件扰动 risk**：给定当前数据、参考拟合和重采样方案，衡量 $q(t)$ 偏离 1 的均方量。
它**不是**频率学 MSE。若目标改成

$$
\mathbb E\big[(q-1)^2\big]=\operatorname{Var}(q)+(q^*-1)^2,
$$

则把 $\operatorname{Var}_{\rm boot}(q)$ 和 $(q-1)^2$ 相加会重复计入估计噪声；$\operatorname{Var}_{\rm boot}$
只是 $q$ 的抽样方差的一个估计。幅度 $A$ 与参考质量相关时，直接“减掉 variance”也不自动得到无偏 MSE。

**当前证书只认证条件量**：同一组 draw、同一参考、同一幅度拟合下，候选的 $R$ 是否同时低于 baseline。
要认证频率学 MSE，需要先做 P0.4 的已知谱比较（旧指标 / cross-fitted loss / 候选修正），并单独声明。

## 2. 证书统计量

$$
D_{cm}^{(r)}=\log\!\big(R_{bm}^{(r)}+\delta^2\big)-\log\!\big(R_{cm}^{(r)}+\delta^2\big),\qquad
L_c=\min_{m\in M}L_{cm},\qquad G_c=e^{L_c},
$$

其中 $L_{cm}$ 是 $D_{cm}$ 的单侧 $\alpha/(JK)$ 分位下界，$K$ 是模型数、$J$ 是候选数。
$\delta^2$ 是**开发阶段**确定的共同分辨尺度（`risk_floor`），不是随候选调整的救分常数。
Bonferroni 只在下界各自有覆盖时提供同时覆盖；它不能修复失效的 bootstrap。

认证条件（全部满足）：

1. 模型集合与 spec 的 `accepted_models` **精确相等**，候选集合与 `candidate_ids` 精确相等；
2. 每个 risk 数组一维、长度一致，并与 `draw_ids` 对齐（若提供 `draw_ids_hash` 则须匹配）；
3. 原始值有限且 $>0$（在加 floor **之前**检查；负值/NaN 记为 invalid，不用 floor 救）；
4. invalid 比例 $\le$ `invalid_tolerance`；
5. baseline 中位数 $\ge$ `min_baseline_risk`（否则 baseline 不可分辨，abstain）；
6. 所有模型的下界都 $\ge \log$ `min_improvement`，且顶层未 abstain。

任一条不满足：该候选 `certified=False`；顶层 abstain 时所有候选都不得以 `certified=True` 暴露。

## 3. 独立样本单位与 draw

- 独立单位是 configuration / Markov block，不是时间点，也不是单个通道；
- 同一 draw 必须同时用于 baseline、所有候选、所有模型；参考参数在每 draw 内重拟合（P0.6 的
  `joint_draws.npz` 目标）；
- 若暂时只接受调用者给的外部 risk draws，本函数只是**条件汇总器**，必须由独立认证器强制 gate；
- 增加 bootstrap draw 数只减少**模拟**误差，不能替代更多组态。

## 4. 有效范围

- 只认证**所列模型集合内**、**所列候选集合内**的改善；不覆盖集合外系统误差，不能在 validation 后删掉
  不利模型；
- 参考本身未通过 gate（如 `s16t16_gevp200`：$\chi^2=90.6/8$）时，证书必须 abstain，历史分数保持 `null`；
- 纯激发态可以拿到极小的旧 q-risk；高 SNR 和非零幅度不证明它是基态。能量主终点还须 anchor 身份和
  隔离证据（C3），当前 **NOT_RUN**。

## 5. 无分数条件（必须 abstain）

| 条件 | 行为 |
| --- | --- |
| `accepted_models` 为空 | `validate_spec` 直接拒绝 |
| 未批准模型 / 候选集合不符 | 抛错，不认证 |
| 模型集合为空或身份字段缺失 | 拒绝/abstain |
| draw 长度不一致、二维数组（广播） | 抛错 |
| 负 risk / NaN / Inf | 记 invalid；比例超限则 abstain |
| baseline 近零（低于分辨尺度） | abstain |
| 参考未认证 / anchor 缺失 | abstain，score `null` |
| 错误激发态、集合外模型 | 不在认证范围，拒绝 |

## 6. 联合 resampling 的实现（P0.6）

`tasks/pion_projection/joint.py` 实现了 4.2/4.3 的联合方案：

1. 每个 draw 先重采样 development blocks，重拟合参考（single / two-state），并在同一份重采样数据内
   bootstrap 参考质量；
2. 同一 draw 的参考质量用于所有候选、所有模型；validation block 重采样也按 draw 共享；
3. 输出 `joint_draws.npz`（draw ID、模型、候选、质量）与 `metric_results.json`（证书、失败原因、
   per-model 下界、point risks）；
4. evaluate 与 joint 共用 `identity.verify_run_identity`（冻结 kind/protocol/dev hash/report 一致性）；
   evaluate 没有 joint draws 时写 `certification=abstain`，不自行发证书；
5. 若 `validation_results.json` 的 `reference_passed=False`，证书强制 abstain，所有候选
   `certified=False`——即使下界很大；
5. spec 可设 `require_draw_ids=True`：缺少 draw_ids 直接拒绝；`draw_ids_hash` 不符也拒绝。

示例（探索性）：`results/pion_projection/joint_20261118/`，200×64，single+twostate 两个模型，
因参考 gate 失败而 abstain。
