# 静态势：v3 结果报告（2026-11-18）

第二个任务的独立结果报告。对应 `tasks/wilson_loop/`，协议 `protocol_v3.json`，
产物 `results/wilson_loop/s16t16_v3/` 与 `paired_s16t16_v3/`。

## 1. 设置

- **系综**：新生成 200 个 $16^4$、$\beta=6.0$ 纯规范组态（`data/S16T16B`，HMC，接受率 97.7%）；
- **划分**：开发 0–99，gap 100–119，留出 120–199（**80 组态 = 16 blocks**，block 长度 5）；
- **可观测量**：空间 Wilson 圈 $W(R,T)$，$R=1,2,3$，$T=4..8$，对空间点、3 个方向、16 个时间原点平均；
- **核**：identity，以及协变 APE（$\alpha/4$、双边极分解投影、三个空间方向）：
  APE(0.4, 5)、APE(0.4, 10)、APE(0.6087, 10)（后者的 staple/identity 比例对应 v1 的 $\alpha=0.7$）；
- **势**：$V(R)=-\partial_T\log W(R,T)$，在名义窗口 $T=4,5,6,7$ 上线性拟合；
- **risk**：$\mathrm{mean}_R[(\mathrm{SE}_{\rm boot}(V)/V_{\rm ref})^2+(\mathrm{spread}_{\rm windows}(V)/V_{\rm ref})^2]$，
  $V_{\rm ref}$ 取开发集 identity；
- **参考**：identity 的 $V(R)>0$ 且窗口相对散布 $\le5\%$——本 run **通过**。

## 2. 势与误差

| 核 | $V(1)$ | $V(2)$ | $V(3)$ | SE(3) |
| --- | ---: | ---: | ---: | ---: |
| identity | 0.4107 | 0.5971 | 0.7150 | 0.0073 |
| APE(0.4, 5) | 0.4106 | 0.5981 | 0.7086 | 0.0035 |
| APE(0.4, 10) | 0.4106 | 0.5973 | 0.7066 | 0.0034 |
| APE(0.6087, 10) | 0.4106 | 0.5970 | 0.7060 | 0.0034 |

![potential](../results/wilson_loop/s16t16_v3/potential.png)

$V(1)$ 几乎不变，$V(3)$ 的统计误差从 0.0073 降到 0.0034（约 2 倍），中心值从 0.715 移到
0.706——涂抹主要改善的是大 $R$ 处的信噪比，这符合预期（空间 Wilson 圈在 $R$ 大时最需要平滑）。

## 3. 评分与配对检验

| 核 | 留出 risk | 留出 score |
| --- | ---: | ---: |
| identity | $5.625\times10^{-5}$ | 100 |
| APE(0.4, 5) | $2.136\times10^{-5}$ | **263.3** |
| APE(0.4, 10) | $2.416\times10^{-5}$ | 232.8 |
| APE(0.6087, 10) | $2.535\times10^{-5}$ | 221.9 |

预注册配对分析（500 outer × 500 inner，统计量 risk(identity)/risk(candidate)）：

| 核 | 中位数 | 95% 区间 | $P(\text{优于 identity})$ |
| --- | ---: | ---: | ---: |
| APE(0.4, 5) | 3.08 | [1.32, 5.93] | 0.986 |
| APE(0.4, 10) | 2.94 | [1.25, 5.82] | 0.982 |
| APE(0.6087, 10) | 2.84 | [1.17, 6.34] | 0.982 |

![paired](../results/wilson_loop/paired_s16t16_v3/score_paired.png)

三个区间均不含 1：**协变 APE 相对 identity 的改善在预注册分析下成立**。

## 4. 与前两版的对比

| 版本 | 留出组态 / blocks | 结论 |
| --- | --- | --- |
| v1（旧核，Gram–Schmidt） | 30 / 6 | 核不协变，结果作废 |
| v2（协变核） | 30 / 6 | 点估计 3×，但区间跨 1 → 未证明 |
| **v3（协变核 + 200 系综）** | **80 / 16** | **区间不含 1 → 显著** |

v2 与 v3 的点估计一致（约 3 倍），差别只在统计量：6 个 block 的功效不足以区分 3–5% 的
分数差异，16 个 block 可以。

## 5. 限制

- 纯规范、单一体积 $16^4$、$\beta=6.0$、$\kappa$ 无关（静态势不涉及夸克）；
- $R$ 只到 3，$T$ 窗口只有 4–8；更大 $R$ 的信噪比和有限体积效应未检查；
- 势的拟合是无权线性回归，不是相关 $\chi^2$；
- 参考 gate 只检查正势与窗口散布，不是完整的谱模型检验；
- 留出集在同一台机器、同一天测量，与开发集共享生成链，不能当作独立盲测。

## 6. 复现

```bash
# 生成 200 组态（约 10.5 分钟）
sbatch pilot/hmc200.sbatch
# 测量（每个 split 一个 job，约 6.5 s/组态）
WILSON_PROTOCOL=tasks/wilson_loop/protocol_v3.json sbatch pilot/wilsonv3.sbatch \
  --output results/wilson_loop/s16t16_v3 measure --split development
WILSON_PROTOCOL=tasks/wilson_loop/protocol_v3.json sbatch pilot/wilsonv3.sbatch \
  --output results/wilson_loop/s16t16_v3 measure --split validation
# 开发、评分、配对
WILSON_PROTOCOL=tasks/wilson_loop/protocol_v3.json python -m tasks.wilson_loop \
  --output results/wilson_loop/s16t16_v3 develop
WILSON_PROTOCOL=tasks/wilson_loop/protocol_v3.json python -m tasks.wilson_loop \
  --output results/wilson_loop/s16t16_v3 evaluate
WILSON_PROTOCOL=tasks/wilson_loop/protocol_v3.json WILSON_RUN=results/wilson_loop/s16t16_v3 \
  python -m tasks.wilson_loop.paired
```

## 7. 扩展：加入 $R=4$（v4）

同一个 200 组态系综上把 $R$ 扩到 4（`protocol_v4.json`，留出 80 组态 / 16 blocks）：

| 核 | $V(4)$ | SE(4) | 留出 risk | 留出 score |
| --- | ---: | ---: | ---: | ---: |
| identity | 0.8022 | 0.0233 | $2.79\times10^{-4}$ | 100 |
| APE(0.4, 5) | 0.7840 | 0.0068 | $4.20\times10^{-5}$ | 663.3 |
| APE(0.4, 10) | 0.7844 | 0.0059 | $3.74\times10^{-5}$ | 745.8 |
| APE(0.6087, 10) | 0.7846 | 0.0053 | $3.45\times10^{-5}$ | **808.0** |

配对分析（500×500）：中位数 6.6 / 8.2 / 9.5，95% 区间 [2.49, 21.6] 等，**全部不含 1**，
$P(\\text{优于 identity})=1.0$。

- 在 $R=4$ 上涂抹的收益最大：identity 的 SE 是 0.0233，APE 约 0.005–0.007（3–4 倍改善）；
- $V(4)$ 的中心值从 0.802 移到 0.784；
- **但分数尺度与 v3 不可比**：$R=4$ 的 identity 噪声把 baseline risk 抬高了 5 倍（$2.8\\times10^{-4}$
  vs $5.6\\times10^{-5}$），所以 663–808 不是“比 v3 更好”，而是换了 $R$ 集合后的新刻度。

结论：v4 把“涂抹对大 $R$ 最有用”这件事量化了，也再次确认改善显著；正式报告仍以 v3 为主，
v4 作为 $R$ 依赖的补充。
