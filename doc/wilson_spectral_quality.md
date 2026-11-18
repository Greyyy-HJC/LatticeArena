# Wilson 圈的门控能证明什么（P3）

## 1. 有限 T 的静态源谱

固定空间分离 $R$ 的 Wilson 圈在时间方向有 transfer matrix 谱：

$$
W(R,T)=\sum_n c_n(R)\,e^{-E_n(R)\,T},
\qquad E_0(R)=V(R)+\text{const}.
$$

有效势 $V_{\rm eff}(T)=-\log\frac{W(R,T+1)}{W(R,T)}\to E_0(R)$ 当 $T\to\infty$，
激发态贡献是 $O(e^{-\Delta E\,T})$。

**关键**：这是“有限 T 的谱展开”，**不能**直接套 pion 的 $\cosh$ 形式，也不能假设回绕项
$\exp[-V(T_t-2T)]$ 总是小量。在 $T=T_t/2$ 时它等于 1——旧文档里“回绕 1e-4–1e-6”的普适断言
没有依据，已删除。

## 2. risk 测的是什么

当前 risk：

$$
\mathcal R=\frac1{|R|}\sum_R\left[\left(\frac{\mathrm{SE}_{\rm boot}(V(R))}{V_{\rm ref}(R)}\right)^2
+\left(\frac{\mathrm{std}_{\rm windows}(V(R))}{V_{\rm ref}(R)}\right)^2\right].
$$

它衡量的是**固定窗口下的精度**（SE）与**窗口间稳定性**（spread），不是 $E_0$ 的准确度。

### 反例（合成，已跑）

令 $W(T)=e^{-(V_0+\delta)T}$（所有窗口完全一致），$V_0=0.5$、$\delta=0.04$、噪声 1%：

| 量 | 值 |
| --- | ---: |
| 测得 $V$ | $0.5399\pm0.0002$ |
| 真 $E_0$ | 0.50 |
| 偏差 | 0.04（**170 个 SE**） |
| 窗口 spread | $<10^{-3}$ |
| risk | $2.7\times10^{-7}$ |

也就是说：**gate 会通过，risk 极小，但答案错了 170σ**。risk 只要求“平”，不要求“对”。

## 2.1 两态反例（`spectral_physical_v1/`）

$W(T)=c_0e^{-E_0T}+c_1e^{-E_1T}$、$E_0=0.5$、$E_1=1.2$、0.5% 噪声、16 blocks：

| $c_1$ | bias/SE | spread/V | max\|window z\| | spread gate |
| ---: | ---: | ---: | ---: | --- |
| 0.05 | 3.1 | 0.0004 | 1.12 | 通过 |
| 0.1 | 5.6 | 0.0014 | 2.49 | 通过 |
| 0.2 | 13.1 | 0.0019 | 5.49 | 通过 |
| 0.5 | 32.5 | 0.0058 | 14.94 | 通过 |

**spread gate 看不见共同偏差**；未使用的 `window_z` 在 $c_1\ge0.2$ 才能察觉。
有限 $N_t$ 的谱为 $W(R,T)=\sum_n c_n(e^{-E_nT}+e^{-E_n(N_t-T)})$，回绕项在 $T=N_t/2$ 与正向同量级；
但等幅回绕只是特例，不能反过来断言 Wilson 圈必含该项——本反例用零温形式。

## 3. v3/v4 的窗口值

v3（16 blocks、80 留出组态）identity 的三个窗口：

| $R$ | $V_{\rm win1}$ | $V_{\rm win2}$ | $V_{\rm win3}$ | spread |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.4107 | 0.4107 | 0.4105 | $1\times10^{-4}$ |
| 2 | — | — | — | $2.7\times10^{-3}$ |
| 3 | — | — | — | $2.8\times10^{-3}$ |

v4 加 $R=4$ 后 spread 到 $9.5\times10^{-3}$。窗口间一致只能说明**没有明显的 T 依赖**，
不能排除一个所有窗口共享的偏移（共同错误平台）。

## 4. 四类主张必须分开

| 主张 | risk 能测吗 |
| --- | --- |
| 降低当前窗口的 risk | 能（定义如此） |
| 降低 SE | 能（第一项） |
| 改善 overlap（提高 $c_0$） | 间接，需谱分析 |
| 提高 $E_0$ 的准确度 | **不能**（需要独立 anchor 或更晚 $T$） |

因此正式报告只能说“在当前固定窗口的 risk 上更好”，不能把 gate PASS 升级为“$E_0$ 更准”。

## 5. 门控当前含义

- `reference_passed` = identity 势全正 **且** 窗口 spread$/V\le$ `reference_max_relative_se`；
- `reference_max_window_z` 仍是未使用字段（todo 要求同步语义或删除；当前保留并注明）；
- `reference_nominal_window` 现在是显式 spec（不再是硬编码 `windows[1]`），`risk(..., nominal=...)` 使用它，
  越界会拒绝。
