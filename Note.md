# LatticeArena 使用说明

> 一句话：**LatticeArena 是一个"格点 QCD 科研考试系统"**——它出题（设计一个更好的π介子 sink）、提供标准考场（固定的测量和打分流程）、最后给你一个分数。
>
> 这份文档尽量用大白话，从"它是什么"讲到"怎么在远端跑""LLM 到底在哪里"。
> 读完你应该能：看懂每个文件、自己跑一遍、知道分数为什么有时是 `null`、知道怎么把大模型接进来。

**怎么用这份文档（按需要跳读）：**

| 你想干什么 | 看哪节 |
| --- | --- |
| 5 分钟搞懂它是啥 | §1 + §2 |
| 想知道每个文件干嘛 | §3 |
| 想自己跑一遍 | §4 + §6 |
| 想在远端集群跑 | §7 |
| 想把 LLM 接进来 | §8（先看 8.1 的结论） |
| 想知道为什么没分 / 分怎么算 | §5 |
| 想加一个新任务 | §9 |
| 报错了 | §11 |

---

## 目录

1. [它是什么（大白话）](#1-它是什么大白话)
2. [整体架构（一张图）](#2-整体架构一张图)
3. [目录和每个文件的作用](#3-目录和每个文件的作用)
4. [一次完整运行发生了什么](#4-一次完整运行发生了什么)
5. [分数到底怎么算](#5-分数到底怎么算)
6. [在本机跑（能跑什么、不能跑什么）](#6-在本机跑能跑什么不能跑什么)
7. [在 kunshan 远端跑，并拿到 LatticeArena 的分](#7-在-kunshan-远端跑并拿到-latticearena-的分)
8. [LLM 是怎么被调用的（重点）](#8-llm-是怎么被调用的重点)
9. [怎么加一个新任务](#9-怎么加一个新任务)
10. [名词表](#10-名词表)
11. [排错清单](#11-排错清单)
12. [一页速查](#12-一页速查)

---

## 1. 它是什么（大白话）

### 1.1 一句话

格点 QCD（lattice QCD）里有很多"调参/设计"问题：怎么设计一个更好的夸克源、更好的 sink、更好的拟合窗口……LatticeArena 把这类问题包装成**有标准答案、有固定评分规则**的任务，让 AI（或人）去提交方案，然后自动打分。

当前仓库里**只有一个任务**：`pion_projection`（π 介子 sink 设计）。

### 1.2 打个比方：一场考试

| 考试里的东西 | LatticeArena 里对应的东西 |
| --- | --- |
| 考试大纲 | `protocol.json`（所有数字都写死在这里） |
| 考场和监考 | `measure.py`（固定测量流程，考生不能改） |
| 标准答案的"参考线" | `statistics.py` 里的参考拟合（一个共用的基态质量） |
| 考生交的答卷 | 一个 `build_kernel(...)` 函数 / 一个 JSON 定义 |
| 阅卷老师 | `experiment.py` 的 `evaluate`（按固定公式算 loss） |
| 成绩单 | `validation_results.json`（score 可能是 `null`） |
| 及格线/免考规则 | 各种 "gate"（参考不稳就不给分） |

### 1.3 它**不是**什么

- 不是 QUDA/PyQUDA 那种"算物理量"的库。它自己不做格点计算，而是**调用** PyQUDA 去算。
- 不是通用的 benchmark 框架。目前没有任务注册表、没有排行榜、没有多任务调度，就是一个任务的一整套流水线。
- **默认流程不是 LLM 应用**：`measure`/`develop`/`evaluate`/`candidate` 都不调用大模型；只有可选的 `agent`/`report` 子命令会（详见第 8 节）。

---

## 2. 整体架构（一张图）

```
                        ┌─────────────────────────────┐
                        │  1. 规范场组态 (gauge)       │
                        │  NERSC 文件 wilson_b6.*      │
                        └──────────────┬──────────────┘
                                       │  只读
                        ┌──────────────▼──────────────┐
                        │  2. measure.py（可信测量）   │  ← 只有出题人能跑
                        │  · Wilson 反演 12 列          │
                        │  · 12 列残差检查             │
                        │  · 传播子缓存（带哈希）       │
                        │  · 7 个基矢 H^n 的关联函数    │
                        └──────────────┬──────────────┘
                                       │  development.npz / validation.npz
                        ┌──────────────▼──────────────┐
                        │  3. develop（出参考+冻结）    │  ← 只看 development
                        │  · 拟合共用质量 m0            │
                        │  · 7 个基线 + 优化器混合       │
                        │  · 写 submission.json（冻结） │
                        └──────────────┬──────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
┌───────▼────────┐          ┌──────────▼──────────┐        ┌──────────▼─────────┐
│ 考生 A：人/LLM │          │ 考生 B：候选 JSON    │        │ 考生 C：别的核      │
│ 写 build_kernel│          │ candidate.py 导入    │        │ 同一个入口          │
└───────┬────────┘          └──────────┬──────────┘        └──────────┬─────────┘
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                       │  只在 development 上跑，冻结
                        ┌──────────────▼──────────────┐
                        │  4. evaluate（阅卷）         │  ← 只看 validation
                        │  · 算 loss（方差 + 偏离）     │
                        │  · 200×128 嵌套 bootstrap    │
                        │  · 检查 gate，决定给不给分    │
                        └──────────────┬──────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │  5. validation_results.json  │
                        │  score: 数字 或 null         │
                        └──────────────┬──────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │  6. plots.py → 图             │
                        └─────────────────────────────┘
```

**三层职责**：

| 层 | 文件 | 谁能改 | 干什么 |
| --- | --- | --- | --- |
| 任务定义层 | `protocol.json`、`doc/pion_projection.md`、`tasks/pion_projection/README.md` | 出题人 | 规定考什么、怎么算分 |
| 可信测量层 | `measure.py`、`kernels.py`（参考核）、`common.py` | 出题人 | 把组态变成关联函数，缓存，校验 |
| 统计评分层 | `statistics.py`、`experiment.py`、`candidate.py` | 出题人 | 拟合参考、算 loss、冻结、打分 |
| 考生层 | 你自己写的 `build_kernel` 或 JSON | **考生（人/LLM）** | 只提交"核"，别的一律不能碰 |

---

## 3. 目录和每个文件的作用

### 3.1 目录树

```
LatticeArena/
├── README.md                  项目门面：这是什么、第一个任务、怎么起步
├── requirements.txt           Python 依赖（不含 PyQUDA，PyQUDA 要单独编译）
├── pytest.ini                 pytest 配置：只扫 tests/
├── .gitignore                 忽略 .venv/.cache/results 等
├── scripts/
│   └── run_pion.sh            主机专用启动脚本（硬编码了别人的路径，基本别用）
├── doc/
│   ├── environment.md         环境搭建：PyQUDA 编译（QUDA 假定已装）、MPI、冒烟测试
│   ├── pion_projection.md     任务的科学定义（最重要的一份，302 行）
│   ├── pion_projection_results.md  第一次真实实验报告（score = null）
│   └── kunshan_dcu_run.md     上一次在 kunshan DCU 上试跑的报告
├── core/                      框架层（可被所有任务复用）
│   └── io.py                  digest / fingerprint / write_json
├── agents/                    考生层：LLM 作答与解读
│   ├── llm.py                 调 DeepSeek（key 在 ~/Desktop/api.key）
│   ├── loop.py                agent 子命令：让 LLM 提权重并打分
│   └── report.py              report 子命令：前三名 + LLM 解读
├── tasks/
│   ├── __init__.py            空文件，告诉 Python "tasks 是个包"
│   ├── pion_projection/
│       ├── README.md          这个任务的"考生须知"（命令、契约、限制）
│       ├── __init__.py        空
│       ├── __main__.py        命令行入口（5 个子命令）
│       ├── common.py          门面：复用 core.io，保留 protocol/indices/load_data
│       ├── kernels.py         参考答卷：扩散核 H^n 的多项式
│       ├── measure.py         可信测量：反演、检查、缓存、算 7 个基矢
│       ├── statistics.py      数学工具：折叠、分块、协方差、参考拟合、bootstrap、loss
│       ├── experiment.py      develop（出参考+冻结）/ evaluate（留出打分）
│       ├── candidate.py       通用交卷入口：导入你的模块、冻结、打分
│       ├── plots.py           画 4 面板诊断图
│       ├── protocol.json      所有数字的唯一真源
│   │   └── submissions/
│   │       └── local.json     一份"示例答卷"（恒等核），**代码其实不读它**
│   └── wilson_loop/           第二个任务：静态势（Wilson 圈、APE 核、平台稳定性打分）
├── tests/
│   └── test_pion_projection.py  CPU 单元测试（含 test_optimize_gaussian/test_wilson_loop 共 51 项）
└── results/                   跑出来的结果（被 .gitignore 忽略）
    ├── pion_projection/s16t16_v1/
    └── kunshan_logs/
```

### 3.2 逐个文件说明

#### 根目录

| 文件 | 干什么 | 备注 |
| --- | --- | --- |
| `README.md` | 项目介绍、评价哲学、第一个任务、legacy 分支说明 | 里面 "legacy 分支" 的说法是**错的**：仓库只有 `main`，旧实现在提交 `f62c613` 里 |
| `requirements.txt` | 钉死版本的 Python 依赖（numpy/scipy/matplotlib/pytest/cupy/mpi4py…） | PyQUDA 不在里面，要按 `doc/environment.md` 自己编译 |
| `pytest.ini` | 告诉 pytest 只去 `tests/` 找测试 | 所以测试文件放在根目录 `tests/`，不是每个任务自己带 |
| `.gitignore` | 忽略 `.venv/`、`.cache/`、`results/`、`__pycache__` | 注意：**`results/` 被忽略**，结果默认不会进 git |
| `scripts/run_pion.sh` | 别人主机上的一键脚本 | 第 4 行 `source /home/jinchen/env/pyquda.sh`，换机器必挂；在自己的机器上请自己写 |

#### `doc/`

| 文件 | 干什么 |
| --- | --- |
| `environment.md` | 怎么建 `.venv`、怎么编译 PyQUDA（QUDA 假定已经装好）、MPI 怎么配、最后一个单组态冒烟测试的完整代码。它是"环境说明书"，不是评测代码 |
| `pion_projection.md` | **任务的科学定义**：物理目标、固定参数、允许优化什么、提交接口、打分公式、验收检查。想认真做题就读这一份 |
| `pion_projection_results.md` | 第一次真实实验的记录：7 个基矢、agent 混合、7.03 的 loss 比、为什么 score 是 null |
| `kunshan_dcu_run.md` | 上一次在昆山 DCU 上从零产组态到跑完的完整报告（含踩坑） |

#### `tasks/pion_projection/`

| 文件 | 行数 | 干什么 |
| --- | ---: | --- |
| `__main__.py` | 35 | 命令行入口。5 个子命令：`measure` / `develop` / `evaluate` / `plot` / `candidate`。它是**懒加载**的——`plot` 不会去 import cupy，所以没 GPU 也能看帮助 |
| `common.py` | 32 | 门面：`digest/fingerprint/write_json` 从 `core.io` 再导出；自己保留 `protocol()`、`indices()`、`load_data()`（这三个和任务约定绑定）。原有 import 全部不用改 |
| `kernels.py` | 57 | **参考答卷**：`DiffusionKernel` 实现 `K = Σ w_n H^n`，`H = (I + 0.2·hop)/2.2`。`apply(field)` 是考生接口。同时支持 NumPy 和 CuPy |
| `measure.py` | 133 | **可信测量**（只有出题人跑）：读 NERSC 组态 → 建 Wilson Dirac → 反演 12 个 spin-color 列 → 逐列检查残差 → 缓存传播子（带哈希+来源校验）→ 用 QUDA 的 `wuppertalSmear` 增量算 7 个基矢的关联函数 → 第一条组态上还额外用"考生接口"重算一遍做交叉验证 → 组态预算超时检查 → 每测完一个组态就写一次 checkpoint |
| `statistics.py` | 131 | **数学核心**：`fold`（正反时间平均）、`blocks`（每 5 个组态一块）、`shape`（cosh 形状）、`inverse_covariance`（协方差求逆 + 25% 对角收缩 + 特征值地板）、`ReferenceFit`（共质量、分幅度拟合）、`bootstrap_indices`、`risk`（算 loss 的两个罚分）、`effective_mass`（画图用） |
| `experiment.py` | 183 | **出题人流程**：`develop` = 拟合参考 → 检查 gate → 7 个基矢基线 → SLSQP 优化带符号混合 → 冻结 `submission.json`；`evaluate` = 校验哈希 → 在 validation 上算 loss → 200×128 嵌套 bootstrap → 决定给不给分；`nested_intervals` 是嵌套区间 |
| `candidate.py` | 143 | **通用交卷入口**：读一个 JSON（`builder: "模块:函数"` + `parameters`）→ 导入你的模块 → 把源码哈希冻进身份 → 在 development 上跑并冻结 → 在 validation 上算分 |
| `plots.py` | 55 | 把结果画成 4 张图：偏离、SNR、有效质量、参考窗口稳定性 |
| `protocol.json` | 39 | **所有数字的唯一真源**：系综路径、几何、split、κ、容差、窗口、bootstrap 次数、gate 阈值、预算……改它 = 换协议 |
| `submissions/local.json` | 8 | 恒等核的示例 JSON。**没有任何代码读它**，只是给你复制改的模板 |

#### `tests/`

| 测试 | 检查什么 |
| --- | --- |
| `test_kernel_covariance_linearity_hermiticity_and_translation` | 核的规范协变、线性、厄米、平移、立方旋转、宇称 |
| `test_risk_scaling_noise_and_excited_state` | loss 对整体缩放不变；噪声变大 → 方差项变大；激发态 → 偏离项变大 |
| `test_invalid_inputs_and_folding` | 零权重、NaN、块数不够、折叠公式、错的 link 形状都要报错 |
| `test_reference_recovers_synthetic_cosh` | 给一个已知 m=0.8 的合成数据，参考拟合要能还原 |
| `test_block_bootstrap_degenerate_outer_draw_is_not_dropped` | 退化重采样不能被悄悄丢掉 |
| `test_development_cannot_reopen_after_validation` | 有 validation 数据后不许回头再 develop |
| `test_zero_overlap_rejected_and_missing_rows_not_ignored` | 零信号要被拒；缺行不能被忽略 |

这 7 个测试**完全不需要 GPU**，`python -m pytest -q` 一两秒跑完。

#### `results/`

跑出来的东西。默认被 `.gitignore` 忽略。

- `pion_projection/s16t16_v1/`：`development.npz`、`validation.npz`、`submission.json`、`validation_results.json`、`nested_bootstrap.npz`、`pilot_results.png/pdf`、`*_provenance.json`
- `kunshan_logs/`：上一次远跑留下的日志和组态清单

---

## 4. 一次完整运行发生了什么

记住顺序：**先测 development → 再 develop → 再测 validation → 再 evaluate → 最后 plot**。顺序不能反，代码会拦你。

### 第 1 步：`measure --split development`

```bash
python -m tasks.pion_projection --output results/pion_projection/s16t16_v1 measure --split development
```

- **输入**：`protocol.json` 里的系综路径 + 组态编号 0–29
- **做什么**：
  1. 读组态（NERSC 格式，自动校验收缩/链接迹）
  2. 建 Wilson Dirac（κ=0.14，反周期时间边界）
  3. 反演原点 point source 的 12 个 spin-color 列
  4. 每一列都算一次 `||Dψ - b|| / ||b||`，必须 ≤ 1e-10
  5. 把传播子存进 `.cache/pion_projection/propagators/`，文件名为内容的哈希，旁边存一份来源信息（组态哈希、κ、边界、容差、软件版本）
  6. 用 `wuppertalSmear` 增量算 7 个基矢 `H^n`（n=0,2,4,8,16,32,64）的关联函数
  7. 断言 `H^0` 等于 `Σ|S|²`（这是物理恒等式，防止收缩写错）
  8. 在第一条组态上，再用**考生接口**算一遍 `H^2`，和 QUDA 的结果比，误差 ≤ 1e-10
  9. 每测完一个组态写一次 `development.npz`（中途断了也能看到进度，但半截数据 `load_data` 会拒绝）
- **输出**：`development.npz`（形状 `(30, 7, 16)`）+ `development_provenance.json`
- **失败会怎样**：残差超标、几何不对、缓存哈希不对、超时（300 秒/组态）都会直接抛错

### 第 2 步：`develop`

```bash
python -m tasks.pion_projection --output run develop
```

- **输入**：只有 `development.npz`
- **做什么**：
  1. 折叠（正反时间平均）→ 分块（每 5 个组态一块，30 个组态 = 6 块）
  2. 用 3 个通道（H^0, H^8, H^16）在 3 个窗口（4–8, 5–8, 6–8）分别拟合一个**共用质量** m0
  3. 检查 5 个 gate：质量相对误差 ≤5%、拟合 p ≥0.01、窗口漂移 ≤2.5σ、没撞到质量边界、前后向对称 z ≤4
  4. 算 7 个单基矢的 loss，选出最好的（上次是 `H^32`）当"调好的 Gaussian 基线"
  5. 用 SLSQP 在 7 个系数上优化一个带符号混合（`Σu=1`、`Σ|u|≤8`、8 个起点）
  6. 把所有东西（权重、幅度、协方差、哈希、gate 结果）写进 `submission.json`
- **输出**：`submission.json`（冻结）、`development_results.json`、`reference_bootstrap.npz`
- **注意**：如果 `submission.json` 或 `validation.npz` 已存在，它**拒绝再跑**（防止偷看考卷后改答案）

### 第 3 步：`measure --split validation`

和第一步一样，只是组态编号换成 35–49（30–34 是 gap，跳过不用）。它会先检查 `submission.json` 的协议哈希和 development 数据哈希，对不上就拒绝——保证"先冻结、后看考卷"。

### 第 4 步：`evaluate`

- **输入**：`development.npz` + `validation.npz` + `submission.json`
- **做什么**：
  1. 校验 4 个哈希（协议、development、`statistics.py`、`kernels.py`）——防止你偷偷改评分代码
  2. 用 development 重新拟合参考（参考本身不碰 validation）
  3. 在 validation 上算 3 个候选的 loss
  4. 算 200×128 嵌套 bootstrap，给 loss 比一个 95% 区间
  5. 如果参考 gate 没过 → 所有 `score = null`；过了 → `score = 100 × (基线 loss + 1e-12) / (你的 loss + 1e-12)`
- **输出**：`validation_results.json`、`nested_bootstrap.npz`

### 第 5 步：`plot`

只读结果、画图，不重算分数。4 张图：偏离、SNR、有效质量、参考窗口稳定性。

### 第 6 步（可选）：`candidate`——交自己的卷子

```bash
python -m tasks.pion_projection --output run candidate \
  --definition my_candidate.json --split development
python -m tasks.pion_projection --output run candidate \
  --definition my_candidate.json --split validation
```

`my_candidate.json` 长这样：

```json
{
  "builder": "tasks.pion_projection.kernels:build_kernel",
  "parameters": {"powers": [0, 8, 32], "weights": [-0.2, 0.5, 0.7], "alpha": 0.2}
}
```

`candidate` 会：导入这个模块 → 把**源码文件**的哈希算进身份 → 在每条组态、每个时间片上调用 `build_kernel(links, parameters)` → 用 `apply(field)` 算关联函数 → development 上冻结、validation 上打分。同一个定义重复提交会命中同一个哈希目录（防止改答案）。

`--verify-only` 是给扩散核准备的：只跑第一条组态，和你之前测的基矢线性组合对比，确认"你交的算子"和"你报的组合"是同一个东西。

---

## 5. 分数到底怎么算

### 5.1 先修一条"理论曲线"

π 介子的两点函数，在大时间 t 下应该长这样：

```
f0(t) = exp(-m0·t) + exp(-m0·(16-t))
```

`m0` 是出题人从 development 数据里拟合出来的**共用基态质量**（上一次 kunshan 跑出来是 1.086）。考生不能自己拟合 `m0`——所有人用同一条参考线。

### 5.2 然后看两件事

对每个时间点 t（评分窗是 t=2,3,4,5）：

```
q(t) = 实测关联函数 / (幅度 A × f0(t))
```

幅度 `A` 是出题人在 t=6,7,8 上拟合出来的归一幅值（考生也不能改）。

- 如果 `q(t) = 1`，说明和基态形状完美吻合
- 如果 `q(t)` 乱跳，说明有激发态污染或统计噪声

于是有两个罚分：

1. **方差项** `Var_boot(q)`：你的信号抖不抖（bootstrap 估出来的不确定度）
2. **偏离项** `(q-1)²`：你离基态形状有多远

两个加起来取平均，就是 **loss**：

```
loss = mean_t [ 方差项 + 偏离项 ]
```

loss 越小越好。注意：**整体乘一个常数不改变分数**（幅度会跟着变，q 不变），所以别想着"把信号放大"来作弊。

### 5.3 换算成 100 分制

```
score = 100 × (基线 loss + 1e-12) / (你的 loss + 1e-12)
```

基线是**恒等 sink**（最朴素的局域 sink），它永远等于 100 分。你的 loss 是它的一半 → 200 分。

### 5.4 什么时候**不**给分（gate）

只要下面任何一条不满足，`score` 就是 `null`，并在结果里写明原因：

- 参考质量的相对误差 > 5%
- 参考拟合的近似 p 值 < 0.01
- 换个拟合窗口，质量漂移 > 2.5 个标准误
- 拟合质量撞到了边界（0.1 或 3.0）
- 前后向反射对称性 z > 4
- 你的幅度统计上不可辨认（|A|/SE < 3）
- 出现 NaN、零信号、负关联函数（在拟合窗内）
- 反演残差不合格

### 5.5 为什么上次是 `null`

因为**参考本身没通过**。上一次在 kunshan 跑的时候，换个窗口，m0 就从 1.112 变到 1.086 再到 1.072，漂了 21.6σ 和 7.8σ。参考都不稳，分数就没有意义，代码按设计把 `score` 置空。

这不是 bug，是"宁可没分，也不给假分"。

---

## 6. 在本机跑（能跑什么、不能跑什么）

### 6.1 需要什么

| 模式 | 需要 | 能干什么 |
| --- | --- | --- |
| 只读代码 | Python + numpy + scipy + matplotlib | 看代码、跑 7 个单测、`--help` |
| 有数据没 GPU | 上面这些 + 一份 `development.npz`/`validation.npz` | `develop` / `evaluate` / `plot` |
| 有 GPU 有数据 | 上面这些 + CUDA/DCU + QUDA + PyQUDA + MPI + NERSC 组态 | 全套 |

### 6.2 最小验证（不需要 GPU）

```bash
cd ~/LatticeArena
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q     # 7 passed，约 1 秒
python3 -m tasks.pion_projection --help            # 看 5 个子命令
```

两个坑：

- 必须用 `python -m pytest`。直接敲 `pytest` 会报 `No module named 'tasks'`（没配 pythonpath）。
- 必须在仓库根目录跑，`.cache/`、`results/` 都是相对当前目录的。

### 6.3 只有关联函数、没有 GPU 时

`develop`/`evaluate`/`plot` 是纯 NumPy/SciPy/Matplotlib。只要你有形状对、编号对、带正确 `protocol_hash` 的 npz，就能跑完整打分。这也是"笔记本上做分析、集群上只做测量"的推荐姿势。

---

## 7. 在 kunshan 远端跑，并拿到 LatticeArena 的分

这一节是上一次实际跑通的步骤，换别的集群也一样，只要改路径。

### 7.1 登录、环境

```bash
ssh kunshan
source ~/env_pyquda_conda.sh     # module load + conda activate pyquda_env
```

- Slurm 队列：`kshdnormal`（每节点 4 张 DCU，`gfx906`）
- 软件：QUDA 1.1.0（gfx906 版）、PyQUDA 0.10.53、cupy 12.3.0、Python 3.8
- 远端**没有 pytest**；要跑单测可以 `pip install --target pilot/pylibs pytest`（装在自己目录里，别污染 conda）

### 7.2 准备组态（两条路）

**路 A：拷贝现成的组态。** 只要有 `wilson_b6.0`…`wilson_b6.N` 的 NERSC 文件，放进一个目录，改 `protocol.json` 的 `ensemble` 和 split 即可。

**路 B：自己用 PyQUDA 的 HMC 产。** 上次就是这么干的（下面只是示意，完整可运行版本见远端 `pilot/hmc_generate.py`）：

```python
from pyquda.action import GaugeAction, abstract
from pyquda.hmc import HMC, Integrator
from pyquda_utils import core
from pyquda_utils.core import X, Y, Z, T

# 1) 随机 SU(3) 热启动（千万别用单位场冷启动，动量一踢就飞）
a = rng.normal(size=gauge.data.shape) + 1j * rng.normal(size=gauge.data.shape)
q, _ = np.linalg.qr(a)
q = q / np.linalg.det(q)[..., None, None] ** (1/3)
gauge.data[:] = cupy.asarray(q)

# 2) Wilson 纯规范作用量，β=6.0
loop = abstract.LoopParam([[X,Y,-X,-Y],[X,Z,-X,-Z],[Y,Z,-Y,-Z],
                           [X,T,-X,-T],[Y,T,-Y,-T],[Z,T,-Z,-T]], [1.0]*6)
hmc = HMC(info, [GaugeAction(info, loop, 6.0)], BAB(n_steps))

# 3) 每条轨迹：gaussMom → integrate(τ) → accept → save/load
# 4) 热平衡 400 条，之后每 5 条存一个：io.writeNERSCGauge(f"wilson_b6.{i}", gauge)
```

上次用的参数：`n_steps=120, τ=0.5, thermal=400, trajectories=900, save-every=5`，得到 **100 个组态**，接受率 97.2%，plaquette 0.59376。

**两个必踩的坑**：

- **冷启动会死**：单位场是作用量极小点，动量却是 O(1)，第一条轨迹 dE≈1.8e4，永远拒绝。必须热启动。
- **步长太大会死**：`τ=1.0, n_steps=10`（dt=0.1）时接受率只有 2%，链冻住。扫描发现 dE ∝ dt²，所以老老实实 120 步、τ=0.5。

### 7.3 让本仓库认这台机器（只改 3 个文件）

| 文件 | 改什么 | 为什么 |
| --- | --- | --- |
| `protocol.json` | `ensemble` 改成本机数据路径；split 改成 60/10/30 | 原路径是 `/data/jinchen/...`，且只有 50 个组态 |
| `measure.py` | `libquda.so` 改成从 `QUDA_PATH` 拼；版本查询容忍缺包 | 原代码硬编码 `/home/jinchen/...`，且 `cupy-cuda12x` 在这个 conda 环境里叫 `cupy` |
| `plots.py` | `layout='constrained'` 加 `try/except` 回退；标题从结果里读实际组态数 | 远端 matplotlib 3.3 不认 `layout=`；原标题硬编码 "30+15" |

`scripts/run_pion.sh` **不要用**，它第一行就 `source /home/jinchen/...`。自己写一个 sbatch 模板：

```bash
#!/bin/bash
#SBATCH -J la_pipe
#SBATCH -p kshdnormal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=dcu:1
#SBATCH -t 04:00:00
#SBATCH -o .../logs/pipe_%j.out
#SBATCH -e .../logs/pipe_%j.err
set -euo pipefail
source /public/home/<user>/env_pyquda_conda.sh
cd /public/home/<user>/dianjun/LatticeArena
export MPLBACKEND=Agg OMP_NUM_THREADS=1
python -m tasks.pion_projection "$@"
```

**必须是 1 个 rank、1 张卡**：`measure.py` 里写死了 `if MPI.COMM_WORLD.Get_size() != 1: raise`。

### 7.4 五条命令

```bash
cd ~/dianjun/LatticeArena
OUT=results/pion_projection/s16t16_v1

sbatch pilot/pipeline.sbatch --output $OUT measure --split development
sbatch pilot/pipeline.sbatch --output $OUT develop
sbatch pilot/pipeline.sbatch --output $OUT measure --split validation
sbatch pilot/pipeline.sbatch --output $OUT evaluate
sbatch pilot/pipeline.sbatch --output $OUT plot
```

### 7.5 怎么知道成功、怎么读分

看 `$OUT/validation_results.json`：

```bash
python3 -c "
import json; r=json.load(open('$OUT/validation_results.json'))
print(r['score_status'], r['reference_passed'])
for k,v in r['results'].items(): print(k, v['risk'], v['score'])
"
```

- `score_status = "scored"` → 有分
- `score_status = "unscored_reference_failed"` → 参考没过，`score` 全 `null`，但 `risk` 和区间仍然是有用的诊断
- 每个候选都有 `risk`、`variance`、`departure_squared`、`amplitude`、`risk_95_interval`

### 7.6 上次的真实数字（供对照）

| 项目 | 值 |
| --- | --- |
| 组态 | 100 个 16⁴，β=6.0，plaquette 0.59376 ± 0.00050 |
| 测量 | 90 个组态 × 12 列 = 1080 次反演，最大残差 9.99e-11 |
| 耗时 | HMC 7:26；measure dev 3:57；develop 0:56；measure val 2:23；evaluate 1:16；plot 0:36 |
| 参考 | m0=1.08617(31)，p=1.7e-8，窗口漂移 21.6σ / 7.8σ → 失败 |
| 分数 | 全 `null` |
| 诊断 | local 0.3127；tuned H³² 0.01256；agent 0.00770；agent vs tuned = 1.63 [1.40, 2.19] |

### 7.7 远端常见失败

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| `libimf.so: cannot open` | 没 source 环境脚本 | `source ~/env_pyquda_conda.sh` |
| `ModuleNotFoundError: cupy` | 用的系统 python | 用 conda 的 `python`（脚本里已 source） |
| `No such file: /home/jinchen/...` | 没打补丁 | 见 7.3 |
| `unscored_reference_failed` | 物理/协议问题，不是代码问题 | 换更长 T、更多组态，或改参考模型 |
| 任务一直 PENDING | 队列满 | `squeue -u $USER`，或换 `kshdnormal02/04` |

---

## 8. LLM 是怎么被调用的（重点）

### 8.1 先说结论：LLM 是**可选**的一层

默认流水线（`measure`/`develop`/`evaluate`/`candidate`）**不调用**任何大模型，分数完全由本地统计
决定。仓库里另有两条可选命令会调用 DeepSeek：

- **`agent`**：`agents/loop.py` + `agents/llm.py`，让 LLM 每轮提 3 组权重、本地打分、把结果喂回下一轮；
- **`report`**：`agents/report.py`，LLM 只写 `REPORT.md` 的文字解读。

API key 读 `~/Desktop/api.key` 或 `$LLM_API_KEY`，默认 `deepseek-flash`，POST 到
`https://api.deepseek.com/v1`（`response_format: json_object`、`thinking` 关闭）。
**计算节点没有 DNS**，所以这两个命令要在登录节点或本地跑；GPU 只用于测量/评估。

### 8.2 代码里的 "agent" 是什么

两个地方出现了 "agent"：

1. `README.md` 说"这个项目是给 AI agent 做研究用的"——这是**目标**；
2. `validation_results.json` 里的 `agent_signed_mixture` 是 `experiment.py` 里 **SLSQP** 算出的权重，
   不是 LLM 生成的（它是本地强基线）。

LLM 的入口是独立的 `agent` 子命令（见 8.3），它只写 `<output>/agent/`，不碰冻结的提交。

### 8.3 现在已经实现的入口（`agent` / `report`）

仓库里已经加好了，两条命令：

```bash
python -m tasks.pion_projection --output results/pion_projection/s16t16_v1 agent  --rounds 4
python -m tasks.pion_projection --output results/pion_projection/s16t16_v1 report
```

- `agents/llm.py`：读 `~/Desktop/api.key`（或环境变量 `LLM_API_KEY`），POST 到 `https://api.deepseek.com/v1`，默认 `deepseek-flash`，可用 `--model` 换。
- `agents/loop.py`：从 `<output>/development_provenance.json` 取协议、从 `<output>/development.npz` 取关联函数，拟一次参考，然后每轮让 LLM 给 7 个归一化系数 u（内部按 `u/A_n` 转成核权重）、算 risk、刷历史；最后用**开发集冻结的度量**算一次留出诊断 risk，写 `<output>/agent/best.json`。它只是在 CPU 上做线性组合，所以不需要 GPU；对扩散核族这和 `candidate` 的结果数学上等价。
- `agents/report.py`：读 `validation_results.json`，按 risk 排前三，代码生成表格，LLM 只写解读，写 `<output>/REPORT.md`。

**边界**：这条链只写 `<output>/agent/` 和 `REPORT.md`，不碰 `submission.json`，所以不会污染冻结的提交；`best.json` 里明确写了 `post-hoc diagnostic, not an official submission`。

### 8.4 LLM 应该怎么接进来？三条路（设计参考）

#### 路线 1：离线（今天就能做，最简单）

你在 ChatGPT / Claude / DeepSeek 里贴一段 prompt，让它写一个 `build_kernel`，你把代码存成 `.py`，跑 `candidate`。

```
人 → LLM → 代码 → 人工检查 → candidate → 分数
```

优点：零开发。缺点：慢，靠人搬。

#### 路线 2：半自动循环（推荐，需要自己写 ~80 行）

循环（已经实现为 `agents/loop.py`，下面只是它的逻辑）：

```
读题目（protocol + README + 上一次的分数）
        ↓
调 LLM API（要求它输出 JSON 或 Python 代码）
        ↓
本地静态检查 + 7 个单测（协变/线性/形状）
        ↓
candidate --split development  →  拿到 risk
        ↓
把 risk 和诊断喂回 LLM，让它改
        ↓
重复 N 轮（只在 development 上！）
        ↓
冻结 → candidate --split validation（只跑一次）
```

#### 路线 3：全自动（现在不建议）

自动改代码、自动提交、多轮自我对弈。需要沙箱、配额、防作弊，仓库里都没有。**别在共享集群上直接跑 LLM 生成的代码**——见 8.7。

### 8.5 交卷接口长什么样（LLM 只需要产出这个）

**方式 A：给参数**（适合让 LLM 调权重）

```json
{
  "builder": "tasks.pion_projection.kernels:build_kernel",
  "parameters": {"powers": [0, 8, 32], "weights": [-0.2, 0.5, 0.7], "alpha": 0.2}
}
```

**方式 B：给代码**（适合让 LLM 设计新形状）

```python
# my_kernel.py
def build_kernel(spatial_links, parameters):
    return MyKernel(spatial_links, **parameters)

class MyKernel:
    def __init__(self, links, ...): ...
    def apply(self, field):      # field: (z, y, x, 3, nrhs) -> 同形状
        ...
```

规则（LLM 必须遵守，人来检查）：

- 只用当前时间片的**空间** link，形状 `(3, z, y, x, 3, 3)`
- 对输入**线性**；不能看传播子；不能按组态归一化
- **规范协变**、**平移协变**、**时间局域**、立方对称 `A1+`
- 参数冻结：所有组态、所有时间片用同一套参数
- 不能碰 validation，不能改评分代码

### 8.6 一个 LLM 循环的伪代码（示意；仓库里的实现见 `agents/loop.py`）

```python
# 逻辑示意；实际实现在 agents/loop.py（无需 openai，用 urllib）
import json, os, re, subprocess, pathlib
from openai import OpenAI

# 前置：先跑过一次 develop，保证 results/.../submission.json 存在
OUT = "results/pion_projection/llm_run"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

prompt = pathlib.Path("doc/pion_projection.md").read_text() + """
你只能修改 K = sum_n w_n H^n 的 weights。
输出 JSON: {"powers": [...], "weights": [...]}
要求 sum(weights)=1, sum(|weights|)<=8。
"""

history = []
for turn in range(5):
    msg = prompt + "\n历史:\n" + json.dumps(history)
    reply = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": msg}]
    ).choices[0].message.content
    # 模型常把 JSON 包在 ```json 里，先剥掉；解析失败就重问
    text = re.sub(r"^```(?:json)?|```$", "", reply.strip(), flags=re.M).strip()
    try:
        cfg = json.loads(text)
    except json.JSONDecodeError:
        history.append({"error": "bad json", "raw": reply[:500]})
        continue
    cfg = {"alpha": 0.2, **cfg}          # build_kernel 需要 powers/weights/alpha 三个键
    if abs(sum(cfg["weights"]) - 1) > 1e-6:
        history.append({"error": "weights 不满足 sum=1"})
        continue

    pathlib.Path("my_candidate.json").write_text(json.dumps({
        "builder": "tasks.pion_projection.kernels:build_kernel",
        "parameters": cfg,
    }))

    run = subprocess.run(["python", "-m", "tasks.pion_projection", "--output", OUT,
                          "candidate", "--definition", "my_candidate.json",
                          "--split", "development"], capture_output=True, text=True)
    if run.returncode != 0:              # 幅度不可辨认 / 非法核都会在这里失败
        history.append({"cfg": cfg, "error": run.stderr[-500:]})
        continue

    # candidate 会打印 "Candidate artifacts: <目录>"，从里面读 risk
    dest = re.search(r"Candidate artifacts: (\S+)", run.stdout).group(1)
    risk = json.loads((pathlib.Path(dest) / "frozen.json").read_text())[
        "development_diagnostic"]["risk"]
    history.append({"cfg": cfg, "risk": risk})

# 只把最好的那次拿去 validation，跑一次，不许回头
```

关键点：

- **每轮只能看 development**。validation 是考卷，看了就不能再改。
- 每轮都有成本：一次 `candidate` 要把 development 的所有组态跑一遍（默认 30 条；传播子有缓存，只重算 sink）。
- LLM 的输出要**先过单元测试**再跑，别直接执行。
- 记录每一轮的 prompt / 输出 / 分数，否则没法复现。

### 8.7 给 LLM 的 prompt 模板（可以直接抄）

```
你在参加一个格点 QCD 的优化任务：设计 π 介子的 sink 核。
固定条件：16^3×16 淬火 Wilson 系综，β=6，κ=0.14，原点 point source，
反周期时间边界，12 个 spin-color 列，残差 1e-10。

你只能改：
K = Σ_n w_n H^n，H = (I + 0.2·hop)/2.2，n ∈ {0,2,4,8,16,32,64}
约束：Σw_n = 1，Σ|w_n| ≤ 8。

目标：最小化
loss = mean_{t=2..5} [ Var_boot(q(t)) + (q(t)-1)^2 ]
q(t) = C(t) / (A·f0(t))，f0(t)=exp(-m0 t)+exp(-m0(16-t))，
m0 由出题人给定，你不能改。

输出严格 JSON（不要加解释、不要加 markdown 代码块）：
{"powers": [0, 2, 4, 8, 16, 32, 64], "weights": [...], "alpha": 0.2}
```

### 8.8 红线（千万别做）

1. **不要让 LLM 生成的代码直接跑在共享机器上**。`candidate` 是"信任本地代码"，不是沙箱——你的代码可以读文件、可以联网、可以 `rm`。先自己看一遍。
2. **不要让 LLM 看 validation 分数来调参**。那就等于偷看考卷，分数没有意义。仓库虽然用哈希冻结了"提交后不能改"，但挡不住你在冻结前反复看。
3. **不要相信 LLM 说的"我优化了 10 倍"**。只有 `validation_results.json` 里的 `risk` 和区间算数，而且参考失败时连分都没有。
4. **不要改 `statistics.py` 或 `kernels.py` 去骗分**。`evaluate` 会校验这两个文件的哈希，改了直接报错。

---

## 9. 怎么加一个新任务

现在加任务要复制一整个目录（没有注册表）。步骤：

1. 复制 `tasks/pion_projection/` 成 `tasks/<新名字>/`
2. 改 `protocol.json`：系综、几何、split、测量参数、窗口、gate
3. 改 `measure.py`：你的可观测量怎么算（这是唯一必须重写的部分）
4. 改 `statistics.py` 里的 `shape()`（如果不是 cosh）和 `experiment.py` 里的基线
5. 改 `__main__.py` 的子命令（或者保持一样）
6. 在 `tests/` 加至少：协变/线性/形状/合成数据还原/失败路径
7. 在 `doc/` 写任务定义，在任务目录写 `README.md`
8. **先跑通 development，再想 validation**

更好的做法（也是之前建议过的）：把 `common/statistics/experiment/candidate` 提到 `src/latticearena/`，任务只留物理。现在它们都在任务目录里，第二个任务会复制约 600 行。

---

## 10. 名词表

| 词 | 大白话 |
| --- | --- |
| 组态（configuration / gauge field） | 一份"宇宙快照"，即四维格点上每个 link 的 SU(3) 矩阵 |
| NERSC 格式 | 一种存规范场的二进制格式，PyQUDA 能直接读 |
| 传播子（propagator） | 夸克从 A 点走到 B 点的"概率幅"，由 Dirac 方程反演得到 |
| point source | 源只放在一个点（这里是原点），最常用的源 |
| sink | 湮灭端。这个任务优化的就是 sink 的空间形状 |
| κ（kappa） | 夸克质量参数，κ 越大越轻 |
| 淬火（quenched） | 没有海夸克，只有价夸克。便宜，但不是真实 QCD |
| plaquette | 最小的 1×1 方块，用来衡量规范场"有多热" |
| HMC | 混合蒙特卡洛，产生规范组态的标准算法 |
| 热平衡（thermalization） | 丢掉最开始那段还没平衡的轨迹 |
| 自相关 | 相邻组态太像，统计上不算独立；所以要分块 |
| bootstrap | 重采样估误差的方法，不依赖正态假设 |
| loss / risk | 越小越好的惩罚分数 |
| gate | 及格线；不过就不给分 |
| fold | 正反时间平均（因为时间方向是周期的） |
| effective mass | 从关联函数反推的"局部质量"，画图看平台用 |
| SLSQP | 一个带约束的数值优化器；现在冒充"agent" |

---

## 11. 排错清单

| 症状 | 最可能的原因 | 怎么办 |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'tasks'` | 不在仓库根目录，或直接用了 `pytest` | `cd` 到根目录，用 `python -m pytest` |
| `--output` 报错 | 它是全局参数，必须放子命令前面 | `--output X measure ...` |
| `Development is closed for this run` | 已经有 `submission.json` 或 `validation.npz` | 换一个新的 `--output` 目录 |
| `missing or reordered configurations` | 用了 `--limit`，或数据不完整 | 别用 `--limit` 做正式跑；补全数据 |
| `Protocol changed since measurement` | 改了 `protocol.json` | 重新测量，别复用旧 npz |
| `No such file: .../libquda.so` | 没打补丁 / 没设 `QUDA_PATH` | 见 7.3 |
| `score` 全是 `null` | 参考 gate 没过 | 看 `reference_failure_reasons`；换 T 或组态数 |
| 图里标题写 "30+15" | 用了旧版 `plots.py` | 按 7.3 打标题补丁 |
| 绘图报 `layout` 错 | matplotlib < 3.6 | 用 `constrained_layout=True` 回退 |
| 远端任务 PENDING 很久 | 队列满 | `squeue`，换分区，或等 |

---

## 12. 一页速查

```bash
# ---------- 本机（无 GPU）----------
cd ~/LatticeArena
python3 -m pytest -q                                   # 7 个测试
python3 -m tasks.pion_projection --help                # 看命令

# ---------- 有数据、无 GPU ----------
python3 -m tasks.pion_projection --output run develop
python3 -m tasks.pion_projection --output run evaluate
python3 -m tasks.pion_projection --output run plot

# ---------- 有 GPU、有组态 ----------
python3 -m tasks.pion_projection --output run measure --split development
python3 -m tasks.pion_projection --output run develop
python3 -m tasks.pion_projection --output run measure --split validation
python3 -m tasks.pion_projection --output run evaluate
python3 -m tasks.pion_projection --output run plot

# ---------- 让 LLM 答题 / 解读 ----------
python3 -m tasks.pion_projection --output results/pion_projection/s16t16_v1 agent --rounds 4
python3 -m tasks.pion_projection --output results/pion_projection/s16t16_v1 report

# ---------- 交自己的卷子 ----------
python3 -m tasks.pion_projection --output run candidate \
    --definition my_candidate.json --split development
python3 -m tasks.pion_projection --output run candidate \
    --definition my_candidate.json --split validation

# ---------- 远端（kunshan）----------
ssh kunshan
source ~/env_pyquda_conda.sh
cd ~/dianjun/LatticeArena
sbatch pilot/pipeline.sbatch --output results/pion_projection/s16t16_v1 measure --split development
# ……然后 develop / measure val / evaluate / plot
```

**三个最重要的文件**：

1. `tasks/pion_projection/protocol.json` —— 所有数字都在这
2. `doc/pion_projection.md` —— 所有规则都在这
3. `tasks/pion_projection/experiment.py` —— 参考怎么拟合、分怎么算

**一句话记住**：**LatticeArena 负责考场和阅卷，不负责考生；现在的考生是 SLSQP，LLM 还在场外，要接进来得自己写驱动。**
