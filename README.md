# LatticeArena

LatticeArena 把格点规范理论里的研究问题，改造成**有固定测量流程、固定打分规则**的任务：AI（或人）只优化一个很窄的对象，其余全部由评测器控制。目标不是刷榜，而是看"能不能提出一个真正更好的方案"，并且诚实地报告"没做到"。

当前有两个任务：**`pion_projection`**（π 介子 sink 的设计）和 **`wilson_loop`**（静态夸克势的空间 Wilson 线核）。仓库里包含可执行的 pilot：测量、缓存、参考拟合、基线选择、冻结提交、留出打分、LLM 作答入口。

> **状态（2026-11-18）**：`wilson_loop` 的 v3 参考通过、分数有效（APE 263/233/222，配对区间不含 1）；
> `pion_projection` 的参考仍不可识别，`score` 按设计为 `null`；`optimize_gaussian` 的学习核与 SLSQP 打平。
> 总览见 [doc/final_report.md](doc/final_report.md)，todo 覆盖情况见 [doc/todo_coverage.md](doc/todo_coverage.md)。

---

## 快速开始（CPU 就能跑）

```bash
python -m pytest -q                       # 51 个不需要 GPU 的单元测试
python -m tasks.pion_projection --help    # measure / develop / evaluate / plot / candidate / agent / report
```

有 `development.npz` 但没有 GPU 时，分析和打分都能跑：

```bash
python -m tasks.pion_projection --output results/pion_projection/s16t16_v1 report
python -m tasks.pion_projection --output results/pion_projection/s16t16_v1 agent --rounds 3
python -m tasks.wilson_loop --help          # 第二个任务：静态势
```

`agent` / `report` 需要一个大模型 API key，默认读 `~/Desktop/api.key`（也可以用环境变量 `LLM_API_KEY`），默认走 `https://api.deepseek.com/v1` 的 `deepseek-flash`。

---

## 这个任务是什么

在淬火 Wilson 系综（`16^3 x 16`，β=6.0，κ=0.14）上，固定原点 point source，让 agent 设计 sink 的空间核 `K`：

```
K = Σ_n u_n H^n / A_n,    H = (I + 0.2 · spatial_hopping) / 2.2
```

- 只能改 `u`（7 个归一化系数，`Σu=1`、`Σ|u|≤8`，允许带符号）；
- `H^0 = I` 是局域 sink，也就是 100 分基线；
- 评测器负责反演、收缩、折叠、参考拟合、bootstrap 和打分。

分数：

```
q(t) = C(t) / (A · f0(t)),   f0(t) = exp(-m0 t) + exp(-m0 (16-t))
risk = mean_{t∈{2,3,4,5}} [ Var_boot(q(t)) + (q(t)-1)² ]
score = 100 · (risk_baseline + 1e-12) / (risk_candidate + 1e-12)
```

参考不稳（窗口漂移、拟合 p 值、反射对称等）时，`score` 一律为 `null`。

---

## 仓库结构

| 路径 | 作用 |
| --- | --- |
| `tasks/pion_projection/` | 任务一（π 介子 sink）：`measure.py`（可信测量）、`statistics.py`（参考/风险）、`experiment.py`（develop/evaluate）、`candidate.py`（交卷）、`kernels.py`（参考核）、`protocol.json`（唯一数值真源） |
| `tasks/wilson_loop/` | 任务二（静态势）：Wilson 圈测量、APE 核、平台稳定性打分 |
| `tasks/optimize_gaussian/` | 学习型 smear 的骨架（smoke）：规范不变逐点系数、厄米条件、线性响应训练；结果仅用于验证流程 |
| `core/io.py` | 通用哈希与 JSON 工具，供所有任务复用 |
| `agents/` | LLM 作答入口：`loop.py`（`agent` 子命令）、`report.py`（`report` 子命令）、`llm.py`（API 客户端） |
| `tests/` | 14 个 GPU-free 测试（两个任务：协变、线性、折叠、Wilson 圈暴力对照、bootstrap、gate） |
| `doc/` | 任务科学定义、环境搭建、第一次实验结果、kunshan DCU 试跑报告 |
| `results/` | 真实运行产物（默认被 git 忽略，只提交小件），见 [results/README.md](results/README.md) |
| `Note.md` | 一份很详细的中文使用说明：架构、每个文件、远端部署、LLM 接入 |

---

## 跑完整流程

```bash
python -m tasks.pion_projection --output run measure --split development   # 需要 GPU + QUDA
python -m tasks.pion_projection --output run develop
python -m tasks.pion_projection --output run measure --split validation
python -m tasks.pion_projection --output run evaluate
python -m tasks.pion_projection --output run plot
```

顺序不能反：`develop` 会冻结 `submission.json`，之后不许重开。

在 DCU 集群（Slurm）上的完整步骤、移植补丁和踩坑记录见 [doc/kunshan_dcu_run.md](doc/kunshan_dcu_run.md)。

---

## LLM 是怎么参与的

仓库里的 "agent" 有两层含义，别混淆：

1. **`agent` 子命令**：真的调用大模型。它把题目、7 个通道的幅度和风险、历史提案给模型，模型输出 3 个候选 `u`，脚本逐个打分、记录、迭代；最后在开发集上冻结的度量下算一次留出诊断。产物在 `<run>/agent/`，**不会**碰冻结的 `submission.json`。
2. **`agent_signed_mixture`**：这是 `develop` 里 SLSQP 优化器算出来的候选，不是大模型。

`report` 负责"前三名 + LLM 解读"：表格由代码从 `validation_results.json` 生成，大模型只写叙述，产物是 `<run>/REPORT.md`。

---

## 诚实的边界

- 这是 **pilot**：参考态没有通过门控，所有 `score` 为 `null`；`agent_signed_mixture` 相对 Gaussian 的 1.63 倍（区间 [1.40, 2.19]）只是冻结度量下的诊断，不是 SOTA。
- 16⁴、淬火、κ=0.14 的 π 是"重 π"，不能和物理质量比较。
- 提交的核是**受信任的本地代码**，不是沙箱；不要直接跑不可信来源的模块。
- 学习核（`optimize_gaussian`）修复特征后：比 Gaussian 基座好 2.12 倍，候选与 7 参数 SLSQP 打平（0.998 [0.78, 1.18]）；细节见 [tasks/optimize_gaussian/README.md](tasks/optimize_gaussian/README.md)。
- 第二个任务（`wilson_loop`）的 pilot 结论是负面的：开发集上 APE 比不平滑好 4.5 倍，但留出集只有 6 个块，差异消失（score 95.7–102.7）。见 [tasks/wilson_loop/README.md](tasks/wilson_loop/README.md)。
- 仓库里的 `tasks/pion_projection/protocol.json` 是 30/5/15 的原始协议；已提交的这次运行用的是 60/10/30（协议快照在 `results/pion_projection/s16t16_v1/protocol.json`），所以 `develop`/`evaluate` 不能直接在这份数据上重跑。

---

## 历史

旧的多任务实现（`core/task.py` 注册表、`wilson_loop`、`pion_2pt`、`gsfit_2pt`、CI、排行榜页面）保留在提交 `f62c613` 中；重置提交是 `5e0ceb9`，当前 pilot 从 `3a88b10` 开始。访问方式：

```bash
git show f62c613:core/task.py
```
