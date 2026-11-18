# optimize_gaussian P3：最小有判别力实验（预注册 + 停止规则）

本文件冻结 P3 第二步（GPU）的设计；**执行 NOT_RUN**。CPU 第一步的 teacher 结果见
[`results/optimize_gaussian/p3_teacher_v1/`](../results/optimize_gaussian/p3_teacher_v1/README.md)。

## 1. 假设

H1：在七幂扩散类之外加入一个 gauge-covariant、O_h 投影、非对易的 $Q$ 修正
（$K=P_u(H)+\sum_j\theta_j B_jQ_jB_j$）能在**新数据**上降低预注册的主终点。
H2：收益来自类本身，而不是优化器（SLSQP vs Powell）或训练集过拟合。

## 2. 主终点与比较族

- 主终点：新留出数据上的配对 risk 比（强谱类 baseline / 新类候选），block bootstrap 95% 区间；
- 比较族：2 类（七幂、嵌套 $Q$）× 2 优化器（SLSQP、Powell）= 4 个候选，冻结后再看 validation；
- 成功：区间下界 $>1$ 且超过预注册实际意义阈值（例如 1.2）；
- 反证：真实与 shuffled 无差别、强谱基解释全部收益、或新类只降低训练误差；
- 证据不足：区间跨 1 → INCONCLUSIVE，不能称等效。

## 3. 数据与样本量

- 新 gauge/propagator 必须独立于已反复观察的 v1 70–99；旧数据只辅助开发；
- 同样本同时测强谱类、uniform、real、shuffled 四臂；
- 样本数由块方差与目标效应的 power simulation 决定（不是固定 30/60）；预注册模拟脚本与种子；
- 每臂记录 inversion、setup、apply、训练、measurement、total time 与硬件/精度/残差。

## 4. 成本账本（预注册字段）

`setup / solve / kernel apply / 训练 / 测量 / IO / 分析 / total`，区分冷/热缓存、精度、设备、并行度；
不使用未定义的“risk×秒”。

## 5. 停止规则

完成预定预算/轮次后：

1. 无新数据收益（配对区间跨 1 或下界低于阈值）；
2. 有效秩不增加；
3. 成本超过预注册容忍界；
4. 资源不可用（无 GPU 授权）。

任一成立即归档负结果并结束本轮；**不无限加模型/调 loss 追求正结果**。

## 6. 当前状态

| 项 | 状态 |
| --- | --- |
| CPU teacher 跨样本实验 | ✅ PASS（`p3_teacher_v1/`） |
| class×optimizer 2×2（GPU） | ⛔ NOT_RUN（需要新 GPU 生产与授权） |
| real/uniform/shuffled 消融（GPU） | ⛔ NOT_RUN |
| 新盲测与 power simulation | ⛔ NOT_RUN |
