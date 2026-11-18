# LLM/搜索入口隔离（F2）

## 1. 统一候选校验

`core/candidates.validate_weights(u, size, l1_bound)` 是 optimizer 与 LLM 共用的唯一入口：

- 一维、长度等于幂次数；
- 全部有限（拒绝 NaN/Inf）；
- `sum(u) = 1`（容差 1e-6）；
- `sum|u| <= 8`（`optimizer_l1_bound`）。

`agents/loop.py` 对每个候选调用它；不合法时写入 `agent/rejections.json`（含 round、向量、原因），
不再静默 `continue`。

## 2. 搜索期间不读 validation

`run(..., allow_validation=False)` 在搜索结束后直接写 `best.json`（`validation_status='not_read'`）并返回，
不打开 `validation.npz`。测试用拦截 `np.load` 的方式证明：任何对 `validation` 路径的读取都会抛
`AssertionError`，而搜索仍完成。

## 3. 评估入口复用身份检查

`allow_validation=True` 时，读取 validation 之前会检查：

- `validation.npz` 的 `protocol_hash` 等于冻结协议；
- `validation_provenance.json` 的 `protocol` 等于运行协议。

任一不符即拒绝，不再直接 `np.load` 绕过。

## 4. mock 轨迹

`agents/llm.chat(model='mock')` 从 `agents.llm.MOCK_QUEUE` 取预设回复，用于离线回归：
`results/pion_projection/agent_mock_v1/` 记录了四类非法候选的拒绝原因与一个合法候选的接受结果。

## 5. SearchAdapter 与预算（F2 集成）

`agents/loop.py` 现在真正调用 `core.search.search`：每轮候选走 validate → evaluate → 记录；
NaN/Inf 评估结果被拒绝（`non-finite evaluation result`），预算用尽即停。
每轮写 `agent/search_trace.json`（模型、seed、history、rejections、best、validation_status），
不再只是 import 模块却手写循环。

## 6. 仍未做

- 真实 LLM 调用的预算账本（token/费用/超时）与沙箱；
- 统一 `SearchAdapter`（F1 的第 5 类接口）；
- mock 的“错误维数/零和/超 L1”已覆盖，但“合法类型”（非数值、嵌套列表）只做了部分。
