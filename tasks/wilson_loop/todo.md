# Wilson loop 与公共框架：有限范围验收开工单

基线：`21a96cf`，2026-11-18。**保留：未完成；先修CPU集成断路。**
主agent实跑80 tests全过（5.03s）；这不能替代真实入口与科研验收。
这批待办以CPU和现有档案为验收范围；GPU/远程/API生产另列后续实验。新产物写新目录，不覆盖旧results。
下列新增输出路径均为**拟议**。负结果、受控拒绝、明确延期也能验收；不要求正收益。

## 1. 已完成，关闭旧待办

| 完成项 | 证据 | 限定范围 |
|---|---|---|
| 合法空间APE | polar+det修正、α/4、双侧协变测试 | 不再保留Gram–Schmidt bug为待修 |
| 混尺度投影 | 逐矩阵scale、80步上限及收敛检查；`audit_20261118_projection/` | 旧batch scale问题已修；GPU对照未跑 |
| 基本数组/格式 | `load_data`完整shape/ID/hash/finite；NERSC格式/payload/geometry测试 | 不等于输入来源全部认证 |
| 基本冻结 | evaluate检查开发SHA、kernel SHA、参数 | 依赖闭包、恢复事务见P1 |
| paired入口 | 改用load_data/provenance，移除按协议名自动判定 | trust仍不完整，见P1 |
| invalid计数 | `invalid_audit_v1/`所查归档计数为0 | 计数器已做，评分拒绝规则未接入 |
| 谱盲点展示 | `spectral_quality_v1/`和说明文档 | 单指数外加真值，不是完整两态物理反例 |
| nominal配置 | develop/evaluate将协议索引传给risk | paired等旁路仍待统一 |
| 搜索基本防线 | u归一化/L1约束、mock与no-validation测试 | 不重复做；预算/SearchAdapter仍未落地 |
| 公共模块 | 五类接口、evidence、RunState、Ledger等已建立 | 不是“core只有io/stats”；需要集成验收 |
| 实测档案 | Wilson v2/v3/v4、图与JSON存在 | 本轮未重跑GPU，不升级为基态准确度证明 |

数组约定：links `(mu,t,z,y,x,3,3)`；空间d对应axis=3-d，时间axis=0。
圈为 `K(x)P(x+R)K†(x+T)P†(x)`，ReTr/(9×四维体积)，temporal links不smear。
势仍是log(mean W)对T的无权斜率；risk是SE²与窗口spread²之和，经开发Vref归一化。
它量化固定窗口精度/稳定性，不是已确认的E0总误差。

### 历史结果引用边界

- v2原点score：96.669 / 103.253 / 105.535；nested中位数比值3.013 / 3.425 / 3.410。
- v3原点score：263.3 / 232.8 / 221.9；历史nested区间均不含1。
- [已有v3图](../../results/wilson_loop/paired_s16t16_v3/score_paired.png)柱高为100×nested中位数，
  不是原点score；报告必须并列，不把两者混用。
- v4是同一S16T16B扩R=4的相关诊断，不是独立复现；评分尺度不同。
- 当前代码hash已改变，旧run为archived_diagnostic；保留历史实现与条件说明。
- 预注册需要validation打开前的plan身份；缺失则称事后诊断，不补造历史证据。

## 2. P0：最高优先——measure入口阻断（必做，CPU）

位置：`tasks/wilson_loop/measure.py::measure`。
`start=check_resume(...)`原为整数前缀长度，循环内被`start=perf_counter()`覆盖。
保存时 `spec['ids'][:start+len(rows)]` 用浮点slice，首次量测后即失败。
主agent已用**无磁盘mock完整入口**复现：
`TypeError: slice indices must be integers or None or have an __index__ method`，save未被调用。
80 tests全通过并未覆盖这个端到端断路。

- [ ] 分离completed_rows、cfg_start_time、已缓存rows数量，避免状态变量复用。
- [ ] CPU冷场/mock reader运行完整measure，不只直接测试check_resume。
- [ ] 覆盖空目录、部分前缀、已完整、limit、不同split及失败后恢复。
- [ ] 同时检查IDs长度、loops行数、records行数逐次相等，不重复或跳过cfg。

拟议产物：`results/wilson_loop/audit_<new-id>/measure_entry.json`及入口执行文字表。
验收：保存实际发生；不中断与恢复输出逐项一致；完整run重开不重复测量。
依赖：无GPU/原始gauge；用合成小场即可。此项失败时停止新生产。

## 3. P1：恢复事务与认证（必做，CPU）

现有generation/check_resume/trust_status是起点，不再重建另一套身份模块。

- [ ] 比较NPZ和provenance的共同generation；目前只验证NPZ generation。
- [ ] 恢复时验证loops shape/finite/内容身份、IDs与records前缀、已有gauge SHA。
- [ ] 单有provenance无NPZ时不得悄悄从0覆盖；单有NPZ时同样明确拒绝/回滚。
- [ ] 定义跨文件commit marker或完整generation恢复策略；不是仅分别atomic rename。
- [ ] 将measure/statistics/helper依赖闭包加入identity，不能只冻kernels.py。
- [ ] trust绑定protocol hash、development SHA、measurement generation、完整输入清单。
- [ ] 只有身份相符仍不足称physical；独立呈现operator/reference/numerical/statistical状态。
- [ ] 补NERSC checksum支持范围与失败路径；不伪造旧输入hash或旧运行环境。
- [ ] 定位投影失败batch索引，并由测量层附cfg/方向；不只给失败数量。

故障矩阵：NPZ替换前/后、provenance替换前/后、commit前/后；
另测generation错配、loops变更、记录重排、gauge变更、legacy与依赖变更。
拟议产物：同audit目录 `recovery_identity.json`、事务状态机及故障矩阵文字报告。
验收：恢复至同一可信完整前缀或明确拒绝，绝不混拼；每个篡改反例均被识别。
停止：无法证明一致恢复则禁resume并明确只允许新run，不宣称事务完成。

### P0数值边界补充（有限CPU包）

逐矩阵scale修复关闭；只补尚缺的domain/branch证据：

- [ ] 已知奇异值谱扫描，非零但秩亏、近奇异、极端scale、混batch。
- [ ] det相位接近±π、独立左右SU3变换；声明切口与支持域，不要求不存在的全局连续分支。
- [ ] 支持域内det/unitarity/协变容差合格，域外可定位的受控拒绝。

拟议产物：`projection_domain.json`及条件数×scale表/热图。
停止：支持域足够明确即可交付；GPU backend继续NOT_RUN，不无限扩大扫描。

## 4. P2：点值、invalid与非退化校准（必做，CPU）

已有point-vs-nested、unique-block、jackknife和block/seed图可保留；需纠正其含义。
`paired_audit.py`当前用500 inner/seed7重算“point”，不等于原1000/seed20261117点估计。
其coverage toy令cand=base/2，所有重采样比恒2；coverage=1.00是退化恒等，
**不能据此称区间偏保守或已校准**。

- [ ] 精确重放v2/v3/v4冻结seed/count/epsilon，另列诊断seed结果，禁止混作原点值。
- [ ] 诊断v2分布偏移，保留unique块、分子分母相关、每R SE/spread贡献。
- [ ] jackknife去除`index % len(keep)`取模重采样，使用明确的leave-one-out统计定义。
- [ ] 用非退化合成W产生同效/已知增益、不同相关性、共同偏差的数据集。
- [ ] 完整运行mean→log→window→inner风险→outer比值，报告coverage/FPR及Monte Carlo误差。
- [ ] invalid规则接入真实risk/paired/evaluate：非正mean/replica、NaN/Inf、零/近零分母。
- [ ] 预先决定拒绝或有限invalid阈值；不得仅在文档写“剔除”而代码照常NaN。
- [ ] 记录实际outer×inner路径invalid数；现audit seed11/500全0不涵盖全部nested路径。
- [ ] paired/旁路统一nominal、draw IDs、spec，不再默用nominal=1。
- [ ] 将block/τint依据、检验族、多比较与最小有用增益标清；不把exceedance当p值或posterior。

拟议产物：`paired_calibration.json`、精确点值表、coverage/FPR图、invalid触发表。
验收：冻结点值重放；合成零假设与已知效应均有非退化区间；invalid非零反例受控处理。
未完成正式功效/多比较时只称探索区间；跨1可验收INCONCLUSIVE，不需新GPU。

## 5. P3：物理谱反例与质量状态（必做，CPU）

现“平但错”图演示了目标盲点，不能充当非零基态overlap的谱验证。

- [ ] 构造 `W(T)=c0 exp(-E0 T)+c1 exp(-E1 T)`，c0>0，E1>E0，固定可复现噪声。
- [ ] 展示短窗口共同偏离E0但risk小，并扫描可识别/不可识别区间。
- [ ] 从有限Nt热迹/静态源sector写出适用谱，区分零温近似与有限温表达。
- [ ] 不直接套pion cosh，也不把等幅回绕的反例误当Wilson圈必有该项。
- [ ] 有限CPU比较log bias、相关协方差条件数与预声明拟合；不无限搜索参考。
- [ ] 同步gate语义/配置名及unused window_z；独立记录门控不足与abstain。

拟议产物：`spectral_physical_counterexample.json`、两态有效势/窗口图及有限Nt推导文字。
验收：明确哪些共同偏差能/不能被现数据排除；不把risk改善升级为E0准确度。
停止：不可识别结论可交付；更晚T/独立anchor仅列可选新数据路线。

## 6. 公共框架 F0–F3：已建类的有限集成验收（必做，CPU）

不重建五接口；沿现有Task/Run、Operator/Candidate、Measurement、
Reference/Metric/Comparator、Search及evidence/state类接通两任务。
公共文件由一名集成负责人修改；task负责人提供adapter与反例。

### F0：证据必须非空、完整、绑定实际输入

- [ ] `scripts/audit_run`的code/inputs不得默认空后仍认证PASS。
- [ ] 两个kernel hash都missing的None==None不能通过；必需字段缺失为NOT_RUN/拒绝。
- [ ] pion完整ID/行数按RunSpec验证，禁止expected使用len(data)自证。
- [ ] 两个bundle_for使用可信加载，保留IDs、protocol/code/input hashes，不只查shape/finite。
- [ ] `core.operator_checks.covariance_residual` 固定U的检查明确仅测算符与field变换的交换关系；
  完整gauge covariance需同时变换U与field，接口/文字不可混称。

拟议产物：`results/framework/audit_<new-id>/evidence_bindings.json`和缺字段/篡改拒绝表。
验收：两task真实adapter产生非空身份；删必需字段或改输入必拒绝认证。

### F1：draws、广播与比较接口

主agentCPU反例：check_draws接受(3,1)与(3,)同size=3，paired_ratio广播成draws=9。

- [ ] 对一维、shape完全相同、有限性、draw IDs/spec/baseline族作强制验证。
- [ ] 非法shape不flatten后假装配对；合法3 draws输出必须仍3。
- [ ] 两task同spec输入通过同Comparator得到同统计定义；缓存能力声明不推测缺失cross数据。

拟议产物：`comparator_contract.json`，含shape/draw/spec错配矩阵与合法一致性数值表。
验收：广播反例被拒；所有比较结果可追溯同一draw计划。

### F2：SearchAdapter与预算真正进入loop

u约束和no-validation防线已完成，保留回归即可。
当前loop import Budget/Search但仍手写循环；文档“已接入”应按实际修正。
主agent反例：SearchAdapter接受NaN评估结果为accepted。

- [ ] LLM与optimizer共享候选验证/Evaluator/SearchAdapter，不仅导入模块。
- [ ] 拒绝NaN/Inf evaluator结果并计入预算/失败原因；禁止accepted。
- [ ] 持久化提案/评估/失败/早停与seed预算；mock验证超预算不可再评估。
- [ ] mock全链验证development期间不读取validation，保留既有非法u回归。

拟议产物：`mock_search_trace.json`及预算耗尽/非法结果/无validation访问文字报告。
验收：真实loop路径出现预算账本，NaN受控拒绝，无API调用。

### F3：RunState从绑定证据导出，与旧score一致

主agent反例：reference=failed而statistical/comparison=validated仍可返回42，
operator/numerical甚至未跑。develop又按测试文件名/块数直接写validated，缺证据绑定。
evaluate两task的comparison仍draft/scoreNone，但顶层旧score可能有数。

- [ ] 定义conditional diagnostic与physical状态，参考/算符/数值未过不得认证physical score。
- [ ] 状态转换要求hash绑定的验收证据，不能按测试文件名或块数赋validated。
- [ ] 同一结果的顶层score、RunState.score、comparison verdict一致；legacy明确诊断字段。
- [ ] 两task端到端develop/evaluate/bundle/comparator/search测试同一状态流。
- [ ] fold-local/reference/covariance/feature选择与预算、manifest/drawIDs绑定；未接的学习重训明确NOT_RUN。

拟议产物：`state_integration.json`、状态转换矩阵与两task全链文字报告。
验收：所有失败/未跑组合不能返回physical数值；合法conditional诊断不被误删。

## 7. F4–F5：生产边界与有限关闭条件

F4现有HMC toy可逆性/ΔH阶与τint≈0.512仅属窄证据；生产force/reversibility/ΔH/
接受率与自相关完整记录未新增验证，保留NOT_RUN，不能由高接受率认证系综。
CPU可先完善生产证据schema与toy检查；新GPU生产不属于本轮完成的必要条件。

| 可选路线 | 依赖与可交付 | 停止条件 |
|---|---|---|
| stout | 同协变/支撑/成本测试，APE对照图 | 无收益可停止 |
| GEVP | 新cross loops设计、T0/条件数冻结；现Wii不能拼Cij | 无平台/不可识别可验收 |
| 多体积/动量/训练网络 | geometry/相位/表示与新响应身份分别设计 | 不用一个扩展替代全部验证 |
| multilevel/新估计器 | 局域子更新、无偏性与cost-error图 | 独立于kernel-only赛道 |
| 新盲测 | 未见样本、冻结计划、τint/功效预算 | 跨1/质量失败保留负结果 |

原始方法依据：[stout](https://arxiv.org/abs/hep-lat/0311018)、
[GEVP](https://arxiv.org/abs/0902.1265)、[multilevel](https://arxiv.org/abs/hep-lat/0108014)。
它们不保证当前R/T收益。旧已观察validation不能改名恢复盲性。

F5执行顺序：P0入口→P1恢复→F0/F1/F3身份/状态→P2/P3→F2真实mock搜索。
每包交实际路径、命令、seed、输入/源码身份和图或数值文字报告，再更新coverage。
关闭条件：上述必做CPU包逐项有证据，未跑生产与可选路线明确延期/停止，
报告删除已被反例推翻的完成主张；无需positive结果或新GPU才能完成本轮。
