# v4目录产物

主说明与接口契约见 [DATA_PLATFORM.md](../DATA_PLATFORM.md)。执行 `build_catalog.py` 仅作CPU数学/已有源开发证据检查。

- `../TASK_CATALOG.csv`：现有57条任务/模板/编码接口记录，本轮用途另列。
- `DIRECT_RELATION_DEFINITIONS.csv`：35条定义关系与证明草图。
- `RELATION_GRAPH.csv`：132条D8输入动作/仿射答案映射闭包边，包含复合动作。
- `AGGREGATE_RELATIONS.csv`：联合公式及其证明范围。
- `BACKGROUND_DESIGN.json`：两目标、三背景、固定候选槽位和改变理由。
- `BACKGROUND_RELATION_GRAPH.csv` / `BACKGROUND_AGGREGATE_RELATIONS.csv`：实际节点中的关系及联合公式缺失成员。
- `SOURCE_REFERENCE_CURVES.csv` / `SOURCE_ANSWER_DISTRIBUTIONS.csv` / `BACKGROUND_MATCHING.csv`：已有源任务单任务曲线、训练答案分布及背景变化的剩余差异。
- `RELATION_AUDIT.json` / `CATALOG_AUDIT.json` / `SOURCE_REFERENCE_AUDIT.json`：检查范围、结果、来源hash及读取边界。
- `REPLACEMENT_RELATION_COUNTEREXAMPLES.csv`：已有六对候选的八动作答案映射反例，含本轮两对；不宣称数学无关。

这是现有关系的明确登记与复合闭包，不是所有数学关系的完备数据库；不是神经表征机制或迁移收益的证明。
