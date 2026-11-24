已完成 R3 复审并写入：

`/scratch1/ha_review_codex/REVIEW_CODEX_design_r3.md`

结论：**不签**。

关键复核结果：

- 当前 commit 下重新跑完 63 个模型：216 walks → 144 groups，0 violations，91 over-fine，99 infeasible，账实相符。
- `peak_kind` 修复有效，字段确实进入比较；故意改 RTX 的 symbolic law 会立刻红，全 formula 比较会因 substituted peak 产生假反例。
- 空结果无声明 refusal 的路径已保留并参与比较。
- 但 raised refusal 路径未落实：`_one_card` 返回的 raised 记录缺少 `dtypes`/`ops`，`audit_model` 读取 union 时必然 `KeyError: 'dtypes'`，因此 structural refusal 仍没有被 oracle 有效比较。
- `row_signature` 仍不是 canonical row/`PointRow` 投影，漏多处 symbolic law 与 launch-record 字段。
- PP stage 规则本身存在，但 Stage A 需要逐调用点证明替换 class predicate 后 stage guard 确实被询问；`sol_law.cardBinding` 目前没有相邻 stage check。