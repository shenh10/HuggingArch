已完成复审并写入：

`/scratch1/ha_review_codex/REVIEW_CODEX_design_r2.md`

结论：**不签**。

核心理由：

1. **oracle 当前会自我致盲**
   - `tools/walk_answers.py` 用字符串包含关系推导 peak family，`BF16` 与 `bf16` 不匹配；应走已有的 `peak_kind(dtype)` 声明。
   - `_one_card` 捕获所有异常并把所有空行结果标为 `infeasible`，随后统一剔除，可能吞掉结构性反例。

2. **`row_signature` 不足以承担 oracle 结论**
   - 它只比较 8 个 measurement-key 相关字段，漏掉 `aggregation_id`、原始 op、count、cast/collective 结构、launch geometry、symbolic law shape 等。因此不满足本仓“per-op row 集合与顺序含 dtype”的判例要求。

3. **HBM 的归属方向正确，但前提未闭合**
   - HBM 不应进入 walk answer set，这一点与现有 `Card.klass` / `Card.price` 分层一致。
   - 但两张均 feasible 的卡可能 bind 不同 PP stage；§5 目前只写 “layout feasible on both”，未显式加入 same bound stage 前提或声明 projection-time stage gate。

4. **216→144 数字算法本身对，但口径不足以称为 Stage A sound-key 收益**
   - 仅覆盖 vllm、EP8·DP8、一个 request、每类一张代表卡，且已剔除 refused/infeasible 卡。

第 2 条 R1 阻塞（`B: [1]` / static-axis home）允许在 oracle-first 阶段继续作为未决项，但不能进入后续 Stage A 实现。