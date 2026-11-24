已完成评审并写入：

`/scratch1/ha_review_codex/REVIEW_CODEX_design.md`

结论为**不签**。核心阻塞：

1. §5 的五类“答案”不充分：精度 profile 可改变 cache law 与 cast 节点集合；HBM/memory fraction 可改变 refusal 与 PP bound stage；`world_local.local_in` 也不是单纯整除问题。
2. §3.1 从 binding `dims` 推导 workload 静态数的规则不完整，尤其未处理 `B: [1]` 与 `n_verify -> s_verify` 映射，无法保证现集合不变。
3. §8 的 `2,739` 只是 layout 并集，不是按 `(model, layout)` 的 card-answer 等价类计数；4.87× 是上界而非可声称的 Stage A 精确收益。
4. oracle 需覆盖 precision launch、required edge dtype、`materialize_casts`、实际 exact-division 与 stage mapping，并且必须 offline/opt-in，避免侵入 20s 请求路径。