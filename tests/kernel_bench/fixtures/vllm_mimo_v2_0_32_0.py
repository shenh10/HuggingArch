# Verbatim loader oracle from vLLM v0.32.0, Apache-2.0.
# Source: vllm-project/vllm commit 5df63a7203cdb9328d824a27b682e77c7b9928be,
# vllm/model_executor/models/mimo_v2.py:495-632.
# The test extracts this function by AST and supplies torch, cdiv and a shape-
# spying _requantize_fp8; it never imports or calls HuggingArch's declaration.
def _shard_fp8_qkv_proj(
    w_full: torch.Tensor,
    s_full: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    v_head_dim: int,
    tp_rank: int,
    tp_size: int,
    ckpt_tp: int,
    block: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shard the fp8 qkv_proj weights for ``tp_rank``.

    The checkpoint stores the fused QKV pre-sharded for ``ckpt_tp`` ranks (the
    model config's ``num_key_value_heads``), each chunk holding that slice's
    Q, K and V rows:

        [Q_0 | K_0 | V_0 | Q_1 | K_1 | V_1 | ... | Q_n | K_n | V_n]   (n = ckpt_tp)

    Per chunk, Q has ``(num_heads / ckpt_tp) * head_dim`` rows, K has
    ``(num_kv_heads / ckpt_tp) * head_dim`` rows, and V has
    ``(num_kv_heads / ckpt_tp) * v_head_dim`` rows, and the fp8 block scales
    are tiled per chunk too (``ceil(rows_per_chunk / block)`` rows each).

    ``ckpt_tp`` is not the layer's KV-head count: a MiMo-V2.5 SWA layer has 8
    KV heads over 4 chunks, so each chunk carries two KV heads (3712 rows = 29
    blocks) while a GA layer has 4 KV heads over 4 chunks (3392 rows each).

    The forward expects each rank's slice de-interleaved:

        [Q_1 | Q_2 | ... | Q_g | K_1 | K_2 | ... | K_g | V_1 | V_2 | ... | V_g]

    When ``tp_size == ckpt_tp`` the checkpoint chunk *is* that layout, so a
    plain chunk of both weight and scale suffices. Otherwise each rank's Q, K
    and V rows are gathered from the chunks that hold them, dequantized with
    the chunk's own scales, reordered, and re-quantized to fp8.
    """
    assert num_heads % tp_size == 0, (
        f"num_heads={num_heads} must be divisible by tp_size={tp_size}."
    )
    if ckpt_tp <= 0 or num_heads % ckpt_tp or num_kv_heads % ckpt_tp:
        raise ValueError(
            f"fused qkv_proj is pre-sharded at {ckpt_tp} chunks, which do not "
            f"divide num_heads={num_heads} / num_kv_heads={num_kv_heads}."
        )
    # When there are fewer KV heads than ranks, vLLM replicates them
    # (`num_kv_head_replicas`) and rank r owns KV head r // replicas, which
    # keeps every Q head grouped with the KV head it attends to.
    if tp_size <= num_kv_heads:
        assert num_kv_heads % tp_size == 0, (
            f"num_kv_heads={num_kv_heads} must be divisible by tp_size={tp_size}."
        )
        kv_head_ids = list(
            range(
                tp_rank * (num_kv_heads // tp_size),
                (tp_rank + 1) * (num_kv_heads // tp_size),
            )
        )
    else:
        assert tp_size % num_kv_heads == 0, (
            f"tp_size={tp_size} must be divisible by num_kv_heads={num_kv_heads}."
        )
        kv_head_ids = [tp_rank // (tp_size // num_kv_heads)]
    q_head_ids = list(
        range(tp_rank * (num_heads // tp_size), (tp_rank + 1) * (num_heads // tp_size))
    )

    rows_per_chunk = w_full.shape[0] // ckpt_tp
    q_per_chunk = (num_heads // ckpt_tp) * head_dim
    k_per_chunk = (num_kv_heads // ckpt_tp) * head_dim
    v_per_chunk = (num_kv_heads // ckpt_tp) * v_head_dim
    if q_per_chunk + k_per_chunk + v_per_chunk != rows_per_chunk:
        raise ValueError(
            f"fused qkv_proj has {w_full.shape[0]} rows, not {ckpt_tp} chunks of "
            f"{q_per_chunk} Q + {k_per_chunk} K + {v_per_chunk} V rows."
        )

    # The scales are tiled per chunk; they collapse to one continuous grid when
    # a chunk is a whole number of blocks (the SWA chunks are: 3712 = 29 * 128).
    chunk_scale_rows = cdiv(rows_per_chunk, block)
    rows = torch.arange(w_full.shape[0])
    per_chunk_scales = s_full.shape[0] == ckpt_tp * chunk_scale_rows
    if per_chunk_scales:
        scale_index = (rows // rows_per_chunk) * chunk_scale_rows + (
            rows % rows_per_chunk
        ) // block
    elif s_full.shape[0] == cdiv(w_full.shape[0], block):
        scale_index = rows // block
    else:
        raise ValueError(
            f"fused qkv_proj scale has {s_full.shape[0]} rows, expected either "
            f"{ckpt_tp * chunk_scale_rows} ({ckpt_tp} chunks of "
            f"{chunk_scale_rows} rows) or {cdiv(w_full.shape[0], block)} "
            f"(one continuous grid)"
        )

    if tp_size == ckpt_tp and per_chunk_scales:
        # One checkpoint chunk per rank: already [Q | K | V] for that rank, and
        # its scale rows line up with ceil(rows_per_chunk / block).
        return (
            w_full.chunk(ckpt_tp, dim=0)[tp_rank],
            s_full.chunk(ckpt_tp, dim=0)[tp_rank],
        )

    # Gather this rank's Q, K and V rows from the chunks that hold them.
    q_heads_per_chunk = num_heads // ckpt_tp
    kv_heads_per_chunk = num_kv_heads // ckpt_tp
    head_rows = torch.arange(head_dim)
    v_head_rows = torch.arange(v_head_dim)
    row_index: list[torch.Tensor] = []
    for head in q_head_ids:
        chunk = head // q_heads_per_chunk
        row_index.append(
            chunk * rows_per_chunk + (head % q_heads_per_chunk) * head_dim + head_rows
        )
    for head in kv_head_ids:
        chunk = head // kv_heads_per_chunk
        row_index.append(
            chunk * rows_per_chunk
            + q_per_chunk
            + (head % kv_heads_per_chunk) * head_dim
            + head_rows
        )
    for head in kv_head_ids:
        chunk = head // kv_heads_per_chunk
        row_index.append(
            chunk * rows_per_chunk
            + q_per_chunk
            + k_per_chunk
            + (head % kv_heads_per_chunk) * v_head_dim
            + v_head_rows
        )
    index = torch.cat(row_index)
    grouped = w_full[index].to(torch.float32) * s_full[
        scale_index[index]
    ].repeat_interleave(block, dim=1)
    return _requantize_fp8(grouped, index.numel(), block, w_full.dtype)
