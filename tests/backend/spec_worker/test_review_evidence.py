"""A review's claims are checked against the run's snapshot before they count.

A review is an agent's reading of the model, and it can be wrong: a NO_PASS
may order a window or a layer count that config.json and the checkpoint
contradict, and a PASS may describe code the forward does not have. Every
model, file, line and report below is synthetic. Each fixture reproduces a
shape real reports take (a defect with or without tags, a Table F row in the
tag form or one of its variants, a Defects section followed by notes), and
the snapshot facts are the few lines the claims point at.
"""
from __future__ import annotations

from backend.hf.header import SafetensorsHeader
from backend.spec_worker._prompts.fix import build_fix_task_body
from backend.spec_worker.review_evidence import (
    CONTRADICTED, UNVERIFIED, VERIFIED, ReviewEvidence, SnapshotFacts,
)
from backend.spec_worker.reviewer import _final_verdict

# -- a model with a draft stack and a windowed patch encoder ----------------

_ORCA_HEAD = [
    "class OrcaSelfAttention(nn.Module):",                                     # 1
    "    def __init__(self, config, layer_idx):",                              # 2
    "        self.sliding_window = config.sliding_window",                     # 3
    "        self.local_radius = config.local_radius",                         # 4
    "    def _local_mask(self, n, device, dtype):",                            # 5
    "        scores = scores.masked_fill((qpos - kpos).abs() > self.radius, float('-inf'))",  # 6
]
# The load-time skip list sits at the lines the defect below cites.
ORCA_SOURCE = "\n".join(
    _ORCA_HEAD + ["    pass"] * (1404 - len(_ORCA_HEAD)) + [
        "    _skip_keys_on_load = [",                                          # 1405
        '        r"model\\.draft\\..*",',                                      # 1406
        "    ]",                                                               # 1407
    ])

def _header(stored):
    """A header holding `{template: (shape, layers)}`: the tensor at each
    listed index of the template's `{}`."""
    return SafetensorsHeader.from_tensors({
        template.format(i): ("BF16", shape)
        for template, (shape, layers) in stored.items() for i in layers})


ORCA = SnapshotFacts(
    config={"sliding_window": 128, "rope_theta": 5000000, "attention_bias": False,
            "audio_config": {"rope_theta": 320000},
            "vision_config": {"local_radius": 48, "window_size": 96}},
    header=_header({
        "model.draft.layers.{}.enorm.weight": ([4096], range(3)),
        "model.draft.layers.{}.self_attn.qkv_proj.weight": ([12288, 4096], range(3)),
    }),
    sources={"modeling_orca.py": ORCA_SOURCE},
)

# An untagged defect: its numbers (6 layers, a 2048 window) come from a
# model card, and it cites a line without quoting it.
UNTAGGED_DRAFT_DEFECT = (
    "### Defects\n"
    "- **Draft stack unresolved** (step 21). The modeling file skips `model.draft.*` "
    "(modeling_orca.py:1406); the model card lists 6 draft layers with a 2048-token "
    "window. The spec takes `n_draft_layers` from a default, runs the draft "
    "attention at s_window=128, and feeds both of its norms from the same input. "
    "Tags: cost, KV. Fix: read the layer count from the checkpoint, give the draft "
    "attention a window of its own, and route the first norm from the embedding.\n"
)

# A window claim in the tag form, whose `cfg:` value config.json contradicts.
TAGGED_WINDOW_CLAIM = (
    "### Defects\n"
    "- **Draft attention window** `src: modeling_orca.py:1405-1406 "
    '"_skip_keys_on_load = [ ... model\\.draft\\..*"` '
    "`cfg: sliding_window = 2048` Fix: set the draft attention window to 2048.\n"
)

VERIFIED_WINDOW_DEFECT = (
    "### Defects\n"
    "- **Patch window mask missing** `src: modeling_orca.py:6 "
    '"scores.masked_fill((qpos - kpos).abs() > self.radius"` '
    "`cfg: vision_config.local_radius = 48` Fix: use the swa_* ops "
    "on the windowed patch blocks with a radius of 48.\n"
)


def _fix_body(report: str, facts: SnapshotFacts) -> str:
    return build_fix_task_body(
        "acme/Orca-7B", "a: 1", [], {"valid": True},
        ReviewEvidence.audit(report, facts), facts=facts)


def test_window_claim_contradicted_by_config_never_reaches_the_fixer():
    ev = ReviewEvidence.audit(TAGGED_WINDOW_CLAIM, ORCA)
    [finding] = ev.findings
    assert finding.status == CONTRADICTED
    assert "128" in " ".join(finding.reasons())

    body = _fix_body(TAGGED_WINDOW_CLAIM, ORCA)
    assert "set the draft attention window to 2048" not in body
    work_order = body.split("### Work order", 1)[1].split("### Unverified", 1)[0]
    assert "Draft attention window" not in work_order
    # withheld: the false tag and what config.json holds, nothing else
    withheld = body.split("### Withheld", 1)[1]
    assert "config.json `sliding_window`" in withheld and "128" in withheld
    assert "Draft attention window" not in withheld


def test_untagged_defect_is_only_a_lead_to_confirm():
    # Its numbers (6, 2048) come from a model card, not from config.json or
    # the checkpoint, and it quotes no source line: not a work order.
    [finding] = ReviewEvidence.audit(UNTAGGED_DRAFT_DEFECT, ORCA).findings
    assert finding.status == UNVERIFIED
    assert "2048" in " ".join(finding.reasons())

    body = _fix_body(UNTAGGED_DRAFT_DEFECT, ORCA)
    work_order = body.split("### Work order", 1)[1].split("### Unverified", 1)[0]
    assert "Draft stack unresolved" not in work_order
    unverified = body.split("### Unverified", 1)[1].split("### Withheld", 1)[0]
    # a lead: its title and where to look, never its claims or its fix
    assert "Draft stack unresolved" in unverified
    assert "modeling_orca.py:1406" in unverified
    assert "2048" not in body and "model card" not in body
    assert "give the draft attention a window of its own" not in body
    # the fixer sees what the checkpoint stores
    assert "model.draft.layers: 3" in body


def test_layer_count_claim_is_checked_against_the_checkpoint():
    six = ORCA.check_tag("count", "model.draft.layers = 6")
    three = ORCA.check_tag("count", "model.draft.layers = 3")
    assert six.status == CONTRADICTED and three.status == VERIFIED


def test_tensor_shape_claim_is_checked_against_the_checkpoint():
    ok = ORCA.check_tag("ckpt", "model.draft.layers.2.self_attn.qkv_proj.weight = [12288, 4096]")
    wrong = ORCA.check_tag("ckpt", "model.draft.layers.X.self_attn.qkv_proj.weight = [4096, 12288]")
    missing = ORCA.check_tag("ckpt", "model.draft.layers.X.fuse_proj.weight = [4096, 8192]")
    assert ok.status == VERIFIED
    assert wrong.status == CONTRADICTED and missing.status == CONTRADICTED


def test_defect_with_quoted_source_and_backed_numbers_is_the_work_order():
    [finding] = ReviewEvidence.audit(VERIFIED_WINDOW_DEFECT, ORCA).findings
    assert finding.status == VERIFIED
    body = _fix_body(VERIFIED_WINDOW_DEFECT, ORCA)
    work_order = body.split("### Work order", 1)[1].split("### Unverified", 1)[0]
    assert "Patch window mask missing" in work_order
    # A review that orders an edit is the work order, not the validator.
    invalid = build_fix_task_body(
        "m", "a: 1", [], {"valid": False, "errors": ["e"]},
        ReviewEvidence.audit(VERIFIED_WINDOW_DEFECT, ORCA), facts=ORCA)
    assert "No finding of this review checked out" not in invalid
    assert "EXACTLY the defects\nthe review named" in invalid


def test_number_no_verified_tag_carries_leaves_a_defect_unverified():
    report = VERIFIED_WINDOW_DEFECT.replace("a radius of 48", "a radius of 72")
    [finding] = ReviewEvidence.audit(report, ORCA).findings
    assert finding.status == UNVERIFIED
    assert "72" in " ".join(finding.reasons())


# -- a model whose attention has no rotary step -----------------------------

HERON_SOURCE = "\n".join([
    "class HeronAttention(nn.Module):",                                             # 1
    "    def forward(self, hidden, mask=None, cache=None):",                           # 2
    "        q = self.q_proj(hidden).view(shape).transpose(1, 2)",                     # 3
    "        k = self.k_proj(hidden).view(shape).transpose(1, 2)",                     # 4
    "        out, weights = attend(",                                                  # 5
])
HERON = SnapshotFacts({"partial_rotary_factor": 1.0}, _header({}),
                      {"modeling_heron.py": HERON_SOURCE})
TABLE_F = ("| # | forward step | source | spec node(s) | tags | status |\n"
           "|---|---|---|---|---|---|\n")
TABLE_S = ("\n| component / node | op | forward step # |\n|---|---|---|\n"
           "| q_proj, k_proj | linear | 1 |\n")

# A free-form PASS line: no Table F, no tag, a claim about rotary.
FREEFORM_HERON_PASS = (
    "- **Attention**: GQA, 16 query heads / 4 kv heads / head dim 64, rotary on "
    "every dimension (partial_rotary_factor=1.0), no bias — `gqa_attention` "
    "agrees with `HeronAttention` and the tensor shapes.\n")


def test_pass_without_a_verified_forward_mapping_is_not_recorded():
    ev = ReviewEvidence.audit(FREEFORM_HERON_PASS, HERON)
    assert ev.pass_blockers()
    assert _final_verdict("pass", {"valid": True}, ev) == "error"


def test_pass_quoting_code_the_forward_does_not_have_is_not_recorded():
    rope_row = TABLE_F + (
        '| 1 | q/k rotary | `src: modeling_heron.py:3-4 "q, k = '
        'rotate(q, k, cos, sin)"` | q_rope, k_rope '
        "| [sem] | match |\n")
    ev = ReviewEvidence.audit(rope_row, HERON)
    assert ev.rows[0].status == CONTRADICTED
    assert _final_verdict("pass", {"valid": True}, ev) == "error"

    plain_row = TABLE_F + (
        '| 1 | q/k projections, no rotary | `src: modeling_heron.py:3-4 "q = '
        'self.q_proj(hidden) ... k = self.k_proj(hidden)"` '
        "| q_proj, k_proj | [sem] | match |\n") + TABLE_S
    ev = ReviewEvidence.audit(plain_row, HERON)
    assert ev.rows[0].status == VERIFIED
    assert _final_verdict("pass", {"valid": True}, ev) == "pass"


# A docstring that writes its names in backticks, as markdown docstrings do.
PIPIT = SnapshotFacts({}, _header({}), {"preprocess_pipit.py": "\n".join([
    "def pack(images):",                                                       # 1
    '    """`pixel_values` holds every patch; each image has `prod(grid[i])`.',  # 2
    "    Only the `<|media_pad|>` positions are embedding slots.",             # 3
    '    """',                                                                 # 4
])})


def test_a_quote_holding_backticks_is_read_to_its_closing_quote():
    # The quote's own backticks once ended the tag at the first one, so a
    # verbatim docstring quote read as no quote and a PASS was not recorded.
    report = TABLE_F + (
        '| 1 | patches | `src: preprocess_pipit.py:2 "`pixel_values` holds every '
        'patch; each image has `prod(grid[i])`."` | patchify | | match |\n'
        '| 2 | media slots | `src: preprocess_pipit.py:3 "Only the `<|media_pad|>` '
        'positions are embedding slots."` | embed_merge | | match |\n') + TABLE_S
    ev = ReviewEvidence.audit(report, PIPIT)
    assert [r.status for r in ev.rows] == [VERIFIED, VERIFIED]
    assert _final_verdict("pass", {"valid": True}, ev) == "pass"

    wrong = report.replace("holds every patch", "holds one patch")
    assert ReviewEvidence.audit(wrong, PIPIT).rows[0].status == CONTRADICTED


def test_config_key_the_config_does_not_hold_is_contradicted():
    absent = ORCA.check_tag("cfg", "num_draft_layers = 6")
    said_absent = ORCA.check_tag("cfg", "num_draft_layers = absent")
    assert absent.status == CONTRADICTED and said_absent.status == VERIFIED


def test_citation_of_a_file_the_run_did_not_stage_cannot_verify():
    report = ("### Defects\n- **Cache dtype** `src: runtime/kv_cache.py:310 "
              '"cache_dtype = fp8"` Fix: record the dtype.\n')
    [finding] = ReviewEvidence.audit(report, ORCA).findings
    assert finding.status == UNVERIFIED


# vLLM ships one model under several backends, each with its own `model.py`.
_SHARED = "        hidden_states = self.embed_tokens(input_ids)"
TWIN = SnapshotFacts({}, _header({}), {
    "vllm/models/kimi_k3/amd/model.py": "\n".join(
        ["import torch", "", _SHARED, "        return self.amd_head(hidden_states)"]),
    "vllm/models/kimi_k3/nvidia/model.py": "\n".join(
        ["import torch", "", _SHARED, "        return self.lm_head(hidden_states)"]),
})


def test_a_file_name_two_staged_files_share_is_resolved_by_its_quote():
    bare = TWIN.check_tag("src", 'model.py:4 "return self.lm_head(hidden_states)"')
    tail = TWIN.check_tag("src", 'nvidia/model.py:4 "return self.lm_head(hidden_states)"')
    assert bare.status == VERIFIED and tail.status == VERIFIED
    assert bare.lines == tail.lines == ("vllm/models/kimi_k3/nvidia/model.py", 4, 4)


def test_a_quote_both_same_named_files_hold_does_not_say_which_file():
    check = TWIN.check_tag("src", 'model.py:3 "hidden_states = self.embed_tokens(input_ids)"')
    assert check.status == UNVERIFIED
    assert "names 2 staged files" in check.detail
    assert "vllm/models/kimi_k3/amd/model.py" in check.detail


def test_one_bad_tag_withholds_only_its_own_defect():
    # Classification is per top-level Defects bullet: the contradicted window
    # claim does not hold back the separate, verified patch-mask repair.
    report = TAGGED_WINDOW_CLAIM + VERIFIED_WINDOW_DEFECT.removeprefix("### Defects\n")
    ev = ReviewEvidence.audit(report, ORCA)
    assert [f.status for f in ev.findings] == [CONTRADICTED, VERIFIED]
    work_order = ev.work_order().split("### Unverified", 1)[0]
    assert "Patch window mask missing" in work_order


def test_withheld_defect_title_does_not_reach_the_fixer():
    report = TAGGED_WINDOW_CLAIM.replace("Draft attention window", "Set the draft window to 2048 now")
    body = _fix_body(report, ORCA)
    assert "Set the draft window to 2048 now" not in body


def test_top_level_key_repeated_in_a_sub_config_is_not_contradicted_by_shorthand():
    # Two rope_theta values: the backbone's 5e6 and the audio tower's 320000.
    assert ORCA.check_tag("cfg", "rope_theta = 320000").status == UNVERIFIED
    assert ORCA.check_tag("cfg", "audio_config.rope_theta = 320000").status == VERIFIED
    assert ORCA.check_tag("cfg", "config.rope_theta = 5000000").status == VERIFIED
    assert ORCA.check_tag("cfg", "config.rope_theta = 320000").status == CONTRADICTED


def test_a_bare_weight_name_does_not_verify():
    # every norm in the draft layer is a [4096] `weight`; the name must say which
    assert ORCA.check_tag("ckpt", "weight = [4096]").status == UNVERIFIED
    assert ORCA.check_tag("ckpt", "model.draft.layers.X.enorm.weight = [4096]").status == VERIFIED


CLAMPED = SnapshotFacts({"clamp_limit": 6.0}, _header({}), {"modeling_kestrel.py": "\n".join([
    "    def forward(self, gu):",                                          # 1
    "        g, u = gu.chunk(2, dim=-1)",                                  # 2
    "        g = g.clamp(max=self.cap)",                                   # 3
    "        u = u.clamp(min=-self.cap, max=self.cap)",                    # 4
    "        return self.act(g) * u",                                      # 5
])})


def test_elided_quote_pieces_must_appear_in_source_order():
    # the clamped gated FFN clamps the gate, then the up projection
    ordered = CLAMPED.check_tag("src", 'modeling_kestrel.py:3-4 "g = g.clamp(max=self.cap)'
                                       ' ... u = u.clamp(min=-self.cap, max=self.cap)"')
    reversed_ = CLAMPED.check_tag("src", 'modeling_kestrel.py:3-4 "u = u.clamp(min=-self.cap,'
                                         ' max=self.cap) ... g = g.clamp(max=self.cap)"')
    assert ordered.status == VERIFIED
    assert reversed_.status != VERIFIED


def test_a_layer_count_of_zero_is_true_when_the_checkpoint_has_none():
    # the Heron checkpoint ships no draft layers
    assert HERON.check_tag("count", "model.draft.layers = 0").status == VERIFIED
    assert HERON.check_tag("count", "model.draft.layers = 1").status == CONTRADICTED


HERON_QK_ROW = (
    '`src: modeling_heron.py:3-4 "q = self.q_proj(hidden) ... '
    'k = self.k_proj(hidden)"`')


def test_pass_with_a_verified_row_whose_status_is_a_mismatch_is_not_recorded():
    report = TABLE_F + f"| 1 | q/k projections | {HERON_QK_ROW} | q_proj | [sem] | MISMATCH |\n"
    ev = ReviewEvidence.audit(report, HERON)
    assert ev.rows[0].status == VERIFIED
    assert _final_verdict("pass", {"valid": True}, ev) == "error"


def test_pass_listing_an_unverified_defect_is_not_recorded():
    report = (TABLE_F + f"| 1 | q/k projections | {HERON_QK_ROW} | q_proj | [sem] | match |\n"
              + "\n### Defects\n- **Attention rotary** the attention rotates every dimension.\n")
    ev = ReviewEvidence.audit(report, HERON)
    assert _final_verdict("pass", {"valid": True}, ev) == "error"


def test_pass_with_an_unsourced_spec_node_is_not_recorded():
    report = (TABLE_F + f"| 1 | q/k projections | {HERON_QK_ROW} | q_proj | [sem] | match |\n"
              + "\n| component / node | op | forward step # |\n|---|---|---|\n"
              + "| q_rope | rope | UNSOURCED |\n")
    ev = ReviewEvidence.audit(report, HERON)
    assert _final_verdict("pass", {"valid": True}, ev) == "error"


# Round 2: a finding is verified only when every check in it verifies. A
# verified quote grounds a finding; it does not vouch for its other tags.
MASK_QUOTE = ('`src: modeling_orca.py:6 "scores.masked_fill((qpos - kpos).abs() '
              '> self.radius"`')


def _work_order(report: str) -> str:
    return _fix_body(report, ORCA).split("### Work order", 1)[1].split("### Unverified", 1)[0]


def test_pass_row_with_an_uncheckable_config_tag_is_not_recorded():
    # Two rope_theta values (backbone 5e6, audio 320000): the short key
    # cannot be checked, so the row is not verified, whatever its quote.
    ev = ReviewEvidence.audit(
        TABLE_F + f"| 1 | window mask | {MASK_QUOTE} `cfg: rope_theta = 320000` "
        "| attn | [sem] | match |\n" + TABLE_S, ORCA)
    assert ev.rows[0].status == UNVERIFIED
    assert _final_verdict("pass", {"valid": True}, ev) == "error"


def test_defect_with_an_ambiguous_short_key_is_not_a_work_order():
    report = ("### Defects\n- **Audio rotary base** " + MASK_QUOTE
              + " `cfg: rope_theta = 320000` Fix: use the audio base everywhere.\n")
    [finding] = ReviewEvidence.audit(report, ORCA).findings
    assert finding.status == UNVERIFIED
    assert "Audio rotary base" not in _work_order(report)
    assert "use the audio base everywhere" not in _fix_body(report, ORCA)


def test_defect_with_a_bare_tensor_name_is_not_a_work_order():
    # every draft norm is a [4096] `weight`: the tag names none of them
    report = ("### Defects\n- **Draft norm width** " + MASK_QUOTE
              + " `ckpt: weight = [4096]` Fix: widen the draft norm.\n")
    [finding] = ReviewEvidence.audit(report, ORCA).findings
    assert finding.status == UNVERIFIED
    assert "widen the draft norm" not in _fix_body(report, ORCA)


def test_untagged_phrase_disagreeing_with_config_keeps_a_defect_out_of_the_work_order():
    # The attention has no bias (`attention_bias: false`); an untagged phrase
    # saying otherwise cannot contradict, but it cannot verify either.
    report = ("### Defects\n- **Attention bias** " + MASK_QUOTE
              + " the projections use `attention_bias=true`. Fix: add q/k/v biases.\n")
    [finding] = ReviewEvidence.audit(report, ORCA).findings
    assert finding.status == UNVERIFIED
    assert "add q/k/v biases" not in _fix_body(report, ORCA)


def test_pass_needs_an_exact_row_status_and_a_table_s():
    row = f"| 1 | q/k projections | {HERON_QK_ROW} | q_proj | [sem] |"
    ok = ReviewEvidence.audit(TABLE_F + row + " match (no rotary) |\n" + TABLE_S, HERON)
    matched = ReviewEvidence.audit(TABLE_F + row + " matched |\n" + TABLE_S, HERON)
    no_s = ReviewEvidence.audit(TABLE_F + row + " match |\n", HERON)
    assert _final_verdict("pass", {"valid": True}, ok) == "pass"
    assert _final_verdict("pass", {"valid": True}, matched) == "error"
    assert _final_verdict("pass", {"valid": True}, no_s) == "error"


# -- a Defects section that says None, then carries notes -------------------
# No defects, then notes under a label that says they are not defects; every
# Table F row verified.
FINCH_DEFECTS_AND_NOTES = (
    "### Defects\n"
    "None. No row is MISMATCH, UNMAPPED or UNSOURCED. The deterministic validator "
    "returns `valid: True` with a storage diff of 0.\n"
    "\n"
    "Review notes (not defects):\n"
    "- The expert bank is declared `[E, 2F, D]` in `__init__` and stored `[E, D, 2F]`; "
    "the draft records that with `stored_axes: [1, 0]`.\n"
    "- The two halves of the fused projection are sliced without a start offset, as "
    "the shipped library FFNs do.\n"
    "- Table S was assembled from the spec and its components by hand.\n")

FINCH = SnapshotFacts({}, _header({}), {"modeling_finch.py": "\n".join([
    "class FinchExpertBank(nn.Module):",                                           # 1
    "        self.wi = nn.Parameter(torch.empty(self.n_experts, "
    "2 * self.ffn_dim, self.model_dim))",                                          # 2
])})
FINCH_PASS_TABLES = (
    TABLE_F + '| 1 | routed experts wi | `src: modeling_finch.py:2 '
    '"self.wi = nn.Parameter("` | experts.wi | [cost] | match |\n'
    "\n| component / node | op | forward step # |\n|---|---|---|\n"
    "| experts.wi | linear_param | 1 |\n\n")


def test_notes_inside_the_defects_section_are_defects():
    """The template puts notes under their own `### Notes` heading. A label
    inside the Defects section, whatever it says, does not end it; and
    `None.` followed by prose on its line is not the empty section."""
    ev = ReviewEvidence.audit(FINCH_PASS_TABLES + FINCH_DEFECTS_AND_NOTES, FINCH)
    assert len(ev.findings) == 3
    assert _final_verdict("pass", {"valid": True}, ev) == "error"


def test_a_defect_that_begins_with_none_is_a_defect():
    report = (FINCH_PASS_TABLES + "### Defects\n"
              "- None of the 48 layers applies the q norm "
              + REAL_DEFECT.removeprefix("- "))
    ev = ReviewEvidence.audit(report, FINCH)
    assert len(ev.findings) == 1
    assert _final_verdict("pass", {"valid": True}, ev) == "error"


def test_notes_section_after_defects_is_not_read_as_defects():
    report = (FINCH_PASS_TABLES + "### Defects\nNone.\n\n### Notes\n"
              "- Table S was assembled by hand.\n")
    ev = ReviewEvidence.audit(report, FINCH)
    assert ev.findings == []
    assert _final_verdict("pass", {"valid": True}, ev) == "pass"


# Only the next section heading ends the Defects section, and a PASS can
# carry no defect at all.
REAL_DEFECT = ('- **Expert half order** `src: modeling_finch.py:2 '
               '"self.wi = nn.Parameter("` Fix: swap the halves.\n')


def test_a_label_does_not_end_the_section():
    for label in ("Additional defects:", "**Important**",
                  "Review notes (not defects):", "#### Minor"):
        report = FINCH_PASS_TABLES + "### Defects\n" + label + "\n" + REAL_DEFECT
        ev = ReviewEvidence.audit(report, FINCH)
        assert [f.title for f in ev.findings] == ["Expert half order"]
        assert _final_verdict("pass", {"valid": True}, ev) == "error"


def test_content_after_none_blocks_a_pass():
    for tail in ("\nAdditional defects:\n" + REAL_DEFECT, REAL_DEFECT):
        report = FINCH_PASS_TABLES + "### Defects\nNone.\n" + tail
        ev = ReviewEvidence.audit(report, FINCH)
        assert _final_verdict("pass", {"valid": True}, ev) == "error"


def test_a_pass_listing_a_verified_defect_is_not_recorded():
    report = FINCH_PASS_TABLES + "### Defects\n" + REAL_DEFECT
    ev = ReviewEvidence.audit(report, FINCH)
    assert [f.status for f in ev.findings] == [VERIFIED]
    assert _final_verdict("pass", {"valid": True}, ev) == "error"
    # the same verified defect is Agent Fix's work order under a NO_PASS
    assert "Expert half order" in ev.work_order().split("### Unverified", 1)[0]


# -- the tag variants review agents write -----------------------------------
# Correct quotes cited in a variant of the tag form: `:L612`, `:L612-L612`,
# several quotes joined by `/`, `\"` inside a quote, or no backticks. Each
# must check the same as the canonical form.
_KESTREL_LINES = {
    612: "        self.q_tail_norm = KestrelPlainRMSNorm(eps=config.norm_eps)",
    640: "        q = self.q_tail_norm(q)",
    701: "        out = rotate(out.transpose(1, 2), cos, -sin).transpose(1, 2)",
    820: "        g = self.gate_proj(x).clamp(max=self.cap)",
    821: "        u = self.up_proj(x).clamp(min=-self.cap, max=self.cap)",
    866: "        g = g.clamp(max=self.cap)",
    867: "        u = u.clamp(min=-self.cap, max=self.cap)",
    1104: '    _skip_keys_on_load = [r"(^|\\.)draft\\..*"]',
}
KESTREL = SnapshotFacts(
    config={"clamp_limit": 6.0, "rope_dim": 64},
    header=_header({"layers.{}.mixer.pos_table": ([96, 256], range(24))}),
    sources={"modeling_kestrel.py": "\n".join(
        _KESTREL_LINES.get(i, "    pass") for i in range(1, 1300))},
)

# Backticked, `L` line numbers, `\"` inside the quote, `[library: ...]`
# prefixes and `**Repair:**`.
KESTREL_L_PREFIX = (
    "### Defects\n\n"
    "- [library: kestrel_moe_a.yaml / kestrel_moe_b.yaml] The shared expert runs the "
    "plain gated FFN with no clamp. The source has `src: modeling_kestrel.py:L820 "
    '"g = self.gate_proj(x).clamp(max=self.cap)"` and `src: modeling_kestrel.py:L821 '
    '"u = self.up_proj(x).clamp(min=-self.cap, max=self.cap)"`. **Repair:** give '
    "the shared expert the clamped FFN, its cap read from `config.clamp_limit`.\n"
    "- [sem] The draft group (`kestrel_draft_block`) is not in the forward. The "
    "source skips its weights on load: `src: modeling_kestrel.py:L1104 "
    '"_skip_keys_on_load = [r\\"(^|\\.)draft\\..*\\"]"`. **Repair:** note the '
    "draft group's reference implementation in a comment.\n"
)

# No backticks, `L..-L..` ranges, quotes joined by `/`.
KESTREL_BARE_RANGES = (
    "### Defects\n\n"
    "- **The draft attention is missing the plain per-head q norm (q_tail_norm)** — "
    'src: modeling_kestrel.py:L612-L612 "self.q_tail_norm = KestrelPlainRMSNorm('
    'eps=config.norm_eps)"; src: modeling_kestrel.py:L640-L640 "q = self.q_tail_norm(q)". '
    "Spec node: `components/drafts/kestrel_attention.yaml`. [sem] Fix: insert "
    "`q_tail_norm: rmsnorm_unscaled <- q_split`.\n"
    "- **[library: gated_ffn.yaml] The routed experts are missing the clamp** — "
    'src: modeling_kestrel.py:L866-L867 "g = g.clamp(max=self.cap)"'
    ' / "u = u.clamp(min=-self.cap, max=self.cap)"; cfg: config.clamp_limit = 6.0. '
    "[sem] Fix: use a clamped FFN variant with `max: config.clamp_limit`.\n"
)

# No backticks, digit line numbers; a Table F row and a defect.
KESTREL_BARE_ROW = TABLE_F + (
    "| 14 | Expert FFN with clamp_limit clamps | src: modeling_kestrel.py:820 "
    '"g = self.gate_proj(x).clamp(max=self.cap)"; src: modeling_kestrel.py:821 '
    '"u = self.up_proj(x).clamp(min=-self.cap, max=self.cap)"; cfg: clamp_limit = 6.0 '
    "| ffn.experts (gated_ffn) | [sem] | DEFECT (no clamp) |\n")
KESTREL_BARE_DEFECT = (
    "### Defects\n\n"
    "- **The attention is missing the inverse rotation on its output.** src: "
    'modeling_kestrel.py:701 "out = rotate(out.transpose'
    '(1, 2), cos, -sin).transpose(1, 2)". Spec node: `kestrel_attention` attn_apply '
    "→ o_merge. [sem]. Fix: add `o_inverse_rope: rope_emb_inverse`.\n"
)


def _statuses(report: str) -> list[str]:
    ev = ReviewEvidence.audit(report, KESTREL)
    return [f.status for f in ev.findings + ev.rows]


def test_l_prefixed_citations_with_correct_quotes_verify():
    assert _statuses(KESTREL_L_PREFIX) == [VERIFIED, VERIFIED]
    body = build_fix_task_body("acme/Kestrel-30B", "a: 1", [], {"valid": True},
                               ReviewEvidence.audit(KESTREL_L_PREFIX, KESTREL),
                               facts=KESTREL)
    work_order = body.split("### Work order", 1)[1].split("### Unverified", 1)[0]
    assert "shared expert runs the plain gated FFN" in work_order


def test_l_prefixed_citation_with_a_wrong_quote_is_contradicted():
    wrong = KESTREL_L_PREFIX.replace("u = self.up_proj(x).clamp(min=-self.cap, max=self.cap)",
                                     "u = self.up_proj(x).clamp(max=self.cap)")
    assert _statuses(wrong) == [CONTRADICTED, VERIFIED]


def test_unbackticked_tags_with_line_ranges_verify():
    assert _statuses(KESTREL_BARE_RANGES) == [VERIFIED, VERIFIED]


def test_unbackticked_tags_with_a_wrong_quote_or_value_do_not_verify():
    wrong_quote = KESTREL_BARE_RANGES.replace('"q = self.q_tail_norm(q)"', '"q = self.q_norm(q)"')
    assert _statuses(wrong_quote)[0] == CONTRADICTED
    # quotes joined by `/` are pieces that must appear in this order
    swapped = KESTREL_BARE_RANGES.replace(
        '"g = g.clamp(max=self.cap)" / "u = u.clamp(min=-self.cap, max=self.cap)"',
        '"u = u.clamp(min=-self.cap, max=self.cap)" / "g = g.clamp(max=self.cap)"')
    assert _statuses(swapped)[1] == UNVERIFIED
    wrong_value = KESTREL_BARE_RANGES.replace("config.clamp_limit = 6.0", "config.clamp_limit = 4.0")
    assert _statuses(wrong_value)[1] == CONTRADICTED


def test_unbackticked_row_and_defect_with_digit_citations_verify():
    assert _statuses(KESTREL_BARE_ROW) == [VERIFIED]
    assert _statuses(KESTREL_BARE_DEFECT) == [VERIFIED]


def test_unbackticked_row_and_defect_with_a_wrong_quote_are_contradicted():
    assert _statuses(KESTREL_BARE_ROW.replace("-self.cap, max", "0, max")) == [CONTRADICTED]
    assert _statuses(KESTREL_BARE_DEFECT.replace("cos, -sin", "cos, sin")) == [CONTRADICTED]


def test_unbackticked_count_tag_is_checked_like_a_backticked_one():
    # a spec template name tagged as if it were a checkpoint prefix
    report = KESTREL_BARE_DEFECT.replace("[sem].", "count: kestrel_mixer_block = 24. [sem].")
    assert _statuses(report) == [CONTRADICTED]
    report = KESTREL_BARE_DEFECT.replace("[sem].", "count: layers = 24. [sem].")
    assert _statuses(report) == [VERIFIED]


# -- a source cell beside its tag -------------------------------------------
# The source column holds a bare citation and the tags column a `src:` tag
# quoting those same lines.
WREN_SOURCE = "\n".join(
    ["    pass"] * 210
    + ["        mix = torch.sigmoid(mix_w * mix_scale + mix_b) + self.mix_eps"]      # 211
    + ["    pass"] * (1032 - 211)
    + ["        hidden = embeds.unsqueeze(2).expand("
       "-1, -1, self.config.n_streams, -1).contiguous()"])                         # 1033
WREN = SnapshotFacts({"text_config": {"n_streams": 4}}, _header({}),
                     {"modeling_wren.py": WREN_SOURCE})
WREN_EMBED_ROW = (
    "| 1 | embed_tokens | modeling_wren.py:1033 | text_model.embed_tokens | "
    'src: modeling_wren.py:1033 "hidden = embeds.unsqueeze(2).expand('
    '-1, -1, self.config.n_streams, -1).contiguous()" | match |\n')
WREN_EXPAND_ROW = (
    "| 2 | expand to n_streams copies | modeling_wren.py:1033 | stream expand node "
    "| cfg: text_config.n_streams = 4 | match |\n")
WREN_MIX_ROW = (
    "| 3 | stream mix before attention (plain norm, sigmoid mix + eps) "
    "| modeling_wren.py:207-223 | wren_block.mix_pre "
    '(+mix_w/scale/b) | src: modeling_wren.py:211 "mix = torch.sigmoid(mix_w * '
    'mix_scale + mix_b) + self.mix_eps" | match |\n')


def _wren_rows(*rows: str) -> list[str]:
    return [r.status for r in ReviewEvidence.audit(TABLE_F + "".join(rows), WREN).rows]


def test_a_bare_source_cell_is_the_citation_of_the_verified_tag_at_its_lines():
    assert _wren_rows(WREN_EMBED_ROW) == [VERIFIED]
    ev = ReviewEvidence.audit(TABLE_F + WREN_EMBED_ROW + TABLE_S, WREN)
    assert _final_verdict("pass", {"valid": True}, ev) == "pass"


def test_a_bare_source_cell_no_verified_tag_quotes_stays_unverified():
    assert _wren_rows(WREN_EXPAND_ROW) == [UNVERIFIED]
    # the tag at those lines quotes code the file does not have
    wrong = WREN_EMBED_ROW.replace("n_streams, -1", "n_streams, 1")
    assert _wren_rows(wrong) == [CONTRADICTED]


def test_a_bare_source_cell_citing_other_lines_than_the_tag_stays_unverified():
    # a range the tag quotes one line of
    assert _wren_rows(WREN_MIX_ROW) == [UNVERIFIED]
    # the same lines in another file
    other_file = WREN_EMBED_ROW.replace("| modeling_wren.py:1033 |", "| modeling_wrens.py:1033 |")
    assert _wren_rows(other_file) == [UNVERIFIED]
    # disjoint lines
    elsewhere = WREN_EMBED_ROW.replace("| modeling_wren.py:1033 |",
                                       "| modeling_wren.py:1032 |")
    assert _wren_rows(elsewhere) == [UNVERIFIED]


# -- Table F statuses written as marks --------------------------------------
# Every status written as `✅`; the source and config are cut down to the
# cited lines.
_PLOVER_LINES = {
    902: "        self.mixer_kind = config.mixer_kinds[layer_idx]",
    955: '    _skip_keys_on_load = [r"^draft.*"]',
    1010: "        if embeds is None:",
    1011: "            embeds = self.embed_tokens(input_ids)",
    1060: "        hidden = hidden.repeat(1, 1, self.config.stream_count)",
}
PLOVER = SnapshotFacts(
    config={"text_config": {"vocab_size": 100352, "stream_count": 4}},
    header=_header({
        "model.text_model.layers.{}.self_attn.q_proj.weight": ([8, 8], range(8)),
        "model.text_model.layers.{}.mixer.in_proj.weight": ([8, 8], range(8, 32)),
    }),
    sources={"modeling_plover.py": "\n".join(
        _PLOVER_LINES.get(i, "    pass") for i in range(1, 1100))},
)
PLOVER_ROWS = (
    '| 1 | Token embedding | src: modeling_plover.py:1010-1011 "embeds = '
    'self.embed_tokens(input_ids)" | model.text_model.embed_tokens | '
    "cfg: text_config.vocab_size = 100352 | match |\n"
    '| 6 | Widen stream to stream_count copies | src: modeling_plover.py:1060 '
    '"hidden = hidden.repeat(1, 1, self.config.stream_count)" | stream_expand | '
    "cfg: text_config.stream_count = 4 | match |\n"
    '| 25 | Draft weights skipped on load (no forward) | src: modeling_plover.py:955 '
    '"_skip_keys_on_load = [r\\"^draft.*\\"]" | draft group (aux, '
    "storage-only) | — | n/a |\n")
# `count:` naming spec templates, not a checkpoint prefix.
PLOVER_MIXER_KIND_ROW = (
    '| 7 | Mixer kind dispatch (recurrent vs full attention) | src: modeling_plover.py:902 '
    '"self.mixer_kind = config.mixer_kinds[layer_idx]" | per_layer.assign full/recurrent | '
    "count: plover_full_block = 8; count: plover_recurrent_block = 24 | match |\n")
PLOVER_TABLE_S = ("\n| component / node | op | forward step # |\n|---|---|---|\n"
                  "| model.text_model.embed_tokens | embedding | 1 |\n")


def test_rows_with_template_statuses_pass():
    ev = ReviewEvidence.audit(TABLE_F + PLOVER_ROWS + PLOVER_TABLE_S, PLOVER)
    assert [r.status for r in ev.rows] == [VERIFIED] * 3
    assert [r.canonical_status() for r in ev.rows] == ["match", "match", "n/a"]
    assert _final_verdict("pass", {"valid": True}, ev) == "pass"


def test_only_the_template_statuses_are_statuses():
    """A status is one the template names, in any letter case and with an
    optional reason; a mark or another word is no status, so a PASS cannot
    carry the row."""
    row = PLOVER_ROWS.splitlines()[0].removesuffix(" match |")
    for written, status in [("MATCH", "match"), ("**match**", "match"),
                            ("match (embedding)", "match"), ("Mismatch", "MISMATCH"),
                            ("✅", None), ("✓", None), ("matches", None), ("❌", None)]:
        ev = ReviewEvidence.audit(TABLE_F + f"{row} {written} |\n" + PLOVER_TABLE_S, PLOVER)
        assert ev.rows[0].status == VERIFIED
        assert ev.rows[0].allows_pass() is (status == "match"), written
        verdict = "pass" if status == "match" else "error"
        assert _final_verdict("pass", {"valid": True}, ev) == verdict, written


def test_a_match_does_not_verify_a_row_whose_count_names_a_spec_template():
    ev = ReviewEvidence.audit(
        TABLE_F + PLOVER_ROWS + PLOVER_MIXER_KIND_ROW + PLOVER_TABLE_S, PLOVER)
    assert ev.rows[3].allows_pass()
    assert ev.rows[3].status == CONTRADICTED
    assert _final_verdict("pass", {"valid": True}, ev) == "error"


def test_a_fact_the_run_did_not_capture_checks_nothing():
    """A failed config / tensor-index fetch is no evidence against a claim:
    the claim is unverified (a lead for Agent Fix), never contradicted."""
    facts = SnapshotFacts.from_run({"config": {}, "metadata": {}}, [])
    for kind, body in [("cfg", "text_config.sliding_window = 128"),
                       ("ckpt", "model.layers.X.mlp.gate_proj.weight = [2048, 6144]"),
                       ("count", "model.mtp.layers = 3")]:
        check = facts.check_tag(kind, body)
        assert check.status == UNVERIFIED and "cannot be checked" in check.detail
    assert SnapshotFacts.empty().check_tag("cfg", "rope_theta = 1").status == UNVERIFIED
