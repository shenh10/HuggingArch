# HuggingArch — contributor agent 协作准则

这份文件给在这个仓库**改代码**的 agent 看（包括后续 session、Codex、Kimi、未来加入的）。规则只放和「修改 backend 代码时容易踩雷」直接相关的部分。

- 写 model spec / 加 component yaml：**不是这份的入口**，去 §7。
- 和某个 `DESIGN.md` 冲突：**以 `DESIGN.md` 为准**，并提示用户更新这里。CLAUDE.md 是规则索引，不是真相主体。

怎么读：先记住 §1 的形态和原则，改走访 / 形状 / 代价 / 精度时对照 §2，加 op / scheme / role 走 §3，分层校验走 §4，命令和测试走 §5。

| 节 | 内容 |
|---|---|
| [§1 形态与原则](#1-形态与原则) | 一次走访、禁止第二条路径、禁止 adhoc、**出 bug 时的三连反问**、奥卡姆剃刀、先顶层设计、说人话 |
| [§2 硬不变量](#2-硬不变量) | 数值/符号、`in:`、tensors、quant、异构层、quant-agnostic、依赖方向、交换锚 |
| [§3 加东西](#3-加东西的-checklist) | op / quant scheme / GPU SKU / fusion plan / agent backend / role |
| [§4 校验分层](#4-校验流水线的责任划分) | schema → structural warning → bytes gate |
| [§5 工具与测试](#5-工具与测试) | uv、缓存、前端、Ruff、pytest、词汇 |
| [§6 Commit](#6-commit-规约) | trailer |
| [§7 Spec 指针](#7-写-spec--component-yaml) | authoring 入口 |

---

## 1. 形态与原则

### 1.1 一次走访，四层对象

HuggingArch 把模型分析建模成 **DSL spec + 一次求值走访**。YAML 结构只出口一次；形状和代价从这一次走访长出来。

```
YAML / BlockGraph          作者语法（编辑器写这里；schema 在这里拒）
        │  iter_tree       一次出口（spec_tree/build.py）
        ▼
   SpecIRTree              展开后的图：节点、边、role、zone、脊梁、stream 表、这次调用的 I/O
        │  Program.of      不复制图
        ▼
   Program                 走访日程：stack → stage、×N 层、这一跑的 pairing
        │  Evaluator.run   LayoutPropagation 是走访上的 Rule（切分不另造图）
        ▼
   Analysis                这一跑的形状、倍数、FLOPs、字节、KV、per-rank
        │  materialize_comm  按 placement 把集合通信嫁接到一份 Program 拷贝
        │  materialize_casts 把量化 GEMM 的激活 cast 嫁接成节点（comm 之后、融合之前）
        │  Framework.rewrite 融合（在 cast 之后；fused op 没有自己的公式）
        ▼
   Rewrite + 各层 rebase   同一份 Analysis，不是第二次求值
```

谁回答什么：

| 问 | 读 |
|---|---|
| 有哪些节点、怎么连、这次调用的接口、stream 表、层序 | `SpecIRTree` / `BlockMeta` |
| 走哪条 stack、stage 是 op 还是 block×count | `Program` |
| 这一跑的形状 / 代价 / 切分后的 per-rank | `Analysis` |
| 融合吸收了谁、通信节点在哪 | 走访之后的 `Program` 改写（`Rewrite`），数字仍从 Analysis rebase |

切分：走访里是 Rule；成图是 `materialize_comm`。融合：走访且在 comm 与 cast 之后，因为 fused kernel 的代价来自被吸收的那些 op，也因为吸收 cast 节点的是它。Inspector 画融合前的 Program；推理计算图画融合后的 Program（展开时挂上被吸收的子图）。计算图 L1 是盖在 L2 节点上的 view（`views/abstraction.py`），前端只画。

### 1.2 事实只从走访读

消费者（rendering / validate / capacity / playground 显示）都读上面这几层，不各自再推一遍。

**禁止第二条推导路径。** 不要再解析 YAML、再读 `dim_values` / 原始 `config` 去回答树上或 Analysis 上已有的量。不要静默兜底、不要按名字猜、不要加对账层。缺事实：先在 eval 里物化，再读。

任何代码改动先问两句：

1. 会不会让某个 consumer 看到和别人不一致的事实？（这个项目最常见的 bug 模式：两个 consumer 对「同一件事」读出不同答案。）
2. 这个数字是从那一次走访读的，还是又推了一遍？

### 1.3 修法必须通用

**禁止 adhoc 修复。** 任何开发和 bug fix 都必须考虑通用性。看到一个 bug 时不要顺手 patch 触发它的那个特定 case，先问：这个错误的语义类别是什么？同类错误在别处会不会也出现？正确的修法是改抽象边界还是改具体实现？

给 DSv4 加 `if model_id == "deepseek-v4-pro":` 这种 case-by-case 的代码是这个项目最坚决拒绝的反模式 —— spec 系统的整个价值就在于把「模型 diversity」通过抽象层吸收掉，adhoc patch 等于把 diversity 漏回 Python，每加一个就侵蚀一次 harness 的可信度。测试里给某个模型开「可以红」的名单是同一类：门看见了漏洞却放行，harness 同样不可信。红了就修图或修引擎，不要在名单里把它按绿。如果某个 case 真的特殊到无法通用建模，让它在 spec yaml 里通过 `quant_context.overrides` / inline `blocks:` / draft component 这些**已经存在的逃生通道**表达，而不是在 backend 代码里硬编码。

代理条件也是 adhoc：语料里一个模型踩了坑，就用「class 条数 > 1」「token 数碰巧相等」「walk 到的第一个兄弟」当开关，和 `if model_id` 是同一类 —— 把 case 漏回 Python，只是没写出模型名。该层已经有分组就用那个分组（KV 公式走 `growth_formula` 的叶子 Σ：相同则折 `n_layers`，不同则求和），不要另造一个开关。

**出 bug 时的三连反问。** 每修一个 bug，先按顺序答这三问，答案写进 commit message（或评审报告）。顺序是有意的：不先答出本质，第三问就会退化成「再加一道检查」。

1. **这个问题的本质是什么？** 不是「哪一行写错了」，而是**哪一个事实被答错了、它本该由谁拥有**。把事实命名出来（判例：DSv4 router 那个 bug 的本质不是「`refs.py` 硬写了 fp32」，而是「kernel 对某个 role 输出所要求的宽度，这个事实没有任何人声明过」；`d_head` 报成 5120 的本质不是「helper 取了第一个节点」，而是「一个 block 把 dim 绑成多少，这个事实没有主人，于是一个消费方自己发明了一个」）。本质答得对，第三问通常就只剩一句话。
2. **造成这个 bug 的本质原因是什么？** 区分**触发它的 case** 和**让它成为可能的原因**。原因必须能解释「一个称职的人为什么会写出这段代码」—— 如果答案读起来像「某人不小心」，那就还没答到原因。本仓最常见的真实原因只有几种：这个事实没有唯一的生产者；它被复述了好几遍；就地推出来比声明出来便宜得多（三行 vs 三百行），于是每次时间压力都把人推向就地推。
3. **结构上有哪些不合理，才让这个 bug 成为可能？** 这里的「结构」说的是**事实的归属与流向**：哪一层拥有这个事实、它在哪里生产、谁被允许读它、以及声明它和就地推它各要多少代价。所以这一问的答案必须是一个**架构上的改动**，形态就是下面这几种之一：

   | 架构毛病 | 改法 |
   |---|---|
   | 这个事实**没有主人** —— 任何一层都没声明过它 | 去**声明**它（schema / yaml / OpSpec 字段），而不是在消费方补一张表 |
   | 生产者**不发布**它，消费方只能自己再推一遍 | 在生产处**物化成字段**，消费方改成读 |
   | **两层都在回答同一个问题** | 定一层为权威，另一层改成读它；不要加对账层 |
   | 事实**到得太晚**（走访之后才拿到），于是只能在外面打补丁 | 把它变成**走访的输入**，在里面解析一次 |
   | 本该声明一次的东西被**摊成格子**（`op × class` 那 33 条） | 声明维度，不要平铺格子 |
   | **声明这条路比就地推贵得多**（三百行 vs 三行） | 把声明这条路修便宜 —— 没有「声明一次」的地方，就一定会有人手工挂第七次。这是架构问题，不是纪律问题 |

   答案也可以是「结构是对的，这是一次孤立的手误」，但那很罕见、要有证据。**不允许的答案是「再加一道检查」** —— 检查会越积越多，而生成 bug 的那个斜坡一点没变。加门和补 oracle 是修完之后的收尾动作，不是第 3 问的答案。

这三问背后是同一个形态：**一条实现路径 + 若干处复述**。只有一条路 ⇒ 任何检查都只能顺着它问、必然共享假设；又有多处复述 ⇒ 消费方互相矛盾。第 3 问要改的就是这个形态本身。

**收尾动作**（不是第 3 问的答案，答完架构之后再做）：把同类都找出来并点名；补一条 **oracle**，即用结构不同的第二条路把同一个量再算一遍然后对拍（判例：用 launch 追踪抓 aten 实际看到的 dtype 去对一行声明的精度；用 `node_dims` 去对 `block_frame`；从 checkpoint 的 config 重算边的字节）—— 数坏东西的计数器不是 oracle，它分不清「新来一处」和「上面插了一行导致行号位移」，把第一条路再跑一遍也不是；以及说清现有的门为什么没红（共享假设 / 量的是症状 / 红着没人跑 / **报告长着门的形状**：带 `regression` 标记却断言 `now <= floor`，二者选一个 —— **抬基线去吸收新条目等于不声不响把门变成报告，这个动作要拒绝**）。

### 1.4 奥卡姆剃刀

**如无必要，勿增实体。** 对两样东西同时生效：代码里的实体（类、枚举、遍历、抽象层），和话语里的实体（术语、代号、缩写）。

**概念引入前先过四问**（新概念、新词、新常量、新遍历）：

1. **已有的概念能不能表达？** 引入任何新类 / 新枚举 / 新 walker 之前，先找仓库里已有的同类抽象（遍历用 `spec_tree` / `eval/program.py` 的既有 walk，状态词汇用既有封闭词表，quant 走 `eval.formats` / `Analysis.tensors()`）。「已经有的概念不要再引入一遍」—— 两套词指同一件事，漂移只是时间问题（判例：决策 5 同一现象曾在两个模块分别叫 `INTENDED_NONDIVISIBLE` 和 `INTENDED_POLICY`）。
2. **这个区分在本层可观测吗？** SOL 粒度是最强剃刀：如果两个概念的差异在当前抽象层的输出上不可观测，它们就是一个概念（判例：route 状态机被砍 —— SOL 下不可观测）。
3. **它是维度还是格子？** 一批大写常量如果是几个正交维度的乘积展开，就声明维度、别平铺格子；review 循环里「每修一个 finding 添一个具名常量」是概念堆积的主要来源，回头重导最小基。
4. **它有文献名吗？** 有标准词就用标准词（placement、provenance、oracle）；确无文献覆盖才允许自述词，且必须登记进 `docs/parallelism.md` 末尾的术语对照表（Terminology）。冒用文献词但语义偏离，比自造词更糟。

一批围绕同一数据反复传参的 helper 函数 = 一个未命名的类；行为归领域对象的方法，不堆模块级 `_helper`。

**模块级大写名单常量 = 第二张表。** `_GEMM_OPS = {...}`、`_OP_TENSOR_SUFFIXES`、`_SLOT_ROLES`、`_DIRECT_WEIGHT_OPS`、`_STRICT_MODELS` 这类「一组 op 名 / role 名 / 后缀 / 模型名」的常量，八成是给某个 case 打的补丁：它复述的事实已经在 `OP_REGISTRY` 的字段、`roles.yaml`、`dims.yaml`、scheme yaml 或 `TensorDecl` 里声明过一遍，这里是第二遍，而两遍之间没有任何测试会因为漂移变红（2026-09-06 审计：全仓 139 个，48 个名字里就带 OPS/ROLE/SUFFIX/ALIAS/KNOWN；`_OP_TENSOR_SUFFIXES` 在补完 conv1d 的同一周就又和 `TensorDecl` 在 `per_head_linear` 上分叉了）。判据只有一条：**把这个常量删掉，能不能从已有声明派生出同一个集合？能，它就是补丁，改成问声明（`has_weight()` / `arity()==0` / `is_comm_op` / `tensors`）；不能，说明这个事实还没有声明，先去声明它，而不是在消费方列名单。** 允许保留的只有一种：它本身就是那个词表的唯一定义（如 `EDGE_KIND_VOCABULARY`），有 loader/validator 在装载期拿它硬 reject。见到别人的名单不要往里加名字 —— 加名字就是让这张表再活一轮，`per_head_linear` 就是这么漏的。

### 1.5 先顶层设计，再落代码

**写 feature 不要想着尽快打个洞把它串起来。** 一个新能力落地前，先回答它在这四层对象（§1.1）里属于谁、事实由谁声明一次、哪些消费方会读它；想不清楚就先把这几句写下来，再动代码。打洞串起来的代价不在当下——每条通路单看都能工作，合起来却没有任何一处说明这件事的规则是什么，于是第二条、第三条通路各自重新发明一遍，而它们之间没有测试会因为漂移变红。

**判例（2026-10-31，跨卡定价）**：「一次走访能不能挪到另一张卡上重新计价」这条规则，历史上长出了三条通路——preparation 搬（`_restamp_gpu`）、rows 搬（`_point_rows` → `batch_sweep.rows_at`）、step form 搬（`ClassSweep.on_card`）。规则只在第一条上写了，于是 Scaling 面板把 6 张卡里的 5 张按 RTX-4090（cc 8.9）的行集合定价：Kimi-K3 的 decode 在 Ada 上是 1092 行、在 H100/B200/B300 上是 1012 行（`fused_kda_decode` 门控在 cc 9.0），方向一律偏乐观。更隐蔽的是 DeepSeek-V4-Flash：B200 prefill 沿用 H100 走访时行数 270 对 274，而第 22 行的 routed activation 本该是 FP8 却按 BF16 计——**行数相同的时候它就已经错了**。

从这个判例要记住三件事：

1. **规则要有唯一入口，不是让每个消费方「记得去问」。** 三个搬运者各自调用同一个拒绝函数，仍然是三次「记得」；正确形状是让卡进不来，除非经过那道门（搬运者收一个只能由「这次走访 + 同类的卡」铸出来的值，而不是收 `gpu_sku: str`）——第四条通路那样就连写都写不出来。
2. **这类错不崩、不空、只是数字偏乐观**，所以只比数字的测试抓不到：同一个 mutation 下 Kimi-K3 变红，而 Qwen3-30B-A3B 和 Llama-3.2-3B 全绿，因为它们在每个卡类上部署同一张图、行集合本来就一样。**动按卡定价的改动，必须比 per-op row 的集合和顺序（含 dtype），不能只比 `t_sol` 或行数。**
3. **预言过的缺陷要当场修，不要记成观察项。** `_point_rows` 这条路在出事前一小时就被一份评审写成「第二个读者，今天只是构造性安全，因为跨卡重定价只在同一 walk class 内发生」——那个前提当天就被证伪。评审说「今天安全是构造性的」时，那不是观察项，那是待修项。

### 1.6 说人话

面向人的输出 —— 会话汇报、设计文档的导言、commit message 的第一段 —— 先用平实的语言讲清楚「在干什么、为什么、做到哪了」，一个没跟进过程的人读一遍就能懂。内部代号（S0 / 增量号 / 自造缩写）和术语只在定义过之后使用；一段话里术语密到读者需要词汇表，等于没写。判据：向不了解本仓库的工程师复述一遍，对方能不能听懂。术语本身不是罪 —— 文献标准词该用就用 —— 罪的是用代号替代解释。

---

## 2. 硬不变量

§1 是原则；下面是改代码时不要破的、可检查的契约。

| | 不变量 | 一句话 |
|---|---|---|
| 2.1 | 数值和符号同一次走访 | 不要并行 symbolic 实现，不要 `isinstance(..., sympy.Symbol)` |
| 2.2 | `in:` 是 dataflow edge | 改走访时不要当可选字段 |
| 2.3 | `Analysis.tensors()` 是权重字节唯一 join | 不要在别处自己算 quant bytes |
| 2.4 | `quant_context` 是 runtime 唯一事实源 | serving 路径不解析 `config.quantization_config` |
| 2.5 | 异构层不要展平 | 叶子 Σ 相同也不是 hybrid |
| 2.6 | spec 语言 quant-agnostic | component yaml 里不能出现 quant sibling 名 |
| 2.7 | 依赖单向；eval 是唯一生产者 | 不要 lazy import 藏环；下游不自己造事实 |
| 2.8 | 出口交换锚的是区域 | 不要要求恰好一个叶子 |

### 2.1 数值和符号是同一次走访、两套绑定

`Evaluator` 对同一份 `Program` 跑一遍；`Binding` 决定每个 dim 是 int 还是 `sympy` 表达式。**不要写并行的 symbolic 实现**，**不要 `if isinstance(x, sympy.Symbol):` 分支** —— Expr 进去出来还是 Expr，int 进去出来还是 int，自然分流。符号域走 `Binding.symbolic()`，不是另一个 engine 子类。

### 2.2 `in:` 是真正的 dataflow edge

Evaluator 顺着每个节点的 `in:` 把上游已经传播的形状喂给当前 op。`in:` 不是注释、不是只给 UI 渲染用、不是只给生成 spec 的 LLM 看 —— 改走访时不要把它当可选字段。跨 stage、跨 stack 的 residual threading 也不要破：后一层吃前一层的 `shape_out`，aux stack 的 `inputs:` 点名它读的是哪一段。

### 2.3 `Analysis.tensors()` 是权重字节的唯一 join

`storage` ⋈ `formats` ⋈ `instances` 只在这里拉链。**不要在别处自己算 quant bytes** —— 一旦 quant 计算分散在多个角落，混精模型立刻让两边对不上。需要 on-disk 清单（含 format 暗示的 scale sibling）时走 `Analysis.iter_checkpoint()`。

### 2.4 Spec 的 `quant_context` 是 runtime 唯一事实源

Spec 生成阶段 agent 把 `config.quantization_config` 一次性翻译成结构化 `per_role + overrides + default_dtype`。**Serving 路径不解析 `config.quantization_config`** —— 这是「两份 quant 解读」经典坑（前端按 config 显示一种 dtype、validator 按 tensor 名字看到另一种）。部署精度只有一个入口：精度档（`arch/eval/precision.py` 的 `PrecisionProfile`，请求字段 `precision`）。它在走访时解析成 `ResolvedPrecision`（挂在 `Analysis.precision` 上）；权重、边、缓存槽的出厂值都读声明（`quant_context`、`config.torch_dtype`、缓存项的 `dtype:` / `format:`），每项实际值与出厂值由 `Analysis.precision_effective()` 给出（`/precision/effective`），不读 config 的 `quantization_config`。估算路径的权重 dtype 从 `Analysis.tensors()` 取，缺了就 raise 而不是默认 BF16。

### 2.5 异构层不要展平；同构也不要打扮成 hybrid

混合模型的总 FLOPs 是各 assignment 的和式（`Ld·MLA + (L − Ld)·MHA`），不是打成一个公式。`Program.contributions` 按模板计数保留这个维度；改聚合时不要图简洁把它打平。

反过来也对：**叶子 Σ 相同的多段 assignment 不是两种注意力**。DSv3 的 3 层 dense FFN + 58 层 MoE，注意力都是 MLA，KV 公式是一条 `n_layers × (d_kv_lora + d_qk_rope) × bpe`，代入写成 `(3 + 58) × …`。不要把徽章写成 `MLA × 3 + MLA × 58`，也不要用 `len(classes) > 1` 当 hybrid。Hybrid 是不同的 cache family / 不同的叶子 Σ（GQA+SWA）。展示层的「原始公式」读 `growth_formula`，assignment 拆分只出现在公式的代入里。

### 2.6 spec 语言是 quant-agnostic 的

Component yaml / inline `blocks:` 里**不能出现 quant sibling tensor 名**（`*_scale` / `qweight` / `qzeros` / `_packed` / `weight_scale_inv` / …）。Schema validator 会硬 reject。这些是格式层的产物，不是结构事实。

### 2.7 依赖方向是单向的，不靠 lazy import 藏环

- `arch/eval/` 核心（analysis / capacity / evaluator / fusion / cache_view / op_ref …）**不得 import `arch.layout`、`inference`、`rendering`**。per-rank 投影是 rule 契约的一部分：基类 `eval/rule.py` 的 `Rule` 携带恒等答案（local == global），`LayoutPropagation` 覆写 `local_tensors` / `local_instances` / `resident_weights` —— **不要用 `getattr(rule, …, None)` 探针**，那是 `test_eval_boundary` 的 ratchet 会拦的 defensive access。fusion 注册表由调用方注入（`Fusion.named(name, schemes)` / `Framework.of(binding, schemes)` / `fusion_matches(..., schemes)`）。
- 唯一例外：`eval/transforms.py` 是组合根（`Deployment.analyze` 装配 walk + layout rule），只允许它 import layout。
- `rendering` 可读 `arch` 和 `inference` 的事实；`arch` 和 `inference` 都不得 import `rendering`。
- 修 bug 时不要靠函数级 lazy import 掩盖反向依赖 —— 那是信号，说明符号放错了层：下沉事实、调用方编排，或加规则钩子。
- 以上三条由 `tests/meta/test_eval_boundary.py::test_dependency_direction_holds` 以 AST 扫描硬断言（函数级 lazy import 同样计入），越界即红。
- **反方向也有门：eval 是事实的唯一生产者，eval 之外不许自己造事实。** `inference` / `rendering` / `api` / `service` / `kernel_bench` 只消费 `Analysis` 上的类型化层（cost / shapes / traffic / storage / instances / formats …），**不**问 `OP_REGISTRY`、不按 `node.op == "…"` 分支、不读 `dim_values` 表推结构、不读原始 `config`、不保留「eval 没给答案时」的备胎（`placement is None → 按比例摊`）、不按 id 子串判类别。下游需要一个 `Analysis` 上没有的事实，先在 eval 里**物化成字段**，再消费。2026-09-06 量到 86 处（`weight_breakdown` 14、`rendering/builders` 13、`diagrams` 11、`analyzer` 9）；记分板 `downstream_derivation` 跟踪这类地方的数量 —— 它按模式匹配，是报告不是门（`regression` 才跑）：新增条目评审时用 `python -m tools.scoreboard --list downstream_derivation` 看，判断是真问题还是误报。上一场 eval 重构验收的是「eval 不往外看」，缺的正是这条「外面不许自己算」—— 事实账本 66 处复述有 60 处在 eval 外面，就是这么长出来的。

### 2.8 出口交换锚的是区域，不是唯一叶子

`layout/materializer.consumer_of`：INTERNAL_OUT 是一次交换。一个读者 → 锚它；wrapper 已列出这条边 → 锚 wrapper；同一 parent 下的兄弟读者（packed SwiGLU：`permute` → `gate_proj` 和 `up_proj`）→ 锚 parent，再 walk 改写每一个读口。跨两个 parent 仍 fail。不要要求「恰好一个叶子」—— 那是把一种图（expert 包成子组件）当成全部图。Golden 用 `ep_moe=1` 测不到这条；Gallery 的 EP=8 会打到。

---

## 3. 加东西的 checklist

### 加 primitive op（`backend/arch/ops.py`）

- 用 `register_op(OpSpec(...))` 注册（普通函数调用，不是装饰器）
- **必须声明 `shape_kind`**（`preserve` / `broadcast` / `axis_replace` / `axis_select` / `axis_concat` / `contract` / `permute` / `source` / `explicit`）—— 引擎编成 DimMap 处理形状，不要给单个 op 写自定义 shape solver
- `params_fn` / `flops_fn` 返回 sympy `Expr`（接受 Symbol 也接受 int）
- 多输出 op 走 `secondary_outputs={port: (dim, …)}` 模板，下游用 `<node>.<port>` 引用
- 测试按行为归类：多输出模板加 `tests/backend/arch/test_multi_output_ops.py`，params/FLOPs 断言走 `tests/backend/arch/eval/test_eval_flops.py` 这类 eval 测试

### 加 quant scheme（`backend/arch/quant_schemes/<name>.yaml`）

- 用结构化字段声明：`format_variants` / `metadata_siblings` / `numeric_format_resolver` / `scale_strategy` / `required_fields` / `default_fields` / `also_stores`（config 字段声明同一 checkpoint 还带另一 scheme 的 sidecar，例如 DSv4 `scale_fmt: ue8m0` → `fp8_ue8m0` 的 `.scale`）
- **不写 Python** —— 现有 10 个 scheme（gptq / awq / fp8 / fp8_ue8m0 / fp4 / mxfp4 / mxfp4-pack-quantized / compressed-tensors / modelopt / nvfp4）都是纯 yaml，新方案应当延续这条线。部署时可选用的方案声明 `deployment_fields`（补齐部署格式的配置），提供激活选择的再声明 `activation_options`（选项 → `arch/activation_formats/` 里的激活格式，或 `none`）
- 配套测试加在 `tests/backend/arch/quant_schemes/test_quant_schemes.py` 或 `tests/backend/arch/eval/test_eval_formats.py`

### 加 GPU SKU

- `backend/inference/gpu_specs.csv` 加一行（HBM 容量、HBM 带宽、peak BF16/FP8 TFLOPs，scale-up 链路的单向带宽 `scaleup_gbps` 和它的名字 `scaleup_fabric`（`NVLink` / `HCCS` / `UB`，二者同有同无），`device`：框架 dispatch 判断的平台，如 `cuda`，framework binding 的 `when: device` 按它判定；以及 `compute_capability`：CUDA 计算能力，如 `9.0` / `10.0`，binding 的 `when: min_compute_capability`（阈值，如 `"10.0"`）按它判定框架选哪个 kernel（如只在 SM100 及以上启用的 kernel）；`arch` 是代际名，不代替二者；非 CUDA 设备（如 `npu`）留空 `compute_capability`，并确认有一个 framework binding 的 `devices:` 列出这个设备）。一行是一个 rank：多 die 封装（Ascend 910C）按 die 写。无需改 Python。

### 加 fusion plan / framework binding

- plan（`backend/inference/fusion_plans/*.yaml`）写被替换的朴素子图；binding（`backend/inference/framework_bindings/{vllm,sglang,deepinfer}.yaml`）写哪个框架真的跑这个 kernel，`source_anchor` 和注释里的 file:line 必须在 `audited_at` 那个版本的源码里打开 kernel 和调用点核过，prefill 和 decode 分开核。`source_anchor` 里每处引用写成仓库根路径 `<path>[:<line>[-<line>]]`，后端据 `audited_at.repo` + `commit` 生成 GitHub 永久链接；loader 拒绝裸 `:N` 和不带目录的文件名，有本地 clone 时 `test_fusion_registry` 核对每处文件和行号在审计 commit 上存在。
- **一个 fused node 就是一次 launch**：框架里是两三次 launch 的，拆成每次 launch 一个 plan，不要一个节点顶多次。
- **分派条件的粒度见 [`docs/inference_cost_model.md`](docs/inference_cost_model.md) 的 Fusion 一节**（那里是真相主体）：只建模会改字节或主路径、且常见部署下就会触发的（phase、TP/world 整除、GPU 代际、框架、是否投机解码）；只在边缘情况改 launch 数的和部署开关一律按默认服务路径计价，在注释里写明假设，不为它们新增条件键，也不因它们撤 binding。判据是 §1.4 第二问：SOL 输出上看不出的区分不建模。激活尾段（`silu_and_mul` 一族）的输出精度跟着消费它的 GEMM：plan 把 `quantise_activation` cast 列为可选成员，一条规则，不按框架区分。
- 区分靠图上的事实或框架自己的分派键（`architectures` 等），不靠模型名或代理条件；框架分派依赖的事实图上没有，先声明它（如 conv1d 的 `dilation`），不要猜。
- **审计版本里的框架 bug 不挡正确的 plan**：路径因框架 bug 走不到时，把 binding 升级到修好的版本（更新 `audited_at`，全部条目在新版本上重核），不要撤 plan。

### 加 agent backend

- 实现 `backend/agents/base.py` 的接口，新建 `backend/agents/<name>.py`
- 注册到 `spec_worker` 的 backend 选择路径

### 加新 role

- 在 [`backend/arch/roles.yaml`](backend/arch/roles.yaml) 登记：`kind`（attention / ffn / norm / wrapper）+ `meaning`。每个 role 或 slot 名在那里被分类一次，loader 硬 reject 注册表不认识的名字（`loader.py` 的 unclassified-role 检查）
- component 声明的 `role:`（`BlockGraph.role`）和 model `components:` map 的 slot 名、调用点覆写（`GraphNode.role`）由 `backend/arch/schema/graph.py` 的 validator 对照同一份 `roles.yaml` 校验，没有第二张白名单
- 同步更新 `quant_schemes` 的 per_role 消费方
- 同步更新 `backend/arch/components/README.md` 的 role 表格 + leaf-inheritance 规则
- **不要新加 role 来表达「另一种 attention」** —— 那是 component 的事，不是 role 的事

### 加 model spec / component yaml

去 §7 看指针。CLAUDE.md 不重复 spec authoring 规则。

---

## 4. 校验流水线的责任划分

不要让一层去做另一层的事：

| Layer | 文件 | 职责 | 失败方式 |
|-------|------|------|----------|
| 1 | `arch/schema/` | 引用解析、forward DAG、role 白名单、quant-suffix lockdown、CustomOpSpec parity | `raise ValueError` 硬 reject |
| 2 | `arch/validate/_structural.py` | forward-source hints（branch / role 覆盖 / dead-end） | warning 列表，**自身永不 fail** |
| 3 | `arch/validate/_matching.py` + `_orchestrator.py` | bytes-vs-bytes diff + unmatched HF tensor；shape / type-contract / cross-layer findings 也在 orchestrator 汇入 gate | `errors == []` 才能 ship（gate 明细见 `docs/validation.md` §3 "The publish gates"） |

具体注意：

- **Layer 1 不挡新拓扑**：inline 一个新 `blocks:` 是合法的，不要为了「统一」reject 这种 case。
- **Layer 2 的 warning 永不 fail**：forward-source hints 反馈给 agent loop，不进 gate。注意边界已收窄：shape consistency（`_shapes.py`）、cache 声明（`_type_contracts.py`）、跨层 stream（`_cross_layer_channels.py`）的 findings 已由 `_orchestrator.py` 提升为 publish gate 的 errors。把某类 finding 升级成 error 时，不要在检查模块里顺手 raise —— 必须走 orchestrator 的 gate 汇总（`errors.extend`），且先证明该检查 FP-free 才能升级。
- **Layer 3 用 bytes 不用 numel**：混精 checkpoint 里只有物理字节单位一致；logical numel diff 仅作辅助诊断。

---

## 5. 工具与测试

### 5.1 命令

**用 `uv`**：`uv sync` / `uv run pytest` / `uv run python -m backend.api`。不要直接 `pip install`、不要 bare `python`。本地 2053 服务用 `ha restart`（不要另起一个 uvicorn 抢端口）。

### 5.2 API 结果缓存

`backend/measure/cache.py` 是唯一实现（`KeyedCache` + `engine_version` / spec 指纹）。Gallery / sweep / supported-models 都走它。新 endpoint 不要手写 dict+lock。预热在 server 进程外跑：`uv run python -m backend.warm_caches`（`--landing` 只热首页默认视图，`--sweeps` 热 Scaling 面板的全部下拉组合：parallel + Context×Batch；低优先级 + 自带小 sweep pool，写同一套缓存层），server 进程内不预热。Docker 里缓存目录必须在 Fly volume（`/app/.cache`）**外面**，否则挂载会盖掉镜像里烤好的文件。

### 5.3 前端

前端是 `web/`（Vite + React + TypeScript）。`npm ci --prefix web && npm run build --prefix web`（`tsc --noEmit && vite build`）产出 `web/dist`，FastAPI 从那里提供所有页面，打包产物是 `/web-assets/<name>-<hash>.js`（内容哈希即缓存戳）。`web/dist` 不提交：Docker 首阶段构建它，`tests/frontend/test_web_build.py` 在 CI 里构建它，类型错或构建失败就是测试红。依赖版本锁在 `web/package-lock.json`。跨页共享的库由 `vite.config.ts` 的 `manualChunks` 拆成 vendor chunk；新增大型依赖（图表 / 布局类）必须进 `vite.config.ts` 的 `manualChunks` vendor chunk 或随路由 chunk 懒加载（elkjs 即后者），不进首屏 landing bundle；线上靠 `GZipMiddleware` 压体积。

**前端是 ESM**：`web/src` 全是 TypeScript ES module（`export` 是公开面，不导出即私有；不要用下划线表达私有）。构建首次需要 `npm ci --prefix web`。跨模块引用一律走 `import`；`tsc` 在构建期拒绝引用未导出的名字。模块不往 `window` / `globalThis` 挂东西（`/site-config.js` 提供的 `window.SITE_CONFIG` 除外）。

客户端代码不要 fall back 到 `http://localhost:PORT`，默认 same-origin / 相对路径。

### 5.4 Ruff

**Ruff 抓 bug 不抓风格**：gate 开 correctness 家族（`F, B, PLE, PLW, SIM, PERF, PIE, RUF`），纯审美族（`E / W / I / N / UP / COM`）刻意关闭。选用理由 + 本仓刻意约定的 ignore（中文注释 `RUF001-003`、FastAPI `Depends` 白名单等）都写在 `pyproject.toml` 的注释里；改规则前先读那段，别顺手加审美类。

**门的范围是整个仓库**：三个 workflow 一律 `ruff check .`，不是 `ruff check backend/`。曾经只查 `backend/`，于是 `kernel_bench/`（生产代码）、`tools/`、`scripts/`、`tests/` 里的 18 条无人看见——包括 `rms_normalise` 定义了两遍、smoke 的两个 dim 用不同值声明了两遍（后一条静默胜出）。1215 个文件约 60 ms，没有理由再收窄。某条规则真的和某个目录的惯用法冲突时，写 `[tool.ruff.lint.per-file-ignores]` 并注明理由，**不要把目录排除在检查之外**——没人查的目录就是上次漏掉的那个。

### 5.5 测试

- **测试目录镜像源码目录**：`backend/arch/eval/` 的测试在 `tests/backend/arch/eval/`，`kernel_bench/` 的在 `tests/kernel_bench/`，`web/` 的在 `tests/frontend/`。**改了 `<pkg>/` 就跑 `tests/<pkg>/`**，不用 grep 也不用记。唯一没有源码对应物的是 `tests/meta/` —— 那几个测的是仓库自身的规则（结构债、eval/service 边界、README 数字对不对）。`tests/` 自己是包（有 `__init__.py`），所以 `tests/backend/` 不会遮蔽真的 `backend`；共享 helper 在 `tests/snapshots.py` / `tests/corpus_row.py`（`from tests.snapshots import …`）。测试文件仍自带 `sys.path` 引导，可以单独 `python tests/backend/arch/eval/test_eval_flops.py` 跑。模块设计文档跟代码同目录（`backend/arch/DESIGN.md`、`backend/inference/DESIGN.md`）。
- **`regression` 默认不跑**：逐模型检查（走遍全部 snapshot 或一整批模型的测试）和分钟级的文件（`tests/kernel_bench/test_kernel_bench.py`）标 `regression`，`pyproject.toml` 的 `addopts` 把它摘掉了，推送 / PR 的 CI 和部署都不跑。改了会动到所有模型的引擎代码、或改 `kernel_bench/` 时用 `-m regression`（或 `-m regression -k <名字>`）显式跑；线上由手动触发的 `regression` workflow 跑。新写的逐模型测试要标 `regression`。**参数化测试默认只跑第一个用例**：`tests/conftest.py` 在收集期给参数下标不全为 0 的用例打 `regression`（`tests/meta` 的门禁除外），所以默认集是 smoke、全部组合在 `-m regression` 里；排在第一个的必须是那个在缺陷复发时会**红**的用例（判别力优先，同等判别力再取最便宜的）—— 默认集和部署门只看得见它，排错了套件照样绿、默认集却不再守任何东西；要守一对性质（拒这个**且**放那个）就把最可能退化的一半排第一，或拆成两个测试。在函数体里自己循环扫一批模型 / 组合的测试改成 `parametrize`，钩子才管得到。
- **评审不跑测试**：代码评审 / 质量评价类任务只做静态阅读，不要 `uv run pytest` 实跑 —— 套件里存在全 corpus 扫描型测试（如 `test_eval_axis_kinds.py` 单文件 160s），逐文件起 pytest 进程会把评审 loop 卡死（2026-08-30 评审 session 即因此卡死一小时）。
- **多会话并行时不各自跑大套件**：同一台机器上同时有多个会话在改代码时，每个会话只跑和自己改动直接相关的少量测试（改了哪个文件跑哪个测试文件，不加 `-n 16` 这类大并发）；golden / corpus 扫描 / 全目录这类大套件不要每个会话各跑一遍，等各自的改动合进 main 之后，多个 commit 攒在一起由一个会话统一起一次。2026-10-07 十几个会话各自 `-n 12` / `-n 16` 跑大套件，机器 load 长期 25+，前端构建从 19s 拖到 2 分钟、单个估算从 6s 拖到 60s，测试互相拖到超时，等于谁都没跑成。

### 5.6 词汇

概念词表 [`backend/GLOSSARY.md`](backend/GLOSSARY.md) 只收**不可约的领域名词**（身份三元组、mesh 布局、注意力分类 canonical）；操作一律用白话动词、不进词表。维度符号见注册表 [`backend/arch/dims.yaml`](backend/arch/dims.yaml)（每个符号在那里被分类一次：`kind` 决定哪种并行切它，`meaning` 是 tooltip 和 agent 读的那句话）。

---

## 6. Commit 规约

- **不带 `Co-Authored-By: Claude` trailer**。

---

## 7. 写 spec / component yaml

CLAUDE.md 不复制 spec authoring 规则，只指路：

- [`backend/arch/components/README.md`](backend/arch/components/README.md) —— library / adapters / drafts 三层准入、role 表格、encapsulation parity
- [`backend/spec_worker/_prompts/`](backend/spec_worker/_prompts/) —— spec 生成 system / schema / task prompt（`templates/system.md` + `templates/schema.md`）
- [`backend/arch/DESIGN.md`](backend/arch/DESIGN.md) —— spec 系统设计取舍 + Anti-patterns 列表
- [`backend/inference/DESIGN.md`](backend/inference/DESIGN.md) —— parallelism + capacity 模型
