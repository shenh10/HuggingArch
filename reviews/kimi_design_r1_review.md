# 架构审 — `docs/walk_identity.md`(bf49e373d)

评审范围:Stage A 与 §5 那条定理。Stage B 按简报要求不表态。
已核对的代码:`sol_data.py`、`card.py`、`capacity.py`(267/418/626/723/1180/2217/3687)、
`framework_bindings/__init__.py`(DispatchCondition/CardClass/装载期两道门)、
`eval/fusion.py:1420-1613`(GraphCondition/admits_match)、`eval/transforms.py`(Workload/
walked_as_number/tp_divides)、`step_eval.py`(Cell/StepCells)、`sol_law.py:1445`、`serving.py:360`、
`pipeline.py:114`(leaf_peak)、sglang/vllm yaml 被引条目、`docs/symbolic_evaluation.md`、
`docs/parallelism.md` Terminology。

## 1. 归属:方向对,点名的类不对

事实确实有唯一主人,而且每条轴的主人都在:**workload** 轴的声明在 binding yaml 的
`dims:` 条件里(GraphCondition,fusion.py:1440),消费方硬编码在 `Workload.walked_as_number`
(transforms.py:226);**card** 轴的声明在 `DispatchCondition.CARD_KEYS` + `PEAK_COLUMNS`,
由 `_check_dispatch_vocabulary` / `_check_card_fact_classification` 两道装载期门守着
(framework_bindings/__init__.py:755/856);**degree** 轴在 `dims.yaml` kind +
layout 规则。方案说「问题都已声明、所以可派生」——这句是真的,而且是这个方案最结实的地方。

但方案 §6 把 `ServingWalkFacts` 定为「整个 static 侧的主人」,这里有两处不实:

- **归因错误。** card class 今天**不在** `ServingWalkFacts` 里。它在 `ServingWalkKey.card_class`
  (sol_data.py:206-216),degree 在 `StepLawKey.layout_json`(:247)。`ServingWalkFacts`
  是且仅是 workload anchor——它的 docstring「The workload facts a canonical serving walk is
  taken at」是精确陈述,不是可以被「makes that sentence true of all three axes」顺手扩写的
  占位符。三条轴的 static 侧今天是摊在三个类上的,方案说的「让 docstring 成真」实际是
  一次所有权的搬家,方案没有承认这一点。
- **类选错。** `ServingWalkFacts` 装的是 **stated** 事实(producer 选的 14 个字段,
  `from_record` 严格字段表,schema 漂移即拒);答案集合是 **derived** 事实(声明变它就变)。
  把三种轴的派生答案塞进这个类,它就正是简报担心的「什么都往里装的类」,而且会污染它
  现有的良好纪律(「Floors … deliberately absent」)。正确形状:答案集合是一个**独立的
  派生值对象**,挂在 `ServingWalkKey`(card_class 的现主人)旁边/替掉它,序列化进 anchor;
  派生逻辑留在 eval / framework_bindings,不进 sol_data.py。sol_data 这个**模块**是对
  的层(它本来就是「the one place that names those facts」),方案点名的**类**不对。

## 2. 概念数:净增一个实体,可以接受,但要承认

核了 §1.4 四问:

1. **已有概念不够。** `CardClass` 是 per-card、与图无关的粗粒度;`walked_as_number` 是一对
   硬编码;`ContextAnchors`(capacity.py:2217)是同 stage 内共享走访的规则,不是身份。
   「这次走访的 static 侧的规范形」这个粒度今天不存在,所以答案集合是**一个新实体**——
   方案自称「不引入新概念」不成立。但它的账是平的:它**替掉** `walked_as_number` 的
   硬编码、把 `CardClass` 降级为预筛子(两者都从 `card_dispatch_facts` 同一份声明派生,
   无漂移面),净概念数不增。方案应该把这句实话(「新增一个派生值对象,收回两处旧拼写」)
   写出来,而不是说「没有新概念」。
2. **可观测性通过。** 答案不同 ⇒ 行集合/法形/拒绝不同,SOL 粒度上可观测(§1 的实测
   正是按行集合比的)。
3. **是维度不是格子。** 见 Q3。
4. **文献名:方案一个都没用。** 「binding-time analysis / partial evaluation」全文未出现,
   所以**不存在冒用**。但 §1.4 第四问是双向的:这条规则(static/dynamic 二分、
   「不能延后的问题的答案决定走访」)就是教科书 binding-time analysis 的 division
   congruence——「两次运行在全部 static 输入上一致 ⇒ 残余程序同一」。命名它反而白捡
   定理的已知形状。现状是不引用文献、自造「question / answer set」又不登记
   (`docs/parallelism.md:311` Terminology 表里没有),两头都差一点。

## 3. 「答案集合」不会长成第二张表——除了一个没闭环的口子

派生不需要清单,这在结构上成立:问题的**词表**是封闭的(`DispatchCondition.KEYS` +
GraphCondition 字段,装载期拒未分类键),问题的**实例**在已装载的 yaml 数据里,答案由
五个已有谓词给出。没有任何一步需要模块级 Python 名单。

**但 §3.1 的派生规则按字面执行会过推导。** 规则说「the counts any declared dims condition
reads, at the values it reads them at」。sglang:530 恰恰声明了 `dims: {B: [1], …}`——
按字面,B 也该被 walk 成数字 1,而一个 B=1 的走访不再是「every point's walk」,这摧毁
整个 symbolic 设计。方案看到了这个 case,但给的出路是「becomes a declaration rather
than a comment」——**那个 declaration 住在哪、默认是什么,方案没说**。而这正是关键:
n_verify=1 可派生、B=1 不可派生,区分它的事实(「这个 count 是可以离散步进的离散轴」
vs「这个 count 是走访存在的意义所在的 sweep 轴」)今天只在 `walked_as_number` 的 Python
里。如果实现时靠「默认保持符号、手工放行 n_verify」来挡 B,那就**恰恰在方案自己的
核心卖点上长出了一张隐式 denylist**——和 `CardClass` 同一个毛病。这个声明必须有家
(`dims.yaml` 的 kind 或 binding 层),否则「不列清单」的主张在唯一一个现存反例上就是假的。

另注意一个诚实的代价:派生是双向的。明天有 kernel 声明 `dims: {n_verify: [2..8]}`
(GraphCondition docstring :1449 已经预告了这个 case),派生就意味着那 7 个宽度各走
一次——走访数变多。这是语义强制的、对的,但 §3.1 应该把这句写出来。

## 4. Stage A / Stage B 切分:诚实

- **Stage A 独立成立。** 它的收益不依赖 Stage B:(a) 给「dispatch 条件读的 count 必须
  走成数字」这条无主人的不变量找到主人;(b) card 轴折叠(Llama/Qwen 在 A100/4090/H200
  上答案一致 ⇒ 3 捆并为 1;DeepSeek-V3 因 fp8 列缺席正确地不并);(c) oracle 本身。
  反过来 Stage B 依赖 Stage A 的 oracle 先绿——依赖方向是单向的,不是「只有都做了才有
  意义」的自欺。
- **§7 第 2 点是真的正确性收益,不是给性能找正当性。** 我核了:这个失败模式不是臆造,
  `GraphCondition.dims` 的 docstring(fusion.py:1447-1450)已经把「kernel 为 2-8 个
  verify token 编译、于是列出这些宽度」写成了既定语义,而 `walked_as_number` 对此毫无
  响应——静默不绑定、无测试红。给这条不变量找主人是 §1.3 三连反问第三问表里「这个事实
  没有主人 ⇒ 去声明它」的标准形态。
- **一个提示(不算方案的错):** §1 的第一条实测证据(37 layouts → 3 结构)是 **degree
  轴**的折叠,那是 Stage B 的收益;Stage A 自己的证据是第二条(card 轴)。验收口径里
  13,335 → 2,739–5,478 若含 layout 折叠,则不是 Stage A 单独能交付的。数字本身归另一位
  评审核,但「Stage A 值不值」应按 card 折叠 + 正确性主人来判,不应按 degree 折叠的大头
  来判。按前者,仍值(下节)。

**收益是否足以 justify 这次架构改动(验收口径 2):是。** 即便只取下界 2.4×,它砍的是
6–7 小时冷缓存里的主导成本,且是「每个模型都有 Scaling 面板」(今天在请求死线内不可
能)的唯一通路——§8 说的「逐走访微优化八次共 1.8×」与这个是不同量级。再加一条无主人
不变量的正确性收益。架构改动的体量(换一个 key、删一处硬编码、加一个 oracle)与收益
相称,且有 §5 的门挡着最坏情况。

## 5. 代理条件:不是代理,但「定理」这个词超支了

「答案相同所以合并」**不是** §1.3 的代理条件:代理是「这俩碰巧一样」(外延比较输出),
这里是「在全部已声明问题上答案一致」(内涵比较,问题来自声明)。方向对。

但 §5 坚持「It must be a theorem, not an observation」——**在 Python 里它无法被静态
证明**。走访代码任何地方都可以新写一个 `if card.spec.compute_capability >= …`,不经过
那五个谓词。它实际是三样东西叠出来的不变量:

1. **封闭词表 + 装载期门**(新 `when:` 键、新 sheet 列不加分类就拒)——保证新「问题」
   不能从已声明通道里溜进来;
2. **每次走访的运行时硬拒**(recorded ⊆ computed,§5 末段)——保证每次走访自带证明
   义务,不依赖语料;
3. **全语料对拍**——只是**可行性校验**(证明硬拒今天不会在合法走访上误伤),不是
   正确性载体。

这样拆开后它是站得住的:正确性由 1+2 扛,语料对拍是 3。但方案把 3 写成了定理的证明
(「run the oracle over the whole corpus … It must be a theorem」),这是用词超支——
靠全语料实测「验证」的东西按定义不是定理,方案的诚实版本是「每次走访的运行时门 +
装载期的封闭性」。这话必须改,因为它决定后人往哪加防御。

**录制面有一个具体漏洞。** 方案 §4 列的录制点是五个谓词调用点(holds_for / admits_match /
tp_divides / materialize_comm / stage_of)。已发现表外读取:`serving.py:360-371` 在
launch 选择里直接读 `framework_binding.deployment.peak_families`;`LAUNCH_KEYS`
(`unquantized_bf16_launch`)在 `point_keys` 里供给。如果录制只 tap 那五个谓词,这些
通道的答案两边(前置计算 & 走访录制)会**共享同一个遗漏**,对拍永远拍不出来——oracle
变成「把第一条路再跑一遍」(§1.3 明确说那不是 oracle)。录制面必须按「事实读取」
而不是「谓词调用」来定义,或先证明这五个点是全部读取的唯一必经路。

**§2.3「answerable before the walk」对 GraphCondition 家族不成立。** `admits_match`
(fusion.py:1547)读的是 `Analysis.node_dims`、`Rule.world_degree`、traffic 的读者格式——
全是**走访的产物**。前置计算 = 在 spec 树上重写一条不经走访的匹配/解析路。这恰恰是
oracle 需要的「结构不同的第二条路」,方向正确,但它是**新写的推导代码**,不是「问题已
声明所以免费可得」。方案把这笔工作量抹平了,Stage A 第一步(oracle)的真实成本主要在
这里。

## 6. 判例对照:「唯一那道门」在字面上不成立,在意图上成立

`priced_on` **不是**唯一的门。今天直接问 `Card.refuse_unless_one_class` 的至少四处:
`capacity.py:631`(`_restamp_gpu`)、`capacity.py:1215`(逐 phase 搬 rows 的路)、
`capacity.py:3763`(`priced_on`)、`sol_law.py:1456`(`cardBinding`,law 图进卡的
门)。§1.5 判例里的第三条通路 `ClassSweep.on_card` 已不存在(没有命中),`cardBinding`
是它的后继。

补救方案的是:四处调的是**同一个谓词**,谓词只陈述一次。Stage A 换的是谓词,所以四个
门一起换——「第四条通路写出更弱的比较」这个具体失败形态(class 比真相粗)不会复发,
因为任何新通路要比较就得先算出答案集合,而算它就得问那些问题。**残余风险**只剩一种:
拿两张不同图的答案集合互比——这要 §2 性质 2 说的「只能由问过问题的对方铸出来的值」
(类型级不可伪造)才能关死,而 Stage A 明确「keeps the gate」、不动 `_restamp_gpu(p,
card)` 这类签名,所以不可伪造性在 Stage A **没有交付**,方案 §2 性质 2 把终态性质说成了
本阶段的性质。另外 §3 说「that is asked in one place already — `priced_on`」与代码不符
(见上四处),这句话必须改成真实的 census——改完方案反而更稳(谓词单点陈述才是它能
一把换掉的真正原因)。

## 7. 三连反问:`s_verify == 1`

1. **本质。** 方案答到了:「哪一个事实没有唯一主人」——「dispatch 条件需要 n_verify 以
   数字身份可读」这个事实,yaml 声明了需求侧,Python 硬编码了供给侧(transforms.py:234
   vs 四处 yaml 条目),两遍拼写无测试会因漂移变红。方案 §3.1「一个 bit 被写成一对
   (name, value)」的表述略浅(本质不是 bit vs pair 的表示问题,是**供需两侧各拼写
   一遍**),但同一段紧接着自己说出了正确答案(「The Python pair is the second spelling
   of it」),算答到。
2. **原因。** 在消费方补三行比在声明侧派生便宜——§1.3 列的第三种经典原因,方案隐含
   答对。
3. **结构改法。** 「从声明派生,不在 Python 命名」是对的方向,但如 Q3 所述,派生规则
   在 `B` 上没有闭环——**第三问的答案目前不完整**:它声明了 n_verify 的供给侧,却没有
   给「B 永远符号」一个声明的家,而这两者的区分才是这个问题的真正本质(可离散步进的
   离散轴 vs 走访存在所系的 sweep 轴)。方案把本质答成了「bit vs pair」,于是第三问
   退半步成了「把那对名字删掉」——删掉之后 B 这个 case 由什么挡住,没有答案。

## 8. 固定末条

- **复述已有事实:** 方案复述了 §1.5 判例与 CardClass 语义,但用于动机且**点名了它要
  替换的既有文本**(symbolic_evaluation.md §1 的那句、CardClass docstring、sol_data 的
  记录身份)——这正是防止第二真相持有者的纪律,合格。开头「not a second truth holder」
  的声明也正确。
- **静默兜底:** 无。§5 明确「hard refusal, not a warning」。
- **按名字嗅探:** 无。合并依据是声明派生的答案,不读模型名/类数/碰巧相等。
- **与 CLAUDE.md 相悖:** 两处软冲突,均已计入必改——(a) 自造词「answer set /
  question」未登记 Terminology(§1.4 第四问);(b)「定理靠全语料实测验证」的措辞与
  §1.3 对 oracle 的定义相抵(把第一条路的假设再跑一遍不算第二条路,除非录制面按事实
  读取定义——见 Q5)。

## 验收口径

1. **「不引入新概念」:不成立,但可修复为诚实。** 它引入一个实体(答案集合 = static
   侧的规范形派生值),与 `CardClass` 的 key 角色重叠——处理方式是降级 CardClass 为
   预筛子并收回 `walked_as_number` 硬编码,净概念数不增。方案须自己承认这一点。
2. **收益 justify 改动:是**(见 Q4 末段)。

## 签

**签**,附 3 处必改:

1. **归属修正(方案 §3 / §6):** 答案集合不进 `ServingWalkFacts`——它是 stated
   workload anchor + 严格字段表(sol_data.py:51-127),塞派生事实会让它成为什么都装的
   类。改为独立派生值对象挂在 `ServingWalkKey`(sol_data.py:206,card_class 的现主人)
   层级,派生逻辑留在 eval / framework_bindings;并删去 §6「its docstring already
   says … makes that sentence true of all three axes」的错误归因(card class 今天不在
   那个类里)。
2. **派生规则闭环(方案 §3.1):** 「counts any declared dims condition reads」按字面
   会把 sglang:530 的 `dims: {B: [1]}` 也推成静态。必须给出区分「可走成数字的离散轴」
   与「永远符号的 sweep 轴」的声明的家(dims.yaml 或 binding 层)及默认行为;否则
   「不列清单」在现存唯一反例上就是一个隐式 denylist,方案的核心卖点(Q3)不成立。
3. **门与通道的 census + 定理措辞(方案 §3 / §4 / §5):** (a) 「single gate =
   `priced_on`」改为真实清单:`refuse_unless_one_class` 今天被 capacity.py:631、
   capacity.py:1215、capacity.py:3763、sol_law.py:1456 四处直接调用,换谓词全覆盖,
   且 §2 性质 2 的不可伪造性 Stage A 并不交付,不得写成本阶段性质;(b) oracle 录制面
   补全 §2 表之外的读取点(serving.py:360 的 `peak_families`、`LAUNCH_KEYS`、
   GraphCondition 的 `world_local`/`readers_eat`),并在方案中认领「GraphCondition 的
   答案今天读的是走访产物(admits_match 读 Analysis.node_dims),前置计算是新写的第
   二条推导路」这笔工作量;(c) §5 的 theorem 改写为其实际形态——封闭词表(装载期门)
   + 每次走访 recorded⊆computed 硬拒,语料对拍是可行性校验而非证明——并把
   question / answer set 登记进 `docs/parallelism.md` Terminology(或正名为
   binding-time analysis 的 division)。
