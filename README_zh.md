# memory-gate（中文版）

> 英文 [README.md](README.md) 为准。本文件同步翻译，如有出入以英文为准。

**给 LangChain 的上下文压缩加一道 fail-closed 的人工闸门。**

`SummarizationMiddleware` 会用一段 LLM 写的摘要替换掉你的对话历史。它认为不重要的
东西就此离开 state——不报错、不留日志、不能撤销。`memory-gate` 在"即将被摘要"和
"已经被摘要"之间放一个人。

```
pip install git+https://github.com/2xIarge/memory-gate.git
```

还没上 PyPI——现在 `pip install memory-gate` 会 404。这个名字暂时没人占，发布排在后面；
在那之前请从 git 装，或者 clone 下来 `pip install -e .`。

---

## 问题出在哪

压缩是有损的，而且**丢得很安静**。这不是假设，是 LangChain issue 区里正在发生的事：

> "agent 只能从**一段有损的 LLM 摘要**里去推断本该直接读到的状态。这在长任务里造成
> **计划漂移和重复规划**。"
> — [langchain#36624](https://github.com/langchain-ai/langchain/issues/36624)，至今开放

> "被删除的消息可能是**某个决定、指令或推理链的唯一记录**……调用方**看不到任何数据
> 已被销毁的信号**。"
> — [langchain#38867](https://github.com/langchain-ai/langchain/issues/38867)（这个具体
> 故障模式已修复，但这句话指出的问题形状没有）

结构化状态（todo、计划、配置）可以靠每轮从 `state` 重新注入来对压缩免疫。
**其他东西不行。** 用户在第 7 轮用大白话提过一次的那个约束——"我们财年从 4 月 1
号开始"——没有 schema，只活在对话里。摘要把它丢掉的那一刻，什么都没报错，agent
只是从此把 Q1 理解成一月到三月。

`memory-gate` 保护的就是这一类信息：**只以对话形式存在的长期约束**。

## 看效果

`examples/fiscal_year_demo.py` 全程离线，用假模型，不需要 API key。

```
# PART A —— 只有 SummarizationMiddleware
>>> 'fiscal year starts April' present anywhere in the request: False
>>> 没有任何异常，没有任何日志。agent 接下来会把 Q1 当成 1—3 月。

# PART B —— 前面加上 MemoryGateMiddleware
====================================================================
memory-gate 中断了本次运行：
====================================================================
压缩即将丢掉 6 条消息。其中 4 条是你可以保留的对话（57 tokens）。
其余的只以有损摘要的形式存在。没有任何东西被删除——全部留在归档里。

 [   2]*   29 tok  By the way, our fiscal year starts April 1st, so Q1 is Apr-Jun. All pri…
 [   1]    14 tok  Analyse our quarterly revenue for 2026.
 [   3]     7 tok  run query 0
 [   4]     7 tok  run query 1

 * 命中了保护规则：4 条里有 1 条像是长期约束。
   但这些规则会漏掉一部分真约束，所以没有星号的那一行并不等于可以丢。

 另有 2 条消息同样会被丢掉且没有列出（工具结果，以及除非放宽 pin_roles 才会出现的助手回复）。

  keep all     在预算内全部保留   <- 通常就该这么答
  keep 1,2,3  只保留这几个编号
  confirm      直接压缩，不新增保留

你 -> 'keep 2'   （钦定那条财年约束）

>>> 消息窗口中是否还有该约束: False   （压缩确实把它删了）
>>> 最终请求中是否存在:      True    （闸门注入回来的）
>>> 确实是闸门救回来的:      True
磁盘归档: {'archived': 16, 'archived_tokens': 317, 'protected': 1, 'protected_tokens': 29}
```

注意最后三行。那条消息**确实被删掉了**——它已经不在消息窗口里。它重新出现在模型面前，
唯一的原因是闸门把它放了回去。

## 上手

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from memory_gate import MemoryGateMiddleware

summarizer = SummarizationMiddleware(model="openai:gpt-4o-mini",
                                     trigger=("tokens", 24000), keep=("messages", 6))

agent = create_agent(
    model,
    tools=[run_sql, export_chart],
    middleware=[
        MemoryGateMiddleware(               # <- 必须放第一个
            summarization=summarizer,
            archive_dir=".memory-gate",
            protect_patterns=(r"fiscal year", r"do not touch staging"),
        ),
        summarizer,
    ],
    checkpointer=checkpointer,              # 必需：interrupt() 依赖它
)
```

顺序不是风格问题。`before_model` 会各自变成一个图节点、按列表顺序串起来；
`wrap_model_call` 的组合方向是"列表第一个 = 最外层"。排第一，意味着闸门能在压缩
删掉任何消息之前先看到它，也意味着它能在真正调模型之前说最后一句话。这条是对
**已安装版本**实测的，不是照抄文档。

`protect_patterns` 是**替换**默认规则集，不是往里加。想保留实测过的那套再补自己的，
写 `DEFAULT_PROTECT_PATTERNS + (r"你的标记",)`。

## 工作机制

**1. 先归档，无条件。** 任何落到 keep 窗口之外的消息，在任何审阅或压缩发生之前就被
追加写进 `archive.jsonl`。幂等，不需要许可。多归档只花磁盘，少归档是永久丢失。

**2. 只在压缩确实迫在眉睫、且丢失看起来代价高时才问。** 闸门读的是被守护中间件自己的
`trigger`，不是猜。不对齐的后果很实际：早期版本在 16 个回合里打断了 **13 次**，这种
东西装不上第二天就会被关掉。对齐之后同一条轨迹只打断 2 次。另外，同一份候选集绝不
问你第二遍。

**3. 钦定之后，每轮重新注入。** 被保留的原文写进 `keep.jsonl`，并在每次模型调用时合并
进 system message，所以后面几轮压缩再也拿不走它。

## 实测结果

真模型（llama.cpp 跑的本地 Qwen），57 轮会话，在 40 个无关话题中间植入 10 条约束，
压缩触发 2—3 次。最后逐条问模型：

| | 只有压缩器 | 加闸门 |
|---|---|---|
| 存活约束 | **8/10** | **10/10** |
| 压缩轮次 | 2 | 3 |
| 打断人的次数 | 0 | 3 |
| 每次调用注入的 token | 0 | 151 |

**引用这张表之前，先看完下面两条限定。它们说的都是"怎么测的"，不是机制行不行。**

1. **带闸门那一跑是拿着标准答案测的。** 那次传的 `protect_patterns` 是**由植入约束自己的
   关键词**（`财年`、`net`、`_v3`、`Lena`…）拼出来的，再加 4 个标记词；而每次审阅的自动
   回答是 `keep <被标记的编号>`。也就是说那轮的分诊是个 oracle，被标记的东西**就是**判分
   答案。所以它不能证明产品能找对行——它证明的是：**只要一条内容被钉住，它就能活过三轮
   压缩并原文回到模型面前**。这个结论是真的，而且能出 10/10 说明归档幂等、预算、注入这条
   链路全部真的在工作。
2. **默认值之后改过。** 那次用的是 300 token 预算和不截断的清单；现在
   `protect_budget_tokens` 是 trigger 的比例，`render_review_text` 会截断显示。而当时库
   里的 `DEFAULT_PROTECT_PATTERNS` 还是只有祈使句标记的那一版，就是下一节打分的对象。

所以 8/10 对 10/10 说明的是**失败的形状**和**钉住确实能修**，不是审阅判断力的分数。
真正能补上后半句的实验是：`scripts/verify_real_model.py` 用默认 patterns、`--policy
"keep all"`、不带任何关键词 oracle。**这一跑还没做。**

下面这几条限定条件跟表格一样重要：

- **每种配置只跑了一次（n = 1）。** 一条轨迹、一个模型、一套话题。它说明机制在真模型
  上能跑通，不是基准测试。
- **对照组坏得很安静。** 丢掉那两条时，模型没有说"我不知道"，而是给了一个看起来很合理
  的错误财年和错误的数据库环境。
- **丢不丢，很大程度取决于压缩有多激进。** 把 `keep` 放宽，对照组也能拿 10/10。所以如果
  配置由你掌控，放宽压缩比装闸门更便宜。这个包服务的是你**掌控不了**的场景：上下文预算
  紧张、共享 agent、工具密集的长会话。
- **一轮压缩一次审阅，不是反复追问。** 审阅次数与压缩轮次严格一致，57 轮里只问了 3 次。

真正决定结果的有两条，都不是闸门本身聪明：

- `pin_roles` 默认只有 `("human",)`。助手会用自己的话复述你的规则，把复述也钉住等于
  花同一份预算存重复文本，实测能装下的不同约束数量直接少一半。
- 注入预算必须远小于 trigger。钉住的文本每次调用都重发、都被模型重新计数，所以预算一旦
  接近 trigger，就会重新触发它刚从里面活过来的那次压缩。实测：2 轮压缩变成 31 轮。因此
  `protect_budget_tokens` 的默认值改为 trigger 的 2%（夹在 [200, 4000]），而不是一个定值；
  显式传入超过 trigger 10% 的值会告警。

## 哪些行会被打上星号

清单里的 `*` 来自 `DEFAULT_PROTECT_PATTERNS`。它的准确率是产品属性，因为这是审阅人
唯一拿到的分诊信号。由 `scripts/score_patterns.py` 在**规则没有照着它调过**的语料上打分：

| 规则集 | held-out 召回 | held-out 精确 | 真实存档里被标记的占比 |
|---|---|---|---|
| 只有祈使句标记（最初的） | 29% / 58% | 67% / 88% | 16% |
| 当前默认 | 93% / 83% | 93% / 100% | 17% |

差距是结构性的，不是调参能补的：最初那 25 个模式全是祈使标记（`must`、`never`、`必须`、
`一律`），而人们表达约定时，**陈述事实**远比下命令常见——"我们财年从 4 月 1 号起"
"对接人是 Lena"。这个修复的第一版是照着同一份被测轨迹调出来的：在那份轨迹上看着好了
一倍，在 held-out 上反而更差。所以语料必须分开调，那个变差的对照组也留在脚本里。

规则集有天花板。`周报发给 Kevin，抄送整个组` 是一条没有任何标记词的长期约束，正则够不到
它。所以 `keep all` 才是默认动作，星号只是阅读辅助，绝不当筛选器用。

## 清单会有多长

`scripts/render_at_scale.py` 用真实渲染器画了一遍生产规模（200K 窗口、80% 触发）的清单。
结论是清单长度取决于**用户说了多少轮**，而不是 token 量——实测每条用户发言平均 9.4 token：

| 负载画像 | 填满 160K 需要多少轮 | 渲染行数 |
|---|---|---|
| 重工具 agent | 10 轮 | 22 行 |
| 轻工具 | 48 轮 | 53 行 |
| 纯长对话 | 390 轮 | 54 行（截断前是 403 行） |

不截断的长对话版本是 403 行 / 16,497 字符，没人读得完。所以 `render_review_text` 只画
40 行——星号优先、其次从最老的开始——但 payload 里保留全部候选。这个区分很要紧：如果
改成按**钉住预算**截断，被截掉的正好是最老的消息，而"会话开头说过一次"的约束就活在那里。

## 保证

每一条失败路径的设计目标都是：**闸门坏掉时必须看得见，绝不能安静地不存在。**

| 情况 | 行为 |
|---|---|
| 归档写不进去 | 抛异常。本轮不压缩。 |
| 回复解析不出来 | **全部保留**。不猜。 |
| 既没传 `summarization` 也没传 `trigger` | 构造时 `ValueError`。永远不会落下的闸门比没有闸门更糟。 |
| resume 期间候选集变了 | 抛异常。拒绝压缩一份你没批准过的清单。 |
| 候选集与刚问过的那份完全相同 | 跳过追问。决定已在磁盘上。 |
| 压缩摘要出现在清单里 | 按 `lc_source=summarization` 排除。绝不问你要不要保留一段转述。 |

最后一行比看上去重要。问一个人"你要不要保留这段 LLM 写的摘要"，是一个没有好答案的
问题。

## 它不是什么

- **不是压缩算法。** 它不把摘要写得更好，它只决定什么不该被摘要，然后让原来的中间件
  继续干活。
- **不省 token。** 保留是花 token 的。这里买的是正确性。
- **拦不住 state 更新。** 压缩照样重写历史，闸门是从磁盘归档里把你钦定的东西放回去。
  没被钦定的内容能从 `archive.jsonl` 找回来，但不会自动回来。
- **不管工具结果。** 工具输出可再生、且占据大部分 token，所以只归档、不上清单。这是必要
  条件但不是充分条件：长对话里光用户自己的发言就测到 403 行，所以渲染端还要截断。
- **不是通用记忆系统。** 跨会话的长期知识请用 claude-mem 那类方案。这个包只守一个会话
  里的一种特定操作。

## 事后恢复

```python
gate.search("April")        # 在所有归档过的内容里做子串搜索
gate.restore("ac5b5611-…")  # 把某条被忘记的消息提升进永久保留区
gate.archive.stats()        # {'archived': 16, 'protected': 1, ...}
```

"丢失之前判断什么重要"是最难的部分，而且人一定会判断错。每次闸门执行都会写进
`runs.jsonl`，所以事后能审计：当时给出过什么清单、你批准了什么、什么被没问过就丢了。

## 依赖

Python >= 3.10，`langchain >= 1.0`，`langgraph >= 1.0`，以及一个 checkpointer
（`interrupt()` 需要）。开发与验证基于 `langchain 1.4.1` / `langgraph 1.2.11`。

它依赖"`before_model` 是独立图节点"这一事实。正是这一点让 `interrupt()` 落在
`ModelRetryMiddleware` / `ModelFallbackMiddleware` 够不着的地方——那两个中间件历史上
吞过 interrupt（[langchain#38837](https://github.com/langchain-ai/langchain/issues/38837)）。
如果这个组合方式变了，本包也要跟着变，所以测试里专门有一条守着它。

## 自测

```
python scripts/verify.py           # 47 条分阶段断言，全程离线
python scripts/score_patterns.py   # 星号到底能抓住什么
python scripts/render_at_scale.py  # 200K 窗口下的清单长什么样
python examples/fiscal_year_demo.py
python -m pytest                   # 50 条测试
```

## 许可

Apache-2.0。
