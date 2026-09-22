# memory-gate（中文版）

> 英文 [README.md](README.md) 为准。本文件同步翻译，如有出入以英文为准。

**给 LangChain 的上下文压缩加一道 fail-closed 的人工闸门。**

`SummarizationMiddleware` 会用一段 LLM 写的摘要替换掉你的对话历史。它认为不重要的
东西就此离开 state——不报错、不留日志、不能撤销。`memory-gate` 在"即将被摘要"和
"已经被摘要"之间放一个人。

```
pip install memory-gate
```

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
即将压缩 6 条消息（79 tokens 对话内容）。
压缩后它们只会以有损摘要的形式存在。回复编号可永久保留：

 [ 1] 14 tok  human  Analyse our quarterly revenue for 2026.
 [ 2] 16 tok  ai     Sure. Which fiscal calendar should I use?
 [ 3] 29 tok  human  By the way, our fiscal year starts April 1s…
 [ 4]  7 tok  human  run query 0

  keep 1,3,5   保留指定编号
  keep all     全部保留
  confirm      直接压缩

你 -> 'keep 3'

>>> 消息窗口中是否还有该约束: False   （压缩确实把它删了）
>>> 最终请求中是否存在:      True    （闸门注入回来的）
>>> 确实是闸门救回来的:      True
磁盘归档: {'archived': 16, 'protected': 1}
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

## 工作机制

**1. 先归档，无条件。** 任何落到 keep 窗口之外的消息，在任何审阅或压缩发生之前就被
追加写进 `archive.jsonl`。幂等，不需要许可。多归档只花磁盘，少归档是永久丢失。

**2. 只在压缩确实迫在眉睫、且丢失看起来代价高时才问。** 闸门读的是被守护中间件自己的
`trigger`，不是猜。不对齐的后果很实际：早期版本在 16 个回合里打断了 **13 次**，这种
东西装不上第二天就会被关掉。对齐之后同一条轨迹只打断 2 次。另外，同一份候选集绝不
问你第二遍。

**3. 钦定之后，每轮重新注入。** 被保留的原文写进 `keep.jsonl`，并在每次模型调用时合并
进 system message，所以后面几轮压缩再也拿不走它。

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
- **不管工具结果。** 工具输出可再生、且占据大部分 token，所以只归档、不上清单。这样
  审阅列表才短到真有人读。
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
python scripts/verify.py      # 45 条分阶段断言，全程离线
python examples/fiscal_year_demo.py
```

## 许可

Apache-2.0。
