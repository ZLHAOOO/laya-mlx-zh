---
card: emotion
version: 2
description: 情绪理解场景：是否带情绪/是否需要安抚/是否情绪危机（三问独立判断）
author: Fuyao
---

情绪感知卡。输入是任何一条人际消息，三个问题**独立判断**（一条消息可同时触发多个）。

危机判断（urgent）是护栏级判断：宁可保守（倾向 true）也别漏。建议与上游大模型联动：
urgent=true 时无条件升级人工/大模型复判。

## Questions

### emo （noul）
这条消息是否带有明显情绪（喜、怒、哀、焦虑）？

### care （noul）
这条消息是否需要安抚或共情回应？

### urgent （noul）
这条消息是否涉及需要立刻处理的情绪危机？
