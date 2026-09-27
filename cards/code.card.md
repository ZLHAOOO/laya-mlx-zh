---
card: code
version: 2
description: 代码工作场景：bug/功能/重构/文档/部署路由 + 生产问题判断
author: Fuyao
---

研发协作里的消息分流。输入是开发群里/同事发来的技术相关消息，输出该交给哪个研发技能。

边界说明：user 报 bug 但其实是需求的情况，按描述主体判断（"点了没反应"=bugfix，"要是能导出就好了"=feature）。

## Questions

### route （choice）
这条消息该路由给哪个代码类技能？

- bugfix: 排查修复：报错、崩溃、异常、行为不对
- feature: 新功能开发：加功能、改交互、提需求
- refactor: 重构优化：不改行为只改质量，瘦身、改名
- docs: 文档查询：API 用法、配置说明、示例
- deploy: 部署运维：构建、发布、环境、证书
- chat: 与代码无关的消息

### urgent （noul）
这条消息是否涉及必须马上处理的生产问题？
