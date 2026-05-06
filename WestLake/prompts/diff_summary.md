# 版本998 Prompt差异分析报告

> 分析对象：`iteration_reports/version_998/` 目录下的 prompt 记录文件\
> 生成时间：2026-05-04\
> 文件范围：1.txt \~ 20.txt（共20个文件，按时间倒序排列，1为最新）

---

## 📋 文件基本信息

| 属性 | 值 |
| --- | --- |
| 目录路径 | `iteration_reports/version_998/` |
| 文件数量 | 20个 (1.txt \~ 20.txt) |
| 单文件大小 | 约130KB-150KB (约1.8k-2.2k行) |
| 内容类型 | DeepScientist System Prompt + Runtime Context + User Requirements |
| 时间跨度 | 2026-05-04 04:06 \~ 13:16 (约9小时) |

---

## ✅ 核心共性（所有文件相同，约90%+内容）

### 1. System Prompt主体（完全一致）

```
## Style First
- 结论→含义→下一步的沟通规范
- 中文自然表达、避免内部黑话
- 简明里程碑式汇报

## Hard execution redlines
- 禁止原生shell_command，必须使用bash_exec
- 所有终端操作走bash_exec路径

## Mission / Core execution stance
- 长期研究任务导向，非单次问答
- 通过持久化文件和artifact保持连续性

## Tool contract (memory/artifact/bash_exec)
- 三工具命名空间规则
- artifact.interact交互协议
- bash_exec执行纪律

## Metric & comparison discipline
- 保持baseline比较契约
- 指标记录规范

## Canonical research graph
- scout → baseline → idea → experiment → analysis → write → finalize
- decision可随时介入路由
```

### 2. 固定配置项（完全一致）

```
ds_home, quest_id, quest_root: 固定路径
built_in_mcp_namespaces: memory, artifact, bash_exec
default_locale: zh-CN
runner_name: codex
model: inherit
```

---

## 🔍 关键差异点（按影响程度排序）

### 1️⃣ 活跃分支与工作树 (最高频变化)

```diff
# 文件1.txt (最新)
research_head_branch: run/run-sign-balanced-no-content-frozen-mctb-40066
current_workspace_branch: run/run-sign-balanced-no-content-frozen-mctb-40066

# 文件20.txt (最早)
research_head_branch: idea/001-idea-f026016e
current_workspace_branch: idea/001-idea-f026016e
```

**含义**：从 `idea/*` 分支切换到 `run/*` 分支，表示从方案探索阶段进入实验执行阶段。

### 2️⃣ 活跃技能请求 (路由变化)

```
文件1: requested_skill: baseline
文件20: requested_skill: decision
中间变化: idea → experiment → decision → baseline → ...
```

**含义**：技能请求反映当前任务的阶段重心，频繁切换说明任务在"实验-分析-决策"循环中迭代。

### 3️⃣ 活跃锚点 (stage定位)

```
常见值: baseline, idea, experiment, decision
变化模式: experiment → decision → baseline (验证后回退) 或 idea → experiment (确认后推进)
```

### 4️⃣ 通信表面与连接器 (用户交互渠道)

```diff
# 本地调试阶段
active_surface: local
active_connector: local
latest_user_source: web-react

# QQ连接阶段  
active_surface: connector
active_connector: qq
latest_user_source: qq:direct:qq-profile-xxx::xxx
```

**含义**：从本地web界面切换到QQ连接器，用户交互渠道变化。

### 5️⃣ 用户要求历史 (active_user_requirements)

- 每个prompt末尾附加了当时的 `active-user-requirements.md` 快照
- 内容随用户新消息动态更新，是理解任务演进的关键线索

---

## 📊 变化统计摘要

| 变化维度 | 出现频次 | 典型模式 |
| --- | --- | --- |
| branch切换 | 20/20 | idea/\* ↔ run/\* 交替 |
| requested_skill | 20/20 | baseline/idea/experiment/decision 四态循环 |
| active_anchor | 20/20 | 随skill同步变化 |
| connector | \~12/20 | local ↔ qq 切换，反映交互渠道 |
| user_requirements | 20/20 | 持续追加，不覆盖历史 |

---

## 🎯 对调试/复现的价值

1. **分支追踪**：通过 `research_head_branch` 可精确定位每次调用时的代码状态
2. **阶段回溯**：`requested_skill` + `active_anchor` 组合可还原任务执行路径
3. **交互审计**：`active_connector` + `latest_user_source` 可追溯用户指令来源
4. **需求演化**：每个prompt末尾的用户要求快照，是理解任务变更的权威记录

---

## ⚠️ 注意事项

- 这些prompt是**喂给基座模型的完整上下文**，包含系统指令+运行时状态+用户要求
- 文件按时间倒序排列（1=最新），分析时请注意时序
- 90%+内容为固定system prompt，差异集中在Runtime Context区块（约行1309+）
- 如需对比某两个具体文件的差异，可用：

  ```bash
  diff -u version_998/1.txt version_998/20.txt | head -100
  ```

---

> 报告生成：自动分析20个prompt文件的Runtime Context区块差异\
> 下次更新：当有新prompt记录时，可追加到21.txt并更新本摘要