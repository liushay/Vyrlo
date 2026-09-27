# 已知限制清单 — Vyrlo v1.0.0

> **版本**：v1.0.0
> **创建日期**：2026-09-27 (Asia/Shanghai)
> **最后更新**：2026-09-28T02:00:00+08:00（E7 收尾）
> **来源**：整合自 `E6_KNOWN_LIMITATIONS.md`、`E1.5_REPORT.md`、`acceptance-e4/E4_REPORT.md`、`acceptance-e7/E7_REPORT.md`
> **用途**：上线前明确哪些限制是可接受的、哪些需要后续解决

---

## 目录

1. [可接受（Acceptable）](#1-可接受acceptable)
2. [需后续解决（Needs Resolution）](#2-需后续解决needs-resolution)
3. [E7 新发现问题](#3-e7-新发现问题)
4. [风险矩阵](#4-风险矩阵)

---

## 1. 可接受（Acceptable）

> 这些限制在当前阶段被明确接受，不需要在 v1.0.0 上线前修复。

### 1.1 模型能力边界 — F8：读→写占位符幻觉

| 属性 | 值 |
|------|-----|
| **影响范围** | 约 12% 任务（E1 中 6/50：T31, T32, T34, T37, T39, T40） |
| **影响类型** | 跨文件重构 / 读→写复制类任务 |
| **根因** | qwen2.5:7b-instruct 在 read_file→write_file 链路中，倾向于在 write_file 参数中使用自产占位符（如 `"<read_file_result>"`、`"read_file返回的内容"`），而非等待 read_file 工具返回实际内容后使用 |
| **严重程度** | 中等 — 不影响读写分离的单步工具调用场景 |
| **规避方法** | 1) 换用更大模型（qwen2.5:14b+）；2) 将跨文件复制拆分为两个独立任务；3) 在 system prompt 中增加「必须使用工具返回的真实内容，禁止使用占位符」约束 |
| **判定** | ✅ 可接受 — 属于小模型系统性能力边界，非框架代码缺陷 |

**证据**：`E1.5_REPORT.md` — exp4_effective_rate.md、exp3_strong_model/conclusion.md；`E6_KNOWN_LIMITATIONS.md` §1、§5

### 1.2 本地 Ollama 依赖（无云端 fallback）

| 属性 | 值 |
|------|-----|
| **影响范围** | 所有任务 |
| **影响类型** | 若 Ollama 服务不可用，整个系统无法工作 |
| **根因** | 无 OpenAI / Anthropic 等云端 API Key，系统当前仅验证 Ollama 本地部署 |
| **严重程度** | 中等 — 单点依赖 |
| **规避方法** | 1) 部署时确保 Ollama 服务高可用；2) 配置多模型 fallback 链（qwen2.5:3b → qwen2:7b → llama3.1:8b）；3) 监控 `http://localhost:11434` 健康状态 |
| **判定** | ✅ 可接受 — 当前产品定位为本地部署方案，多模型 fallback 已验证（E5） |

**证据**：`E6_KNOWN_LIMITATIONS.md` §6.1；`acceptance-e5/E5_REPORT.md` §三

### 1.3 仅 3 个文件工具

| 属性 | 值 |
|------|-----|
| **影响范围** | Agent 能力受限 |
| **影响类型** | 不支持网络访问、数据库、代码执行等能力 |
| **根因** | v1.0 聚焦于文件操作场景的端到端验证 |
| **严重程度** | 低 — 不影响已覆盖场景 |
| **规避方法** | 通过扩展机制注册自定义工具（见 `docs/USER_GUIDE.md` §3） |
| **判定** | ✅ 可接受 — 工具系统为可扩展架构（E5 验证 3/3 通过） |

**证据**：`E6_KNOWN_LIMITATIONS.md` §6.2；`acceptance-e5/E5_REPORT.md` §四

### 1.4 性能数据适用范围限定

| 属性 | 值 |
|------|-----|
| **影响范围** | 性能基准不可跨环境复用 |
| **影响类型** | E3 的 P95/P50=1.99 仅适用于当前环境（Windows 11 + NVIDIA GPU + Q4_K_M 量化） |
| **根因** | 性能深度依赖硬件配置和模型量化级别 |
| **严重程度** | 低 — 数据标注了适用范围 |
| **规避方法** | 更换硬件/模型后重新运行基准采集 |
| **判定** | ✅ 可接受 — 性能数据已明确标注适用范围 |

**证据**：`E6_KNOWN_LIMITATIONS.md` §3

### 1.5 成本计量为 0（本地 Ollama）

| 属性 | 值 |
|------|-----|
| **影响范围** | 成本追踪数据无意义 |
| **影响类型** | `OllamaAdapter.get_cost()` 返回 0 |
| **根因** | 本地部署无 API 调用费用，实际成本为 GPU 电力和硬件折旧 |
| **严重程度** | 低 — 对生产决策无影响 |
| **规避方法** | 若切换到云端模型，CostGuard 中间件自动启用成本追踪 |
| **判定** | ✅ 可接受 — 当前阶段使用本地模型 |

**证据**：`E6_KNOWN_LIMITATIONS.md` §6.4

### 1.6 长运行内存行为未验证（>1000 轮）

| 属性 | 值 |
|------|-----|
| **影响范围** | 超长运行场景 |
| **影响类型** | E2 仅验证 100 次循环（内存波动 5.66%），>1000 轮的内存行为未知 |
| **根因** | 验证覆盖优先于当前上线场景（单任务通常 <10 轮） |
| **严重程度** | 低 — 单任务不达此规模 |
| **规避方法** | 上线后监控内存趋势；context_compressor 中间件定期压缩上下文 |
| **判定** | ✅ 可接受 — 当前上线场景不触发 |

**证据**：`E6_KNOWN_LIMITATIONS.md` §6.5；`acceptance-e2/E2_REPORT.md`

### 1.7 长任务中断恢复未被触发验证（E7 发现）

| 属性 | 值 |
|------|-----|
| **影响范围** | 中断恢复机制的完整覆盖 |
| **影响类型** | E7 长任务（T56-T60）在本地 qwen2.5:7b 模型下 15-40s 内完成，未达到设计的 >10min 中断触发阈值。中断恢复代码骨架已验证，但真实中断场景未测试 |
| **根因** | 本地快速模型无法产生足够长的执行时间触发中断机制 |
| **严重程度** | **Low** — 中断恢复代码路径已在 E5/D5 验证，E7 首轮 5/5 通过 |
| **规避方法** | 1) 使用云端模型（如 gpt-4o-mini）重跑长任务；2) 在工具执行中注入延迟模拟慢速场景 |
| **判定** | ✅ 可接受 — 首轮正确性已充分验证，完整端到端安排在 v1.1 |

**证据**：`acceptance-e7/E7_SUMMARY.json` §defects[1]、§long_tasks

### 1.8 Cross-Day 对话第二轮需物理等待（E7 发现）

| 属性 | 值 |
|------|-----|
| **影响范围** | Cross-day 对话记忆保持验证 |
| **影响类型** | Run1 5/5 通过（同一 session 内跨天模拟），Run2（12h+ 真实物理跨天）无法在单开发 session 内完成 |
| **根因** | 跨天验证需要真实的物理时间流逝，非技术缺陷 |
| **严重程度** | **Low** — Run1 5/5 通过已覆盖跨天对话逻辑正确性，Run2 仅验证长时间物理存储持久性 |
| **规避方法** | 1) 使用系统时间模拟（需额外开发）；2) 安排自动化 cron 任务在夜间运行 Run2 脚本 |
| **判定** | ✅ 可接受 — Run1 通过 + E5 存储持久性验证覆盖了核心风险。Run2 自动化安排在 v1.1 |

**证据**：`acceptance-e7/E7_SUMMARY.json` §defects[2]、§cross_day

---

## 2. 需后续解决（Needs Resolution）

> 这些限制需要在后续版本中修复，但在 v1.0.0 上线前被明确接受。

### 2.1 可观测性缺口 — G1～G6（E4 批次识别）

| # | 缺口 | P 级 | 影响范围 | 规避方法 | 目标版本 |
|---|------|:---:|---------|---------|:---:|
| **G1** | LLM 请求/响应完整 payload 未记录 | P1 | 深入排查需回溯上下文 | 手动复现 + DEBUG 日志 | v1.1 |
| **G2** | 中间件决策路径未记录 | P1 | 无法快速判断中间件行为 | 检查 event_log 中间接推断 | v1.1 |
| **G3** | tool_result_feedback 注入内容未保留 | P1 | 无法确认反馈是否生效 | 检查下游 LLM 响应推断 | v1.1 |
| **G4** | LLM 响应 tool_calls 原始 JSON 未记录 | P2 | 解析器故障时无法回溯 | 额外启用 DEBUG 日志 | v1.2 |
| **G5** | session/ctx 生命周期图缺失 | P2 | 无时间线父子关系 | 手动追踪 session span | v1.2 |
| **G6** | 环境/配置未与 run 自动关联 | P2 | 故障排查需手动对照 | 故障演练参考 environment.md | v1.2 |

> 注：E4 验证确认现有可观测性可在 2-3 分钟内完成故障定位（3/3 通过）。这些缺口增加排查复杂度但不阻塞上线。

**证据**：`E6_KNOWN_LIMITATIONS.md` §8；`acceptance-e4/E4_REPORT.md`

### 2.2 DeepSeek 系列模型不兼容

| 属性 | 值 |
|------|-----|
| **影响范围** | 无法使用 DeepSeek flash/chat 系列模型 |
| **影响类型** | DeepSeek 的 function calling 使用 `<｜｜DSML｜｜ invoke>` 格式，非标准 OpenAI tool_calls，ToolCallParser 无法解析（E3.5 验证 0/6 成功率） |
| **根因** | ToolCallParser 仅支持 OpenAI native function calling 格式 |
| **严重程度** | 中 — 限制了模型选择范围 |
| **规避方法** | 仅使用已验证的兼容模型（qwen2.5 和 qwen2 系列、llama3.1） |
| **判定** | ⚠️ 需后续解决 — 若需支持 DeepSeek，需开发专用 Parser 策略 |

**证据**：`E6_KNOWN_LIMITATIONS.md` §2；`E6_DECISION.md` §5.2；`acceptance-e3.5/E3.5_REPORT.md`

### 2.3 Anthropic Claude 未验证

| 属性 | 值 |
|------|-----|
| **影响范围** | 无法使用 Claude 系列模型 |
| **影响类型** | Anthropic 使用独有的 tool_use 格式，与标准 OpenAI tool_calls 不兼容，需额外适配器 |
| **根因** | 无 Anthropic API Key + 格式不兼容 |
| **严重程度** | 低 — 非目标场景 |
| **规避方法** | 使用 OpenAI 兼容的模型替代 |
| **判定** | ⚠️ 需后续解决 — 若需支持，需开发 Anthropic Provider 适配器 |

**证据**：`E6_KNOWN_LIMITATIONS.md` §2

### 2.4 全局状态污染未经充分验证

| 属性 | 值 |
|------|-----|
| **影响范围** | 高并发 / 长时间运行场景 |
| **影响类型** | E2 验证了 ctx.shared 的隔离，但模块级缓存、单例等全局状态的跨 session 污染未充分验证 |
| **根因** | 验证覆盖侧重单 session 可靠性 |
| **严重程度** | 中 — 高并发场景可能有潜在风险 |
| **规避方法** | 生产环境避免在工具/中间件中使用模块级可变全局状态 |
| **判定** | ⚠️ 需后续解决 — 上线后持续监控 |

**证据**：`E6_KNOWN_LIMITATIONS.md` §6.5

### 2.5 Multi-Agent 场景 state_pollution 误报（E7 发现）

| 属性 | 值 |
|------|-----|
| **影响范围** | 多 Agent 协作场景 |
| **影响类型** | 状态完整性检查对 `observability` 和 `system_prompt_consistency` key 产生误报（false positive）。子 Agent 继承父 Agent 上下文时，这些 key 在子 Agent 中的作用域语义与检查器预期不同 |
| **根因** | `state_pollution` 检查器未区分 single-agent 和 multi-agent 场景下的 key 作用域语义 |
| **严重程度** | **Low** — 不影响功能正确性，仅误报 |
| **规避方法** | 在 multi-agent 场景中忽略 `observability` 和 `system_prompt_consistency` 两个 key 的 state_pollution 告警；检查其他 key 保持有效 |
| **判定** | ⚠️ 需后续解决 — v1.1 增加 multi-agent 场景的 key 白名单 |

**证据**：`acceptance-e7/E7_SUMMARY.json` §defects[0]

---

## 3. E7 新发现问题

> 本章节汇总 E7 最终验收中新识别的问题，已在其他章节中归类。

| # | 限制 | 类别 | 严重级别 | 规避方法 | 计划修复版本 |
|---|------|------|:---:|------|:---:|
| **E7-L1** | multi-agent 场景 state_pollution 误报（observability / system_prompt_consistency key） | 需后续解决 | Low | 忽略已知误报 key | v1.1 |
| **E7-L2** | 长任务（>10min）中断恢复未被触发验证 | 可接受 | Low | 云端慢模型或延迟注入模拟 | v1.1 |
| **E7-L3** | Cross-day Run2 需 12h+ 物理等待 | 可接受 | Low | 时间模拟或 cron 自动化 | v1.1 |

**证据**：`acceptance-e7/E7_REPORT.md` §4-§6；`acceptance-e7/E7_SUMMARY.json` §defects

---

## 4. 风险矩阵

| # | 限制 | 类别 | 严重程度 | 可逆性 | 上线判定 |
|---|------|------|:---:|:---:|:---:|
| L1 | F8 读→写占位符幻觉 | 可接受 | 中 | 模型切换可缓解 | ✅ |
| L2 | 本地 Ollama 单点依赖 | 可接受 | 中 | 多模型 fallback | ✅ |
| L3 | 仅 3 个文件工具 | 可接受 | 低 | 可扩展（3/3） | ✅ |
| L4 | 性能数据不可跨环境 | 可接受 | 低 | 重新采集 | ✅ |
| L5 | 成本计量为 0 | 可接受 | 低 | 切换云端模型自动启用 | ✅ |
| L6 | >1000 轮内存未验证 | 可接受 | 低 | 监控 | ✅ |
| L7 | 可观测性缺口 G1-G6 | 需后续解决 | 中 | E4 验证 2-3min 可定位 | ✅（v1.1 修复 P1） |
| L8 | DeepSeek 不兼容 | 需后续解决 | 中 | 专用 Parser | ✅（v1.1 适配） |
| L9 | Anthropic 未验证 | 需后续解决 | 低 | 专用 Provider | ✅（v1.2 适配） |
| L10 | 全局状态污染未充分验证 | 需后续解决 | 中 | 上线后监控 | ✅（v1.1 增强） |
| L11 | Multi-agent state_pollution 误报 | E7 新发现 | Low | 忽略已知 key | ✅（v1.1 白名单） |
| L12 | 长任务中断恢复未触发验证 | E7 新发现 | Low | 慢模型/延迟注入 | ✅（v1.1 补充） |
| L13 | Cross-day Run2 需物理等待 | E7 新发现 | Low | 时间模拟/cron | ✅（v1.1 自动化） |

---

## 附录：原始报告索引

| 报告 | 文件 | 与限制相关的章节 |
|------|------|---------|
| E1.5 | `E1.5_REPORT.md` | exp4_effective_rate.md、模型能力边界分析 |
| E2 | `acceptance-e2/E2_REPORT.md` | 可靠性、内存稳定性 |
| E3.5 | `acceptance-e3.5/E3.5_REPORT.md` | DeepSeek 环境验证、F1 修复 |
| E4 | `acceptance-e4/E4_REPORT.md` | 可观测性缺口 G1-G6、故障定位演练 |
| E5 | `acceptance-e5/E5_REPORT.md` | 可降级 3/3、可扩展 3/3 |
| E6 | `E6_KNOWN_LIMITATIONS.md` | **本文件的主要来源**（模型能力边界、FC 支持、性能适用范围、三级成功率口径、工具系统限制、E4 缺口） |
| E7 | `acceptance-e7/E7_REPORT.md`、`acceptance-e7/E7_SUMMARY.json` | E7-L1~L3 来源（state_pollution 误报、长任务中断、Cross-day 物理等待） |