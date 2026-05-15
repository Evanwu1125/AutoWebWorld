# Landing Page Generation — Pipeline Design (Research Report)

> 调研时间: 2026-04-03
> 调研范围: Claude Code Skills/Hooks/Agent SDK/MCP、行业方案 (Framer AI/Durable/Relume/v0)、多 Agent 编排模式

---

## 一、核心结论

**推荐方案: Skills (交互开发) + Agent SDK (生产自动化) + Hooks (质量守门)**

不需要选一个，而是分层组合:

| 层 | 机制 | 职责 |
|----|------|------|
| 交互层 | **Claude Code Skills** | `/gen-kit`, `/gen-content`, `/gen-page` — 开发时单步或串联执行 |
| 质量层 | **Claude Code Hooks** | PostToolUse 自动校验 — 每次写文件都跑质量检查 |
| 生产层 | **Agent SDK** | `pnpm run pipeline --topic "X"` — CI/cron 无人值守执行 |
| 工具层 | **In-process MCP** | `createSdkMcpServer()` 把现有函数包装成 Claude 可调用的工具 |

---

## 二、当前问题诊断

### 2.1 两大系统没有串联

```
Visual Kit Generator ──→ kit.json (颜色/字体/图片)
                                                        ❌ 无合流点
Content Pipeline ──→ LandingPageData (文案/布局)
```

Content Pipeline 里的 `DesignDecision` 是 LLM 自己编的，没有用 Visual Kit 的分析结果。

### 2.2 失败时全量重试

`orchestrator.ts:189` — 质量分 < 60 就完全重跑内容生成。80% 好的内容被丢弃。

### 2.3 串行执行，速度慢

Visual Kit 和 Content Pipeline 无依赖关系，但没有并行。

### 2.4 素材生成是空壳

`orchestrator.ts:217` — gen-assets 步骤是 TODO placeholder。

---

## 三、推荐架构: 5 阶段并行管线

```
                    ┌──────────────────────────────────┐
                    │          Phase 1 (~20s)           │
                    │         并行执行，无依赖           │
                    │                                    │
                    │  ┌─────────────┐ ┌─────────────┐  │
  Topic + Keyword ──┤  │  Research   │ │ Visual Kit  │  │
  + Reference URL   │  │  Agent      │ │ Generator   │  │
                    │  │             │ │             │  │
                    │  │ Serper SERP │ │ 抓取参考站   │  │
                    │  │ Tavily 提取  │ │ LLM 分析风格 │  │
                    │  │             │ │ 生成图片     │  │
                    │  └──────┬──────┘ └──────┬──────┘  │
                    │         │               │          │
                    └─────────┼───────────────┼──────────┘
                              │               │
                              ▼               ▼
                    ┌─────────────────────────────────────┐
                    │          Phase 2 (~15s)              │
                    │       Design Decision + Content      │
                    │                                      │
                    │  VisualKit.colors ──→ DesignDecision │
                    │  VisualKit.layout ──→ (不再让LLM猜)   │
                    │                                      │
                    │  Research + DesignDecision            │
                    │       ──→ LLM Content Generation     │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │          Phase 3 (~5s)                │
                    │         Quality Gate (分层)            │
                    │                                       │
                    │  Layer 1: 规则打分 (< 10ms)            │
                    │    字数限制 / 关键词密度 / 通用词检测    │
                    │                                       │
                    │  Layer 2: LLM-as-Judge (可选)          │
                    │    语气一致性 / 数据可信度 / 差异化      │
                    │                                       │
                    │  ❌ < 60 → 定向修复 (不再全量重试)       │
                    │  ✅ >= 60 → 继续                       │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │          Phase 4 (后台, 30-120s)      │
                    │         Asset Enrichment              │
                    │         (页面已可预览，异步补充素材)      │
                    │                                       │
                    │  ┌──────────┐ ┌──────────┐ ┌───────┐ │
                    │  │Feature   │ │Hero Video│ │OG     │ │
                    │  │Images x3 │ │(Kling)   │ │Image  │ │
                    │  └──────────┘ └──────────┘ └───────┘ │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │          Phase 5                      │
                    │         Assembly + Deploy             │
                    │                                       │
                    │  Merge VisualKit + LandingPageData    │
                    │       ──→ final JSON                  │
                    │       ──→ Write to Strapi CMS         │
                    │       ──→ Trigger Next.js rebuild     │
                    │       ──→ Return preview URL          │
                    └──────────────────────────────────────┘
```

### 关键改进 vs 当前

| 改进 | 影响 | 难度 |
|------|------|------|
| Phase 1 并行 (Research ∥ Visual Kit) | 管线速度快 50% | 低 |
| DesignDecision 从 Visual Kit 派生 | 设计-内容一致性 | 中 |
| 定向修复替代全量重试 | 质量更高、成本降 60% | 中 |
| Phase 4 异步素材 (先出页面，后补图) | 感知速度 40s → 可预览 | 中 |
| Feature images 生成 (3张) | 视觉质量大幅提升 | 低 |

---

## 四、Claude Code 落地方案

### 4.1 Skills (交互开发)

在 `.claude/skills/` 下创建 4 个 skill:

#### `/gen-kit` — 素材生成

```yaml
# .claude/skills/gen-kit/SKILL.md
---
name: gen-kit
description: Generate visual kit (colors, typography, hero image/video) for a landing page topic. Use when user asks to create visual assets or a design kit.
disable-model-invocation: true
allowed-tools: Bash Read Write Glob
---

Generate a visual kit for the given topic.

## Arguments
- $0: Topic/product name (required)
- $1: Reference URL (optional)
- $2: Additional flags (optional, e.g. "--video")

## Steps

1. Run the generator:
   ```
   pnpm gen:kit --topic "$0" --desc "$0" $1 $2
   ```

2. Read the generated kit.json and summarize:
   - Color palette (primary, accent, background)
   - Typography (heading, body fonts)
   - Hero asset status (image generated? video generated?)
   - Animation intensity

3. Show the output directory path so user can inspect assets.
```

#### `/gen-content` — 内容生成

```yaml
# .claude/skills/gen-content/SKILL.md
---
name: gen-content
description: Generate landing page content (hero, features, FAQ, testimonials) via research + LLM. Use when user asks to generate copy or content for a landing page.
allowed-tools: Bash Read Write
---

Generate landing page content for the given topic.

## Arguments
- $0: Topic (required)
- $1: Target keyword (optional, defaults to topic)

## Steps

1. Run the content pipeline:
   ```
   pnpm tsx scripts/gen-content.ts --topic "$0" --keyword "${1:-$0}"
   ```

2. Report results:
   - Quality score and tier
   - Number of features, FAQ items, testimonials
   - Any validation warnings
   - If score < 60, ask user whether to refine or accept

3. Show path to the generated content JSON.
```

#### `/gen-page` — 完整管线

```yaml
# .claude/skills/gen-page/SKILL.md
---
name: gen-page
description: Full landing page generation pipeline - research, visual kit, content, assets, assembly. Use when user asks to generate a complete landing page.
allowed-tools: Bash Read Write Edit Glob Grep
---

Run the complete landing page generation pipeline.

## Arguments
- $0: Topic (required)
- $1: Target keyword (optional)
- $2: Reference URL (optional)

## Pipeline

### Phase 1: Parallel Research + Visual Kit
Run these in parallel (two Bash calls):
- `pnpm tsx scripts/gen-content.ts --research-only --topic "$0" --keyword "${1:-$0}"`
- `pnpm gen:kit --topic "$0" --desc "$0" ${2:+--ref "$2"}`

### Phase 2: Content Generation
Feed research data + visual kit colors into content generation:
- `pnpm tsx scripts/gen-content.ts --topic "$0" --keyword "${1:-$0}" --kit public/kits/<slug>/kit.json`

### Phase 3: Quality Gate
Check quality score. If < 60, run targeted refinement (not full regen).

### Phase 4: Preview
- Start dev server if not running: `pnpm dev`
- Report the preview URL: http://localhost:3000/demo

### Phase 5: Asset Enrichment (background)
- Generate feature images and video in background
- Update the user when assets are ready
```

#### `/test-assets` — 素材验证

```yaml
# .claude/skills/test-assets/SKILL.md
---
name: test-assets
description: Test asset generation APIs (image, video, site analysis). Use to verify API keys and generation capabilities.
disable-model-invocation: true
allowed-tools: Bash Read
---

Run asset generation verification.

## Arguments
- $0: Test type (optional: "image", "kling-image", "video", "analyze", "all")

## Steps

1. Run: `pnpm test:assets ${0:+--$0}`
2. Report pass/fail for each test
3. If failures, diagnose the issue (missing API key? rate limit? network?)
```

### 4.2 Hooks (质量守门)

```jsonc
// .claude/settings.json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write",
        "hooks": [
          {
            "type": "command",
            "command": "node -e \"const f=process.env.TOOL_FILE||''; if(f.endsWith('.json')&&f.includes('landing')){const d=require(f);const e=[];if(!d.hero?.title)e.push('missing hero title');if(!d.features?.length)e.push('no features');if(e.length){console.error('Quality issues: '+e.join(', '));process.exit(1)}}\"",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

### 4.3 Agent SDK (生产自动化)

```typescript
// scripts/run-pipeline.ts — 无人值守执行
import { query, tool, createSdkMcpServer } from "@anthropic-ai/claude-agent-sdk";

// 把现有函数包装成 MCP 工具
const pipelineTools = createSdkMcpServer({
  name: "landing-pipeline",
  tools: [
    tool("research_topic", "...", schema, async (args) => {
      const result = await gatherTopicData(args.topic, args.keyword);
      return { content: [{ type: "text", text: JSON.stringify(result) }] };
    }),
    tool("generate_visual_kit", "...", schema, async (args) => {
      const kit = await generateVisualKit(args);
      return { content: [{ type: "text", text: JSON.stringify(kit) }] };
    }),
    tool("generate_content", "...", schema, async (args) => {
      const result = await generateLandingPage(args);
      return { content: [{ type: "text", text: JSON.stringify(result) }] };
    }),
  ],
});

// Claude 作为编排者
for await (const msg of query({
  prompt: `Generate a landing page for "${topic}".
    Run research and visual kit in parallel.
    Use visual kit colors for design decisions.
    If quality < 60, refine specific issues.`,
  options: { mcpServers: { pipeline: pipelineTools } },
})) {
  // stream progress...
}
```

---

## 五、行业对标

| 产品 | 架构模式 | 速度 | 质量控制 | 我们的对比 |
|------|----------|------|----------|-----------|
| **Framer AI** | 单次 LLM → 模板填充 | ~15s | 结构校验 | 我们更深 (web research + quality scoring) |
| **Relume** | 两步: 线框 → 内容, 人工审核 | ~30s + 人工 | Human-in-loop | 我们可自动化 + 人工可选 |
| **v0 / Bolt** | 生成代码 → 预览 → 用户反馈 → 迭代 | ~10s 首屏 | 用户迭代 | 我们生成 JSON 不是代码，更可控 |
| **Durable** | 行业分类 → 模板选择 → 填充 | ~10s | 模板约束 | 我们更灵活 (无固定模板) |
| **我们 (目标)** | 并行研究+设计 → 内容 → 分层质量门 → 异步素材 | ~40s 可预览 | 规则 + LLM Judge | 质量最高，速度中等 |

### 行业关键洞察

1. **设计决策应该是约束选择，不是生成** — Framer 从 20 个模板里选，不让 LLM 凭空设计
2. **先出页面，后补素材** — v0 和 Bolt 都是秒级首屏，渐进增强
3. **定向修复 >> 全量重试** — 2025 年的主流做法是把质量问题反馈给 LLM 让它只修有问题的部分
4. **图生视频 >> 文生视频** — 先生成静态图，再用 Kling/Runway 做动画，质量远高于纯文生视频

---

## 六、实施优先级

### P0 — 立即可做 (本周)

| 任务 | 文件 | 改动量 |
|------|------|--------|
| 并行 Research + Visual Kit | `orchestrator.ts` 改 `Promise.all` | ~20 行 |
| DesignDecision 从 VisualKit 派生 | `orchestrator.ts` + `prompts.ts` | ~50 行 |
| 定向修复替代全量重试 | `orchestrator.ts` L189 | ~40 行 |
| 创建 `/gen-kit` skill | `.claude/skills/gen-kit/SKILL.md` | 新文件 |
| 创建 `/test-assets` skill | `.claude/skills/test-assets/SKILL.md` | 新文件 |

### P1 — 近期 (下周)

| 任务 | 说明 |
|------|------|
| Feature images 生成 | 每个 showcase item 生成一张图 |
| `/gen-page` 完整管线 skill | 串联所有步骤 |
| LLM-as-Judge 质量门 | 规则打分后加语义评估 |
| 异步素材 + 渐进预览 | Phase 4 后台跑，先渲染 gradient hero |

### P2 — 中期

| 任务 | 说明 |
|------|------|
| Agent SDK 生产脚本 | `scripts/run-pipeline.ts` 无人值守 |
| Section 级别再生成 | 只重新生成 FAQ / Hero 等单个 section |
| OG Image 生成 | hero 图 + 标题合成社交分享图 |
| Strapi CMS 写入 | 生成完直接入库 |

---

## 七、API Key 需求总结

### 最小可用 (只要 1 个 key)

```bash
OPENAI_API_KEY=<openai-api-key>   # DeepWisdom 网关 → LLM + 图片 + 风格分析
```

### 完整能力 (4 个 key)

```bash
OPENAI_API_KEY=<openai-api-key>   # LLM + 图片生成
KLING_ACCESS_KEY=xxx         # Kling 图片 + 视频
KLING_SECRET_KEY=xxx         # Kling 配对
SERPER_API_KEY=xxx           # Google SERP 搜索 (Research)
TAVILY_API_KEY=tvly-xxx      # 网页内容提取 (Research)
```

### 生产自动化 (额外)

```bash
ANTHROPIC_API_KEY=<anthropic-api-key> # Agent SDK 需要 (独立于 Claude Code 订阅)
```
