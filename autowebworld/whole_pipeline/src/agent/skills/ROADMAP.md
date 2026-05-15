# Landing Page Generator — Roadmap

> 优化顺序：Layer 3 (Page) 先跑通 → Skill 闭环验证 → Layer 1/2 按需优化

## 架构原则

### 三层生成

```
Layer 1 (Assets)  → 图片/视频资产
Layer 2 (Content) → 文案/结构内容
Layer 3 (Page)    → 设计语言 + 动效原语 + LLM 生成页面代码
```

### 生成式 Page，不是模板化 Page

**不使用固定组件拼装页面。** LLM 直接生成每个页面的 JSX 代码。

可复用（沉淀）：
- Design tokens（颜色、字体、间距、动效参数）→ `src/page/design/`
- 动效原语（scroll-reveal, text-reveal, particles, spotlight）→ `src/page/components/effects/`
- 可达性模式（focus ring, aria, skip-nav, reduced-motion）→ 内建于原语
- Hooks（GSAP, smooth scroll）→ `src/page/hooks/`

不可复用（每次生成）：
- 页面的具体 section 结构和排版
- 页面节奏（哪些 section、什么顺序、间距）
- 每个 section 内部的 HTML/JSX 组合

### 固化机制

当一个生成的页面质量很好时，可以固化：
- 保存到 `src/page/references/<pattern-name>.tsx`（按设计模式命名，不按主题）
- 头部 metadata 描述**为什么好**（构图手法、设计原则），不是长什么样
- 提取新的动效/布局原语到 `src/page/components/effects/`
- 提取新的 preset 到 `src/page/design/presets.ts`

Reference 按设计模式分类，不按主题/配色。一个 dark-tech 页面的 "asymmetric testimonials" 手法可以被 warm-minimal 页面复用。

### 三种生成模式

| 模式 | 意图 | Reference 策略 | Temperature |
|------|------|---------------|-------------|
| `explore` | 探索新设计方向 | 不给 reference | 高 |
| `refine` | 质量稳定 + 风格多样 | 随机选 1-2 个不同 pattern 的 reference | 中 |
| `replicate` | 复刻指定风格 | 指定某个 reference | 低 |

防收敛机制：
- Reference 从不同 pattern 类别随机选取
- Prompt 要求"提取设计原则，不要复制布局"
- Judge 检查与已有 reference 的相似度，太像扣分
- 定期 explore 发现新设计方向

### Judge / Fix 分离原则

**Judge 和 Fix 必须独立，不能由同一个 skill 既评分又修复。**

```
Judge Skills (只读，不改代码)        Fix Skills (只改代码，不评分)
  /critique  → UX 评分 0-40           /typeset   → 字体层级
  /audit     → 技术评分 0-20          /colorize  → 配色策略
                                      /animate   → 动效编排
                                      /polish    → 像素级打磨
                                      /bolder    → 增强视觉冲击
                                      /quieter   → 降低视觉强度
```

编排流程：
```
gen-page 编排
  Phase 1: gen-kit     → kit.json + hero assets
  Phase 2: gen-content → content.json
  Phase 3: LLM 生成页面 JSX（使用 design tokens + effects + references）
  Phase 4: Judge (/critique + /audit → 评分 + 问题列表)
  Phase 5: Fix (根据问题选对应 skill 修复)
  Phase 6: Re-Judge (独立重新评分)
  循环直到达标 → Ship
  可选: 固化 → 保存到 references/
```

---

## Layer 1: Asset Generation (`src/assets/`)

### P0 — 核心

- [ ] **视频 seamless loop 保障**
  - 当前：prompt 里写了 "seamless loop" 但无技术保障，Kling 不保证首尾帧一致
  - 目标：prompt 强化 ("static camera, identical first/last frame, continuous flowing motion") + 播放侧 CSS crossfade overlay 兜底
  - 文件：`src/assets/video.ts`, `src/page/components/heroes/hero-video.tsx`

- [ ] **图片 prompt 优化**
  - 当前：prompt 模板简单 (`stylePrefix + prompt + "No text"`)
  - 目标：建立 style-aware prompt 模板，或接入 `/prompt-master` skill 在生成前优化 prompt
  - 文件：`src/assets/image.ts`

### P1 — 增强

- [ ] **Feature Showcase 图片批量生成**
  - 当前：只生成 Hero 图，Feature Showcase 2-4 张配图为空
  - 目标：在 kit-generator 中并行生成 feature 图片 (`Promise.all`)
  - 文件：`src/page/design/kit-generator.ts`

- [ ] **图片持久化 + Next.js Image 优化**
  - 当前：base64 data URL 或临时远程 URL
  - 目标：下载到 `public/kits/<slug>/`，通过 `next/image` 提供优化
  - 文件：`src/page/design/kit-generator.ts`, component 渲染层

### P2 — 可选

- [ ] **OG Image 生成**
  - 用于社交分享预览图
  - 可基于 Hero 图 + 标题文字叠加

---

## Layer 2: Content Generation (`src/content/`)

### P0 — 核心

- [ ] **LLM 模型选择**
  - 当前：`gemini-3.1-flash-lite-preview`（快但质量有限）
  - 目标：评估 Claude Opus / Sonnet 用于内容生成的质量差异，可能按场景分模型
  - 文件：`src/lib/llm.ts`

- [ ] **Product Info 动态加载**
  - 当前：硬编码 Atoms 信息在 `config/product-info.ts`
  - 目标：支持从 JSON 文件 / URL / CMS 读取，可做成 Skill (`/load-product`)
  - 文件：`src/content/config/product-info.ts`

### P1 — 增强

- [ ] **Section 级定向修复**
  - 当前：质量 < 60 时全量重生成，浪费 token
  - 目标：识别低分 section，只重新生成该 section
  - 文件：`src/content/generator.ts`

- [ ] **Phase 1 并行化**
  - 当前：research 串行执行
  - 目标：research + visual kit 分析用 `Promise.all` 并行
  - 文件：`src/content/generator.ts`

### P2 — 可选

- [ ] **A/B 变体生成**
  - 同一 topic 生成多套 hero 文案 / CTA，供选择

- [ ] **内容缓存/版本控制**
  - 对同一 topic 的 research 结果缓存，避免重复调研

---

## Layer 3: Page Generation (`src/page/`)

### P0 — 核心

- [ ] **Renderer 消费 kit.json 配色/字体**
  - 当前：全部硬编码 `bg-black`, `text-white`，Visual Kit 的配色白生成了
  - 目标：PageRenderer 接受 VisualKit 参数，通过 CSS variables 或 Tailwind 动态主题驱动组件
  - 文件：`src/page/renderer.tsx`, 所有 component

- [ ] **预设字体违规修复**
  - 当前：6 个预设中 5 个用了 BANNED 的 Inter
  - 目标：替换为高辨识度字体 (e.g. Space Grotesk, Instrument Serif, DM Mono, Satoshi)
  - 文件：`src/page/design/presets.ts`

- [ ] **Feature Icon API 迁移**
  - 当前：13 个硬编码 SVG，LLM 只能从中选择
  - 目标：迁移到 Icon API (Iconify / Lucide CDN)，LLM 可引用任意 icon 名称，渲染时通过 API 解析
  - 文件：`src/lib/icon-registry.ts`, `src/page/components/icons.tsx`

- [x] **Logo Cloud 动态化 (logo.dev API)** — 2026-04-07
  - `src/lib/logo.ts`: `getLogoUrl(domain)` + `inferDomain(name)` 接入 logo.dev API
  - `logo-cloud.tsx`: 接受 `companies` prop (域名/公司名)，通过 logo.dev 动态渲染
  - `content/types.ts`: `LogoCloudData.companies` 字段，LLM 可输出相关公司列表

### P1 — 增强

- [ ] **`/demo/[slug]` 预览路由完善**
  - 当前：路由存在但功能简单
  - 目标：支持从 `output/<slug>/content.json` + `public/kits/<slug>/kit.json` 组合加载渲染

- [ ] **prefers-reduced-motion 适配**
  - 当前：组件缺动画降级方案
  - 目标：所有 GSAP/Framer Motion 动画检测 `prefers-reduced-motion`，提供静态 fallback
  - 文件：`src/page/components/effects/*.tsx`

- [ ] **Light Theme 支持**
  - 当前：只有暗色主题
  - 目标：组件通过 CSS variables 支持亮色/暗色切换

### P2 — 可选

- [ ] **组件变体扩展**
  - 更多 Hero 样式 (3D, particle-field)
  - 更多 CTA 样式 (floating, sticky)
  - Pricing table section

- [ ] **SEO Metadata 完善**
  - JSON-LD structured data
  - canonical URL
  - OG tags from generated content

---

## 跨层 / 全局

- [ ] **`/gen-page` 全流程串联**
  - 当前：`gen:kit` 和 `gen:content` 是分开的 CLI
  - 目标：一个命令串联 Layer 1 → 2 → 3，输入 topic 输出可预览页面

- [ ] **CMS 集成 (Strapi)**
  - 将生成的内容写入 Strapi CMS，而非本地 JSON 文件

- [ ] **部署自动化**
  - 生成完直接部署到 Vercel preview URL

---

## 已完成

- [x] 三层架构重构 (assets / content / page) — 2026-04-07
- [x] 移除 Kling 图片生成通道，只保留 Nano Banana — 2026-04-07
- [x] Icon 名称列表抽离到 `src/lib/icon-registry.ts`，解除 Layer 2 → Layer 3 反向依赖 — 2026-04-07
- [x] 所有 import 路径更新，build 通过 — 2026-04-07
- [x] CLAUDE.md 重写反映三层架构 — 2026-04-07
- [x] 每层创建 barrel export (index.ts) — 2026-04-07
