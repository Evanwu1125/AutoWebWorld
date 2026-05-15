# CC Landing Page Generator — Architecture

## High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLI Entry Points                                │
│                                                                         │
│   pnpm gen:kit          pnpm test:assets          pnpm dev              │
│   (scripts/             (scripts/                 (Next.js              │
│    generate-kit.ts)      test-assets.ts)           dev server)          │
└────────┬────────────────────┬──────────────────────────┬────────────────┘
         │                    │                          │
         ▼                    ▼                          ▼
┌─────────────────┐  ┌────────────────┐  ┌──────────────────────────────┐
│  System 1        │  │  Standalone    │  │  System 3: Next.js App       │
│  Visual Kit      │  │  Asset Tests   │  │                              │
│  Generator       │  │                │  │  app/page.tsx (手写首页)      │
│                  │  │                │  │  app/demo/page.tsx            │
│  visual-kit/     │  │                │  │    └─ PageRenderer           │
│  generator/      │  │                │  │       (JSON → 渲染页面)       │
│  index.ts        │  │                │  │                              │
└────────┬─────────┘  └───────┬────────┘  └──────────────┬───────────────┘
         │                    │                          │
         │         ┌──────────┘                          │
         ▼         ▼                                     ▼
┌──────────────────────────┐               ┌──────────────────────────────┐
│  Asset Generation Layer  │               │  Component Library           │
│  (src/assets-api/)       │               │  (src/components/)           │
│                          │               │                              │
│  nano-banana.ts          │               │  heroes/                     │
│  ├─ generateHeroImage    │               │  ├─ hero-gradient.tsx        │
│  └─ (gemini-3-pro via    │               │  └─ hero-video.tsx           │
│      DeepWisdom gateway) │               │                              │
│                          │               │  sections/                   │
│  kling.ts                │               │  ├─ bento-grid.tsx           │
│  ├─ generateKlingImage   │               │  ├─ feature-showcase.tsx     │
│  ├─ generateHeroVideo    │               │  ├─ stats-counter.tsx        │
│  └─ animateHeroImage     │               │  ├─ testimonials.tsx         │
│                          │               │  ├─ faq.tsx                  │
└────────────┬─────────────┘               │  └─ logo-cloud.tsx          │
             │                             │                              │
             │                             │  effects/                    │
             │                             │  ├─ scroll-reveal.tsx        │
             │                             │  ├─ text-reveal.tsx          │
             │                             │  ├─ aurora-background.tsx    │
             │                             │  ├─ particles.tsx            │
             │                             │  ├─ spotlight.tsx            │
             │                             │  └─ gradient-blur.tsx        │
             │                             │                              │
             │                             │  cta/  layout/  icons.tsx    │
             │                             │  page-renderer.tsx           │
             │                             │  providers.tsx               │
             │                             └──────────────────────────────┘
             │
             ▼
┌──────────────────────────┐
│  Shared Lib              │
│  (src/lib/)              │
│                          │
│  llm.ts                  │
│  ├─ callLLM()            │
│  ├─ parseLLMJson()       │
│  └─ getClient()          │
│                          │
│  utils.ts  (cn, clsx)    │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Content Pipeline        │
│  (src/generators/)       │
│  ※ 与素材模块已解耦       │
│                          │
│  orchestrator.ts         │
│  ├─ research             │
│  ├─ gen-content          │
│  ├─ quality-score        │
│  ├─ validate             │
│  ├─ gen-assets (TODO)    │
│  └─ assemble             │
│                          │
│  research.ts             │
│  prompts.ts              │
│  quality-scorer.ts       │
│  config/                 │
│  ├─ char-limits.ts       │
│  └─ product-info.ts      │
└──────────────────────────┘
```

## System Separation

```
┌──────────────┐     kit.json      ┌──────────────┐    JSON data     ┌──────────────┐
│  System 1    │ ─────────────────▶│  System 2    │ ────────────────▶│  System 3    │
│  Visual Kit  │                   │  Content     │                  │  Renderer    │
│  Generator   │                   │  Pipeline    │                  │  (Next.js)   │
│              │                   │              │                  │              │
│  输入:        │                   │  输入:        │                  │  输入:        │
│  - topic     │                   │  - topic     │                  │  - Landing   │
│  - ref URL   │                   │  - keyword   │                  │    PageData  │
│  - hero img  │                   │  - kit.json  │                  │    (JSON)    │
│  - colors    │                   │              │                  │              │
│              │                   │  输出:        │                  │  输出:        │
│  输出:        │                   │  - Landing   │                  │  - 渲染好的   │
│  - kit.json  │                   │    PageData  │                  │    HTML 页面  │
│  - hero.png  │                   │              │                  │              │
│  - video.mp4 │                   │              │                  │              │
└──────────────┘                   └──────────────┘                  └──────────────┘
```

## External API Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│                     DeepWisdom Gateway                       │
│                (newapi.deepwisdom.ai/v1)                     │
│                  OPENAI_API_KEY (必需)                        │
│                                                              │
│   ┌─────────────────────┐  ┌─────────────────────────────┐  │
│   │ LLM Calls           │  │ Image Generation             │  │
│   │ gemini-3.1-flash-   │  │ gemini-3-pro-image-preview   │  │
│   │ lite-preview        │  │                               │  │
│   │                     │  │ (IMAGE_MODEL env 可覆盖)      │  │
│   │ Used by:            │  │                               │  │
│   │ - content gen       │  │ Used by:                      │  │
│   │ - site analysis     │  │ - hero image gen              │  │
│   └─────────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     Kling AI (可选)                           │
│                   KLING_ACCESS_KEY                            │
│                   KLING_SECRET_KEY                            │
│                                                              │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│   │ Text→Image   │  │ Text→Video   │  │ Image→Video      │  │
│   │ kling-v2-1   │  │ kling-v1-6   │  │ (animate hero)   │  │
│   └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              Research APIs (内容管线用，素材模块不需要)         │
│                                                              │
│   Serper (SERPER_API_KEY)     Tavily (TAVILY_API_KEY)        │
│   Google SERP 搜索              网页内容提取                   │
└─────────────────────────────────────────────────────────────┘
```

## Key Data Flow

```
pnpm gen:kit --topic "X" --ref https://example.com --video
    │
    ▼
1. analyzeSite(url)                    ← LLM (lib/llm.ts)
    │ 抓取 HTML → 提取 CSS/结构 → LLM 分析视觉风格
    │ 输出: SiteAnalysis { colors, theme, fonts, animationIntensity }
    ▼
2. Build ColorPalette + Typography     ← 合并: user colors > site analysis > preset
    ▼
3. generateHeroImage(prompt)           ← DeepWisdom gateway (gemini-3-pro)
    │ 失败? → fallback: generateKlingImage()
    ▼
4. animateHeroImage(image, prompt)     ← Kling image-to-video (if --video)
    ▼
5. Write kit.json + assets → public/kits/<slug>/
    ├── kit.json        (VisualKit 完整规格)
    ├── hero.png        (AI 生成或用户提供)
    ├── hero-video.mp4  (可选)
    └── logo.svg        (可选)
```

## File Tree

```
src/
├── app/                          # Next.js pages
│   ├── page.tsx                  # 手写首页 (showcase)
│   ├── demo/page.tsx             # JSON → PageRenderer demo
│   ├── layout.tsx
│   └── globals.css
│
├── lib/                          # 共享工具 (无业务耦合)
│   ├── llm.ts                    # OpenAI-compatible LLM client
│   └── utils.ts                  # cn(), clsx helpers
│
├── assets-api/                   # 素材生成 (独立模块)
│   ├── index.ts                  # barrel exports
│   ├── nano-banana.ts            # 图片生成 (gateway)
│   └── kling.ts                  # 图片+视频生成 (Kling)
│
├── visual-kit/                   # System 1: 视觉规格生成器 (独立模块)
│   ├── index.ts
│   ├── types.ts                  # VisualKit, KitInput, SiteAnalysis...
│   ├── generator/
│   │   ├── index.ts              # generateVisualKit() 主流程
│   │   └── analyze-site.ts       # 参考网站风格分析
│   └── styles/
│       └── presets.ts            # 6 个预设主题
│
├── generators/                   # System 2: 内容生成管线 (已解耦)
│   ├── index.ts
│   ├── orchestrator.ts           # 主编排: research→gen→score→validate
│   ├── research.ts               # Serper + Tavily 网络调研
│   ├── prompts.ts                # LLM prompt 构建
│   ├── quality-scorer.ts         # 规则打分 + 验证
│   ├── llm.ts                    # re-export from lib/llm
│   ├── types.ts                  # LandingPageData 等核心类型
│   └── config/
│       ├── char-limits.ts
│       └── product-info.ts
│
├── components/                   # System 3: 渲染组件库
│   ├── page-renderer.tsx         # JSON → 页面 (核心桥接)
│   ├── providers.tsx
│   ├── icons.tsx
│   ├── heroes/
│   ├── sections/
│   ├── effects/
│   ├── cta/
│   └── layout/
│
└── hooks/
    ├── use-gsap.ts
    └── use-smooth-scroll.ts

scripts/
├── generate-kit.ts               # pnpm gen:kit CLI
└── test-assets.ts                # pnpm test:assets 验证脚本
```
