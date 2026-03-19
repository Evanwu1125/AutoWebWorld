# Development Checklist - Team Chat Application

## Phase 1: Project Setup
- [ ] Install dependencies (pnpm i) - Will run later
- [ ] Configure Tailwind (content paths) - Default is likely okay, verify later
- [ ] Create generic styles (app.css)

## Phase 2: Core Infrastructure
- [ ] `src/stores/signature.js` (Pinia + PersistedState)
- [ ] `src/stores/data.js` (Mock Data - CRITICAL: Large dataset)
- [ ] `src/router/index.js` (Routes for all 16 pages)
- [ ] `src/main.js` (Plugin setup)
- [ ] `src/App.vue` (Layout & Global Interceptors: Cookie, Permission)

## Phase 3: Page Implementation (16 Pages)
1.  **HOME** (`src/pages/HOME.vue`)
    - [ ] Hero Section
    - [ ] Cookie Consent Modal (Interceptor)
    - [ ] Action: Accept Cookies
    - [ ] Actions: Go to Workspace (Direct, Hover, Menu variants)

2.  **WORKSPACE_OVERVIEW** (`src/pages/WORKSPACE_OVERVIEW.vue`)
    - [ ] Location Permission Modal (Interceptor)
    - [ ] List Workspaces
    - [ ] Actions: Grant Location, Open Default Workspace, Back Home, Open Profile

3.  **CHANNEL_LIST** (`src/pages/CHANNEL_LIST.vue`)
    - [ ] Sidebar Navigation (Channels & DMs)
    - [ ] Search Bar
    - [ ] Filter Controls (Checkbox, Slider, Sort)
    - [ ] Channel List (Regular, Filtered, Matched, Anchor)
    - [ ] Actions: Filters, Search, Scroll, Open Channel, Open DM List, Open Profile, Back WS

4.  **DM_LIST** (`src/pages/DM_LIST.vue`)
    - [ ] DM List Display
    - [ ] Search & Filters
    - [ ] Actions: Filters, Search, Open DM, Back Channels

5.  **CHANNEL_DETAIL** (`src/pages/CHANNEL_DETAIL.vue`)
    - [ ] Chat Header (Settings button)
    - [ ] Message List (Mock messages)
    - [ ] Actions: Compose Message, Open Settings, Schedule Message, Back List

6.  **DM_DETAIL** (`src/pages/DM_DETAIL.vue`)
    - [ ] DM Header
    - [ ] Message List
    - [ ] Actions: Compose DM, Back DM List

7.  **MESSAGE_COMPOSE** (`src/pages/MESSAGE_COMPOSE.vue`)
    - [ ] Text Area
    - [ ] Mention & Emoji buttons
    - [ ] Send Button
    - [ ] Actions: Type, Mention, Emoji, Send, Back

8.  **CHANNEL_SETTINGS** (`src/pages/CHANNEL_SETTINGS.vue`)
    - [ ] Form: Name, Description, Privacy
    - [ ] Actions: Type Name/Desc, Select Privacy, Save, Back

9.  **MESSAGE_SCHEDULE** (`src/pages/MESSAGE_SCHEDULE.vue`)
    - [ ] Message Input
    - [ ] Date Picker Widget
    - [ ] Actions: Type, Pick Date, Submit, Back

10. **PROFILE_VIEW** (`src/pages/PROFILE_VIEW.vue`)
    - [ ] Profile Display Card
    - [ ] Actions: Edit, Back to Channel/WS

11. **PROFILE_EDIT** (`src/pages/PROFILE_EDIT.vue`)
    - [ ] Form: Name, Title, Status
    - [ ] Actions: Type, Select Status, Save, Back

12. **SEND_MESSAGE_SUCCESS** (`src/pages/SEND_MESSAGE_SUCCESS.vue`)
    - [ ] Success Message
    - [ ] Actions: Go Home, Back Channel

13. **CREATE_CHANNEL_SUCCESS** (`src/pages/CREATE_CHANNEL_SUCCESS.vue`)
    - [ ] Success Message
    - [ ] Actions: Go Home, Back Channel List

14. **START_DM_SUCCESS** (`src/pages/START_DM_SUCCESS.vue`)
    - [ ] Success Message
    - [ ] Actions: Go Home, Back DM Detail

15. **SCHEDULE_MESSAGE_SUCCESS** (`src/pages/SCHEDULE_MESSAGE_SUCCESS.vue`)
    - [ ] Success Message
    - [ ] Actions: Go Home, Back Channel Detail

16. **UPDATE_PROFILE_SUCCESS** (`src/pages/UPDATE_PROFILE_SUCCESS.vue`)
    - [ ] Success Message
    - [ ] Actions: Go Home, Back Profile View

## Phase 4: Components & Widgets
- [ ] `src/components/widgets/DateTimePicker.vue` (Verify existence)
- [ ] Global Permission Modal (Component)
- [ ] Global Cookie Modal (Component)

## Phase 5: Verification
- [ ] `pnpm i`
- [ ] `pnpm run lint`
- [ ] `pnpm run build`