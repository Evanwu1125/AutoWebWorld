# Development Checklist

## Phase 1: Infrastructure
- [ ] Update package.json (add pinia-plugin-persistedstate)
- [ ] Configure main.js
- [ ] Setup Router (src/router/index.js)
- [ ] Setup Signature Store (src/stores/signature.js)
- [ ] Setup Mock Data Store (src/stores/data.js) - HEAVY DATA GENERATION
- [ ] Create basic FSM Runtime helper

## Phase 2: Page Implementation (21 Pages)
- [ ] HOME (Cookie Consent Interceptor)
- [ ] SIGNUP
- [ ] DASHBOARD_FEED (Location Permission Interceptor, Feed List, Search, Filter)
- [ ] EXPLORE (Masonry Grid, Search, Filter)
- [ ] BLOG_OVERVIEW
- [ ] BLOG_POSTS_LIST
- [ ] BLOG_INFO
- [ ] FOLLOW_BLOG_CONFIRM
- [ ] FOLLOW_BLOG_SUCCESS
- [ ] POST_DETAIL
- [ ] REBLOG_FORM
- [ ] COMPOSE_TEXT_POST
- [ ] SCHEDULE_POST
- [ ] POST_PUBLISH_SUCCESS
- [ ] POST_SCHEDULE_SUCCESS
- [ ] MESSAGES_INBOX
- [ ] MESSAGE_THREAD
- [ ] MESSAGE_COMPOSE
- [ ] MESSAGE_SEND_SUCCESS
- [ ] ACCOUNT_SETTINGS
- [ ] ACCOUNT_SETTINGS_SAVE_SUCCESS

## Phase 3: Components
- [ ] DateTimePicker (Verify template existence)
- [ ] PermissionModal (Location)
- [ ] CookieConsentModal
- [ ] Common Layout Components (Sidebar, Header)

## Phase 4: Validation
- [ ] Verify all FSM actions map to UI
- [ ] Verify all selectors exist
- [ ] Lint & Build