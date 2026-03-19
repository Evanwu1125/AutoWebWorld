# Development Checklist

## 1. Project Setup
- [ ] Update package.json (add pinia-plugin-persistedstate)
- [ ] Create src/fsm/FSMRuntime.js (FSM Engine)
- [ ] Create src/stores/signature.js (State Management)
- [ ] Create src/stores/data.js (Mock Data with extensive items)
- [ ] Setup src/router/index.js (Routes for all 19 pages)
- [ ] Create src/components/PermissionModal.vue (Location Permission)
- [ ] Create src/components/CookieConsentModal.vue (Cookie Consent)

## 2. Page Implementation (19 Pages)
- [ ] src/pages/HOME.vue
- [ ] src/pages/DASHBOARD.vue
- [ ] src/pages/SCHEDULE_MEETING_FORM.vue
- [ ] src/pages/SCHEDULE_MEETING_REVIEW.vue
- [ ] src/pages/SCHEDULE_MEETING_SUCCESS.vue
- [ ] src/pages/JOIN_MEETING_FORM.vue
- [ ] src/pages/JOIN_MEETING_PREVIEW.vue
- [ ] src/pages/JOIN_MEETING_SUCCESS.vue
- [ ] src/pages/INSTANT_MEETING_LOBBY.vue
- [ ] src/pages/START_INSTANT_MEETING_SUCCESS.vue
- [ ] src/pages/PROFILE_OVERVIEW.vue
- [ ] src/pages/PROFILE_RENAME_FORM.vue
- [ ] src/pages/RENAME_PROFILE_SUCCESS.vue
- [ ] src/pages/PROFILE_CHANGE_PASSWORD_FORM.vue
- [ ] src/pages/CHANGE_PASSWORD_SUCCESS.vue
- [ ] src/pages/SETTINGS_GENERAL.vue
- [ ] src/pages/SETTINGS_VIDEO.vue
- [ ] src/pages/MEETINGS_LIST.vue
- [ ] src/pages/MEETING_DETAIL.vue

## 3. Validation & Build
- [ ] Verify all GUI selectors exist in DOM
- [ ] Verify all signature fields are mapped
- [ ] Run pnpm install && pnpm run lint && pnpm run build