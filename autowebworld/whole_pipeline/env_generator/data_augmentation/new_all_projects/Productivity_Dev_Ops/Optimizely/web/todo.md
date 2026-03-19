# Development Checklist

## 1. Project Setup
- [x] Add `pinia-plugin-persistedstate` to package.json
- [ ] Configure `src/main.js` with Pinia and persistence plugin
- [ ] Create `src/stores/signature.js` (FSM State)
- [ ] Create `src/stores/data.js` (Mock Data)
- [ ] Create `src/router/index.js` (Routes)

## 2. Components & Widgets
- [ ] `src/components/widgets/DateTimePicker.vue` (Use existing)
- [ ] `src/components/PermissionModal.vue` (Location Permission)
- [ ] `src/components/CookieConsentModal.vue` (Cookie Consent)

## 3. Pages (22 Total)
- [ ] `src/pages/HOME.vue`
- [ ] `src/pages/DASHBOARD.vue`
- [ ] `src/pages/EXPERIMENTS_LIST.vue`
- [ ] `src/pages/EXPERIMENT_DETAIL.vue`
- [ ] `src/pages/EXPERIMENT_CREATE_TYPE.vue`
- [ ] `src/pages/EXPERIMENT_EDIT_VARIATIONS.vue`
- [ ] `src/pages/EXPERIMENT_EDIT_TARGETING.vue`
- [ ] `src/pages/EXPERIMENT_SCHEDULE.vue`
- [ ] `src/pages/EXPERIMENT_LAUNCHED_SUCCESS.vue`
- [ ] `src/pages/EXPERIMENT_SCHEDULED_SUCCESS.vue`
- [ ] `src/pages/EXPERIMENT_ARCHIVE_CONFIRM.vue`
- [ ] `src/pages/EXPERIMENT_ARCHIVED_SUCCESS.vue`
- [ ] `src/pages/AUDIENCES_LIST.vue`
- [ ] `src/pages/AUDIENCE_DETAIL.vue`
- [ ] `src/pages/AUDIENCE_CREATE.vue`
- [ ] `src/pages/AUDIENCE_SAVED_SUCCESS.vue`
- [ ] `src/pages/FEATURE_FLAGS_LIST.vue`
- [ ] `src/pages/FEATURE_FLAG_DETAIL.vue`
- [ ] `src/pages/RESULTS_OVERVIEW.vue`
- [ ] `src/pages/ACCOUNT_SETTINGS.vue`
- [ ] `src/pages/BILLING_SETTINGS.vue`
- [ ] `src/pages/ACCOUNT_BILLING_UPDATED_SUCCESS.vue`

## 4. Verification
- [ ] Verify all GUI selectors exist
- [ ] Verify all actions are implemented
- [ ] Verify z-index for modals