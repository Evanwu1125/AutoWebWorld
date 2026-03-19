# Development Checklist

## 1. Infrastructure & Setup
- [ ] `src/main.js`: Setup Pinia, PersistedState, Router, Tailwind.
- [ ] `src/stores/signature.js`: Implement FSM state (Pinia + persist).
- [ ] `src/stores/data.js`: Implement Mock Data (Pinia + persist).
- [ ] `src/router/index.js`: Define routes for all 21 pages + Guards.
- [ ] `src/App.vue`: Root component with RouterView + Global Modals (Cookie, Permission).

## 2. Global Components
- [ ] `src/components/PermissionModal.vue`: For Location Permission.
- [ ] `src/components/CookieConsentModal.vue`: For Cookie Consent (HOME).
- [ ] `src/components/NavBar.vue`: Global navigation (optional, can be part of page layouts if FSM differs).
- [ ] `src/components/widgets/DateTimePicker.vue`: (Already exists, verify).

## 3. Pages (Implementation of FSM)
- [ ] `src/pages/HOME.vue`
- [ ] `src/pages/POST_LIST.vue`
- [ ] `src/pages/POST_DETAIL.vue`
- [ ] `src/pages/COMMENT_FORM.vue`
- [ ] `src/pages/COMMENT_SUBMIT_SUCCESS.vue`
- [ ] `src/pages/NEW_STORY_EDITOR.vue`
- [ ] `src/pages/PUBLISH_OPTIONS.vue`
- [ ] `src/pages/PUBLISH_CONFIRM.vue`
- [ ] `src/pages/SCHEDULE_PICKER.vue`
- [ ] `src/pages/PUBLISH_POST_SUCCESS.vue`
- [ ] `src/pages/SCHEDULE_POST_SUCCESS.vue`
- [ ] `src/pages/PROFILE_OVERVIEW.vue`
- [ ] `src/pages/PROFILE_EDIT.vue`
- [ ] `src/pages/PROFILE_UPDATE_SUCCESS.vue`
- [ ] `src/pages/STORIES_DRAFTS.vue`
- [ ] `src/pages/PUBLICATION_LIST.vue`
- [ ] `src/pages/PUBLICATION_DETAIL.vue`
- [ ] `src/pages/SETTINGS_PREFERENCES.vue`
- [ ] `src/pages/MEMBERSHIP_LANDING.vue`
- [ ] `src/pages/PAYMENT_DETAILS.vue`
- [ ] `src/pages/SUBSCRIPTION_SUCCESS.vue`

## 4. Verification
- [ ] Verify all actions map to UI.
- [ ] Verify all selectors exist.
- [ ] Verify data loading and interactions.
- [ ] Run lint and build.