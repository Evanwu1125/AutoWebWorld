# Development Checklist

## 1. Project Setup
- [ ] Add `pinia-plugin-persistedstate` to package.json
- [ ] Update `src/main.js` to use persistence plugin
- [ ] Configure `src/router/index.js` with all 26 routes
- [ ] Create `src/stores/signature.js` (Pinia store for FSM state)
- [ ] Create `src/stores/data.js` (Mock data with realistic content)

## 2. Components
- [ ] `src/components/widgets/DateTimePicker.vue` (Verify existence/usage)
- [ ] `src/components/CookieConsentModal.vue` (For HOME page)
- [ ] `src/components/PermissionModal.vue` (For Location Permission on NEWS_FEED)
- [ ] `src/components/NavBar.vue` (Shared navigation component, if applicable)

## 3. Pages (Vue Components)
- [ ] `src/pages/HOME.vue`
- [ ] `src/pages/NEWS_FEED.vue`
- [ ] `src/pages/CREATE_POST.vue`
- [ ] `src/pages/CREATE_POST_REVIEW.vue`
- [ ] `src/pages/POST_DETAIL.vue`
- [ ] `src/pages/POST_PUBLISH_SUCCESS.vue`
- [ ] `src/pages/FRIENDS_LIST.vue`
- [ ] `src/pages/FRIEND_SUGGESTIONS.vue`
- [ ] `src/pages/PROFILE_TIMELINE.vue`
- [ ] `src/pages/PROFILE_ABOUT.vue`
- [ ] `src/pages/FRIEND_REQUEST_CONFIRM.vue`
- [ ] `src/pages/FRIEND_REQUEST_SENT_SUCCESS.vue`
- [ ] `src/pages/MESSENGER_INBOX.vue`
- [ ] `src/pages/MESSAGE_THREAD.vue`
- [ ] `src/pages/MESSAGE_COMPOSE.vue`
- [ ] `src/pages/MESSAGE_REVIEW.vue`
- [ ] `src/pages/MESSAGE_SEND_SUCCESS.vue`
- [ ] `src/pages/EVENTS_LIST.vue`
- [ ] `src/pages/EVENT_DETAIL.vue`
- [ ] `src/pages/CREATE_EVENT_DETAILS.vue`
- [ ] `src/pages/CREATE_EVENT_DATE.vue`
- [ ] `src/pages/CREATE_EVENT_REVIEW.vue`
- [ ] `src/pages/EVENT_CREATE_SUCCESS.vue`
- [ ] `src/pages/SETTINGS_ACCOUNT.vue`
- [ ] `src/pages/SETTINGS_ACCOUNT_REVIEW.vue`
- [ ] `src/pages/ACCOUNT_SETTINGS_SAVED_SUCCESS.vue`

## 4. Logic & Integration
- [ ] Implement FSM logic in `signature.js` (actions, effects)
- [ ] Verify JSONPath integration
- [ ] Test Navigation Guard in Router
- [ ] Verify Interceptors (Cookie, Location)

## 5. Validation
- [ ] Verify all selectors match `gui_procedure`
- [ ] Run Lint & Build