# Development Checklist

## 1. Project Setup
- [ ] Install dependencies (pinia-plugin-persistedstate)
- [ ] Setup Tailwind config (already in template, ensure content paths)

## 2. State Management (Pinia)
- [ ] `src/stores/signature.js`: FSM state properties (25+ fields), actions for updating state.
- [ ] `src/stores/data.js`: Mock data for Chats, Contacts, Groups, Calls, Messages.
  - [ ] Generate 15+ Contacts
  - [ ] Generate 15+ Chats (Recent conversations)
  - [ ] Generate 15+ Groups
  - [ ] Generate 20+ Call History items
  - [ ] Generate Message history for active chats

## 3. Router Configuration
- [ ] `src/router/index.js`: Define routes for all 25 pages.
- [ ] Add Navigation Guard to update `currentPageId` in signature store.

## 4. Page Implementation (25 Pages)
### Home & Navigation
- [ ] `src/pages/HOME.vue`
- [ ] `src/pages/CHATS_LIST.vue`
- [ ] `src/pages/CONTACTS_LIST.vue`
- [ ] `src/pages/GROUPS_LIST.vue`
- [ ] `src/pages/CALL_HISTORY.vue`

### Chat & Messaging
- [ ] `src/pages/CHAT_THREAD.vue`
- [ ] `src/pages/SEND_MESSAGE_CONFIRM.vue`
- [ ] `src/pages/SEND_MESSAGE_SUCCESS.vue`
- [ ] `src/pages/CHAT_INFO.vue`
- [ ] `src/pages/DISAPPEARING_MESSAGES_SETTINGS.vue`

### New Chat Flow
- [ ] `src/pages/NEW_CHAT_CHOOSE_CONTACT.vue`
- [ ] `src/pages/NEW_CHAT_COMPOSE.vue`

### Contact Management
- [ ] `src/pages/CONTACT_DETAIL.vue`
- [ ] `src/pages/BLOCK_USER_CONFIRM.vue`
- [ ] `src/pages/BLOCK_USER_SUCCESS.vue`

### Group Management
- [ ] `src/pages/GROUP_DETAIL.vue`
- [ ] `src/pages/GROUP_CREATE_DETAILS.vue`
- [ ] `src/pages/GROUP_CREATE_ADD_MEMBERS.vue`
- [ ] `src/pages/GROUP_CREATE_REVIEW.vue`
- [ ] `src/pages/CREATE_GROUP_SUCCESS.vue`

### Calling
- [ ] `src/pages/START_CALL_SETUP.vue`
- [ ] `src/pages/START_CALL_SUCCESS.vue`

### Settings
- [ ] `src/pages/SETTINGS_PRIVACY.vue`
- [ ] `src/pages/SETTINGS_NOTIFICATIONS.vue`
- [ ] `src/pages/UPDATE_SETTINGS_SUCCESS.vue`

## 5. Components & Logic
- [ ] `src/components/PermissionModal.vue`: For Location permission.
- [ ] `src/components/CookieConsentModal.vue`: For Cookie consent.
- [ ] Implement FSM Logic (Effect application, Precondition checking) in Store or Composables.

## 6. Validation
- [ ] Verify all selectors match `fsm.json`.
- [ ] Verify all actions are implemented.
- [ ] Run Lint & Build.