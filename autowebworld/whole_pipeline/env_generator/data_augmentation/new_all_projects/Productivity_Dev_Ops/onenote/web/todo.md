# Development Checklist

## Phase 1: Infrastructure
- [ ] Install dependencies (pinia-plugin-persistedstate)
- [ ] Configure `src/main.js` with Pinia persister
- [ ] Create `src/stores/signature.js` (FSM State)
- [ ] Create `src/stores/data.js` (Mock Data - CRITICAL: 15+ items per list, real images)
- [ ] Create `src/router/index.js` (Route definitions)
- [ ] Implement `src/components/CookieConsentModal.vue`
- [ ] Implement `src/components/PermissionModal.vue` (Location)

## Phase 2: Pages Implementation (19 Pages)
- [ ] `HOME`
- [ ] `NOTEBOOK_LIST` (Requires Location Permission)
- [ ] `NOTEBOOK_CREATE`
- [ ] `SECTION_LIST`
- [ ] `SECTION_CREATE`
- [ ] `PAGE_LIST`
- [ ] `NOTE_EDITOR`
- [ ] `NOTE_REVIEW`
- [ ] `NOTE_SHARE`
- [ ] `NOTE_DELETE_CONFIRM`
- [ ] `RECENT_NOTES`
- [ ] `QUICK_NOTES`
- [ ] `SETTINGS`
- [ ] `NOTE_CREATE_SUCCESS`
- [ ] `NOTE_UPDATE_SUCCESS`
- [ ] `SECTION_CREATE_SUCCESS`
- [ ] `NOTE_SHARE_SUCCESS`
- [ ] `NOTE_DELETE_SUCCESS`
- [ ] `sign_up_new_account_success`

## Phase 3: Validation
- [ ] Verify all selectors exist
- [ ] Verify image paths (mock data uses /images/ prefix)
- [ ] Verify FSM logic (preconditions/effects mapped to store)
- [ ] Build and Lint