# Development Checklist

## Phase 1: Setup & Config
- [ ] Update package.json with pinia-plugin-persistedstate
- [ ] Configure Pinia persistence in src/main.js
- [ ] Implement src/fsm/FSMRuntime.js
- [ ] Implement src/stores/signature.js (FSM State)
- [ ] Implement src/stores/data.js (Mock Data with relationships)
- [ ] Setup Router in src/router/index.js

## Phase 2: Components & Modals
- [ ] src/components/CookieConsentModal.vue (Home page)
- [ ] src/components/PermissionModal.vue (Location permission)
- [ ] Verify src/components/widgets/DateTimePicker.vue exists

## Phase 3: Pages Implementation
### Core
- [ ] src/pages/HOME.vue

### Repositories
- [ ] src/pages/REPOSITORIES_LIST.vue
- [ ] src/pages/REPOSITORY_DETAIL.vue
- [ ] src/pages/NEW_REPOSITORY.vue
- [ ] src/pages/REPO_CREATE_SUCCESS.vue

### Issues
- [ ] src/pages/ISSUES_LIST.vue
- [ ] src/pages/ISSUE_DETAIL.vue
- [ ] src/pages/NEW_ISSUE.vue
- [ ] src/pages/ISSUE_CREATE_SUCCESS.vue

### Pull Requests
- [ ] src/pages/PULL_REQUESTS_LIST.vue
- [ ] src/pages/PULL_REQUEST_DETAIL.vue
- [ ] src/pages/NEW_PULL_REQUEST.vue
- [ ] src/pages/PR_CREATE_SUCCESS.vue

### Branches
- [ ] src/pages/BRANCHES_LIST.vue
- [ ] src/pages/NEW_BRANCH.vue
- [ ] src/pages/NEW_BRANCH_SUCCESS.vue
- [ ] src/pages/COMPARE_BRANCHES.vue

### Profile
- [ ] src/pages/PROFILE_OVERVIEW.vue
- [ ] src/pages/PROFILE_SETTINGS.vue
- [ ] src/pages/PROFILE_UPDATE_SUCCESS.vue
- [ ] src/pages/PROFILE_FOLLOWERS.vue

## Phase 4: Validation
- [ ] Verify all 21 pages exist
- [ ] Verify all actions mapped
- [ ] Verify all selectors exist
- [ ] Build and Lint