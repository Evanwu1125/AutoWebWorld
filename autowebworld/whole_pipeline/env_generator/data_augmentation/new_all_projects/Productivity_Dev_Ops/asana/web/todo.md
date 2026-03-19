# Development Checklist

## 1. Project Setup
- [ ] Update `package.json` (add `pinia-plugin-persistedstate`).
- [ ] Configure `src/main.js` (Pinia + Persistence).
- [ ] Create `src/router/index.js` (Routes for all pages).
- [ ] Create `src/stores/signature.js` (FSM State).
- [ ] Create `src/stores/data.js` (Mock Data).

## 2. Pages (FSM Implementation)
- [ ] `src/pages/HOME.vue`
  - [ ] Cookie Consent Modal (Interceptor)
  - [ ] Navigation Actions (Direct, Hover, Menu)
- [ ] `src/pages/PROJECTS_LIST.vue`
  - [ ] Location Permission Modal (Interceptor)
  - [ ] Filter by Status (Checkbox)
  - [ ] Filter by Priority (Slider)
  - [ ] Sort Dropdown
  - [ ] Search & Open Project
  - [ ] Create Project Button
- [ ] `src/pages/PROJECT_CREATE_FORM.vue`
  - [ ] Inputs: Name, Description
  - [ ] Date Picker
  - [ ] Submit & Back
- [ ] `src/pages/PROJECT_CREATE_SUCCESS.vue` (Terminal)
- [ ] `src/pages/PROJECT_BOARD.vue`
  - [ ] Board Filters (Assignee, Priority)
  - [ ] Sort Tasks
  - [ ] Search & Open Task
  - [ ] Add Task / Add Section Buttons
- [ ] `src/pages/TASK_CREATE_FORM.vue`
  - [ ] Inputs: Name, Description
  - [ ] Date Picker
  - [ ] Assignee Dropdown
  - [ ] Submit & Back
- [ ] `src/pages/TASK_CREATE_SUCCESS.vue` (Terminal)
- [ ] `src/pages/SECTION_CREATE_FORM.vue`
  - [ ] Input: Name
  - [ ] Submit & Back
- [ ] `src/pages/SECTION_CREATE_SUCCESS.vue` (Terminal)
- [ ] `src/pages/TASK_DETAIL.vue`
  - [ ] Mark Complete Checkbox
  - [ ] Comment Input & Submit
  - [ ] Back to Board
- [ ] `src/pages/TASK_COMPLETE_SUCCESS.vue` (Terminal)
- [ ] `src/pages/COMMENT_ADD_SUCCESS.vue` (Terminal)
- [ ] `src/pages/MY_TASKS_LIST.vue`
  - [ ] Filter Today (Checkbox)
  - [ ] Filter Priority (Slider)
  - [ ] Sort Dropdown
  - [ ] Search & Open Task

## 3. Components
- [ ] `src/components/widgets/DateTimePicker.vue` (Use existing)
- [ ] `src/components/PermissionModal.vue`
- [ ] `src/components/CookieConsentModal.vue`

## 4. Verification
- [ ] Verify all selectors match FSM `gui_procedure`.
- [ ] Verify mock data quantity & quality.
- [ ] Verify builds.