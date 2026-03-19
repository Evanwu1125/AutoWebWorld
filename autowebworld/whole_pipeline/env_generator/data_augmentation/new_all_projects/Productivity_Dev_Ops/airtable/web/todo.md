# Development Checklist

## Phase 1: Infrastructure & Stores
- [ ] `src/stores/signature.js` - Pinia store implementing FSM signature schema and state logic.
- [ ] `src/stores/data.js` - Mock data store (Bases, Tables, Records, Automations, Forms).
- [ ] `src/router/index.js` - Vue Router configuration for all 18 pages.
- [ ] `src/components/CookieConsentModal.vue` - Global cookie consent modal.
- [ ] `src/components/PermissionModal.vue` - Global location permission modal.

## Phase 2: Page Implementation (18 Pages)

### Home & Onboarding
- [ ] `src/pages/HOME.vue`
    - Actions: `ACT_HOME_ACCEPT_COOKIES` (via global modal logic), `ACT_HOME_OPEN_BASES_DIRECT`, `ACT_HOME_OPEN_BASES_HOVER`, `ACT_HOME_OPEN_BASES_MENU`.

### Bases Dashboard
- [ ] `src/pages/BASES_DASHBOARD.vue`
    - Signature: `location_permission_granted` (Permission Modal trigger).
    - Actions: `ACT_BASES_GRANT_LOCATION`, `ACT_BASES_FILTER_*`, `ACT_BASES_SORT`, `ACT_BASES_OPEN_*`, `ACT_BASES_SCROLL_*`, `ACT_BASES_SEARCH_*`, `ACT_BASES_GO_BACK_HOME`, `ACT_BASES_GO_CREATE_BASE`.

### Base Creation
- [ ] `src/pages/BASE_CREATE.vue`
    - Form flow for creating a base.
- [ ] `src/pages/BASE_CREATED_SUCCESS.vue`
    - Success confirmation.

### Base Workspace & Views
- [ ] `src/pages/BASE_WORKSPACE.vue`
    - Navigation hub for a specific base (Grid vs Kanban vs Automations).
- [ ] `src/pages/TABLE_GRID_VIEW.vue`
    - Main spreadsheet view.
    - Actions: `ACT_GRID_FILTER_*`, `ACT_GRID_SORT_*`, `ACT_GRID_OPEN_*`, `ACT_GRID_SCROLL_*`, `ACT_GRID_SEARCH_*`, `ACT_GRID_CREATE_RECORD`.
- [ ] `src/pages/KANBAN_VIEW.vue`
    - Kanban board view.
    - Actions: `ACT_KANBAN_FILTER_*`, `ACT_KANBAN_SCROLL_*`, `ACT_KANBAN_OPEN_*`.

### Record Operations
- [ ] `src/pages/RECORD_DETAIL.vue`
    - View record, nav to Edit or Form View.
- [ ] `src/pages/RECORD_CREATE_FORM.vue`
    - Create new record form.
- [ ] `src/pages/RECORD_CREATED_SUCCESS.vue`
    - Success state.
- [ ] `src/pages/RECORD_EDIT_FORM.vue`
    - Edit existing record.
- [ ] `src/pages/RECORD_UPDATED_SUCCESS.vue`
    - Success state.

### Automations
- [ ] `src/pages/AUTOMATIONS_DASHBOARD.vue`
    - List automations.
- [ ] `src/pages/AUTOMATION_CREATE_TRIGGER.vue`
    - Step 1: Trigger selection.
- [ ] `src/pages/AUTOMATION_CREATE_ACTION.vue`
    - Step 2: Action configuration.
- [ ] `src/pages/AUTOMATION_CREATED_SUCCESS.vue`
    - Success state.

### Forms
- [ ] `src/pages/FORM_VIEW_SUBMISSION.vue`
    - Public-facing form view.
- [ ] `src/pages/FORM_SUBMISSION_SUCCESS.vue`
    - Success state.

## Phase 3: Validation
- [ ] Verify all FSM actions are mapped to UI elements.
- [ ] Verify all selectors (`#id`, `.class`) exist.
- [ ] Run lint and build.