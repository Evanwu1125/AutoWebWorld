import { createRouter, createWebHistory } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

// Import Pages
import HOME from '../pages/HOME.vue'
import BASES_DASHBOARD from '../pages/BASES_DASHBOARD.vue'
import BASE_CREATE from '../pages/BASE_CREATE.vue'
import BASE_CREATED_SUCCESS from '../pages/BASE_CREATED_SUCCESS.vue'
import BASE_WORKSPACE from '../pages/BASE_WORKSPACE.vue'
import TABLE_GRID_VIEW from '../pages/TABLE_GRID_VIEW.vue'
import RECORD_DETAIL from '../pages/RECORD_DETAIL.vue'
import RECORD_CREATE_FORM from '../pages/RECORD_CREATE_FORM.vue'
import RECORD_CREATED_SUCCESS from '../pages/RECORD_CREATED_SUCCESS.vue'
import RECORD_EDIT_FORM from '../pages/RECORD_EDIT_FORM.vue'
import RECORD_UPDATED_SUCCESS from '../pages/RECORD_UPDATED_SUCCESS.vue'
import KANBAN_VIEW from '../pages/KANBAN_VIEW.vue'
import AUTOMATIONS_DASHBOARD from '../pages/AUTOMATIONS_DASHBOARD.vue'
import AUTOMATION_CREATE_TRIGGER from '../pages/AUTOMATION_CREATE_TRIGGER.vue'
import AUTOMATION_CREATE_ACTION from '../pages/AUTOMATION_CREATE_ACTION.vue'
import AUTOMATION_CREATED_SUCCESS from '../pages/AUTOMATION_CREATED_SUCCESS.vue'
import FORM_VIEW_SUBMISSION from '../pages/FORM_VIEW_SUBMISSION.vue'
import FORM_SUBMISSION_SUCCESS from '../pages/FORM_SUBMISSION_SUCCESS.vue'

const routes = [
  {
    path: '/',
    name: 'HOME',
    component: HOME
  },
  {
    path: '/bases',
    name: 'BASES_DASHBOARD',
    component: BASES_DASHBOARD
  },
  {
    path: '/base/create',
    name: 'BASE_CREATE',
    component: BASE_CREATE
  },
  {
    path: '/base/create/success',
    name: 'BASE_CREATED_SUCCESS',
    component: BASE_CREATED_SUCCESS
  },
  {
    path: '/base/workspace',
    name: 'BASE_WORKSPACE',
    component: BASE_WORKSPACE
  },
  {
    path: '/base/grid',
    name: 'TABLE_GRID_VIEW',
    component: TABLE_GRID_VIEW
  },
  {
    path: '/record/detail',
    name: 'RECORD_DETAIL',
    component: RECORD_DETAIL
  },
  {
    path: '/record/create',
    name: 'RECORD_CREATE_FORM',
    component: RECORD_CREATE_FORM
  },
  {
    path: '/record/create/success',
    name: 'RECORD_CREATED_SUCCESS',
    component: RECORD_CREATED_SUCCESS
  },
  {
    path: '/record/edit',
    name: 'RECORD_EDIT_FORM',
    component: RECORD_EDIT_FORM
  },
  {
    path: '/record/edit/success',
    name: 'RECORD_UPDATED_SUCCESS',
    component: RECORD_UPDATED_SUCCESS
  },
  {
    path: '/base/kanban',
    name: 'KANBAN_VIEW',
    component: KANBAN_VIEW
  },
  {
    path: '/automations',
    name: 'AUTOMATIONS_DASHBOARD',
    component: AUTOMATIONS_DASHBOARD
  },
  {
    path: '/automation/create/trigger',
    name: 'AUTOMATION_CREATE_TRIGGER',
    component: AUTOMATION_CREATE_TRIGGER
  },
  {
    path: '/automation/create/action',
    name: 'AUTOMATION_CREATE_ACTION',
    component: AUTOMATION_CREATE_ACTION
  },
  {
    path: '/automation/create/success',
    name: 'AUTOMATION_CREATED_SUCCESS',
    component: AUTOMATION_CREATED_SUCCESS
  },
  {
    path: '/form/view',
    name: 'FORM_VIEW_SUBMISSION',
    component: FORM_VIEW_SUBMISSION
  },
  {
    path: '/form/success',
    name: 'FORM_SUBMISSION_SUCCESS',
    component: FORM_SUBMISSION_SUCCESS
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach((to, from, next) => {
  const store = useSignatureStore()
  // Ensure we track current page ID in store for FSM logic
  if (to.name) {
    store.setCurrentPageId(to.name)
  }
  next()
})

export default router