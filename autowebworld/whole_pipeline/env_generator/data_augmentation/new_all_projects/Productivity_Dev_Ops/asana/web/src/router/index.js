import { createRouter, createWebHistory } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

// Import all pages
import HOME from '../pages/HOME.vue'
import PROJECTS_LIST from '../pages/PROJECTS_LIST.vue'
import PROJECT_CREATE_FORM from '../pages/PROJECT_CREATE_FORM.vue'
import PROJECT_CREATE_SUCCESS from '../pages/PROJECT_CREATE_SUCCESS.vue'
import PROJECT_BOARD from '../pages/PROJECT_BOARD.vue'
import TASK_CREATE_FORM from '../pages/TASK_CREATE_FORM.vue'
import TASK_CREATE_SUCCESS from '../pages/TASK_CREATE_SUCCESS.vue'
import SECTION_CREATE_FORM from '../pages/SECTION_CREATE_FORM.vue'
import SECTION_CREATE_SUCCESS from '../pages/SECTION_CREATE_SUCCESS.vue'
import TASK_DETAIL from '../pages/TASK_DETAIL.vue'
import TASK_COMPLETE_SUCCESS from '../pages/TASK_COMPLETE_SUCCESS.vue'
import COMMENT_ADD_SUCCESS from '../pages/COMMENT_ADD_SUCCESS.vue'
import MY_TASKS_LIST from '../pages/MY_TASKS_LIST.vue'

const routes = [
  { path: '/', name: 'HOME', component: HOME },
  { path: '/projects', name: 'PROJECTS_LIST', component: PROJECTS_LIST },
  { path: '/projects/new', name: 'PROJECT_CREATE_FORM', component: PROJECT_CREATE_FORM },
  { path: '/projects/success', name: 'PROJECT_CREATE_SUCCESS', component: PROJECT_CREATE_SUCCESS },
  { path: '/board/:id?', name: 'PROJECT_BOARD', component: PROJECT_BOARD }, // id optional for demo
  { path: '/tasks/new', name: 'TASK_CREATE_FORM', component: TASK_CREATE_FORM },
  { path: '/tasks/success', name: 'TASK_CREATE_SUCCESS', component: TASK_CREATE_SUCCESS },
  { path: '/sections/new', name: 'SECTION_CREATE_FORM', component: SECTION_CREATE_FORM },
  { path: '/sections/success', name: 'SECTION_CREATE_SUCCESS', component: SECTION_CREATE_SUCCESS },
  { path: '/task/:id?', name: 'TASK_DETAIL', component: TASK_DETAIL },
  { path: '/task/complete', name: 'TASK_COMPLETE_SUCCESS', component: TASK_COMPLETE_SUCCESS },
  { path: '/comment/success', name: 'COMMENT_ADD_SUCCESS', component: COMMENT_ADD_SUCCESS },
  { path: '/my-tasks', name: 'MY_TASKS_LIST', component: MY_TASKS_LIST },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

router.beforeEach((to, from, next) => {
  const signatureStore = useSignatureStore()
  const dataStore = useDataStore()
  
  // Always (re)initialize mock data on navigation to avoid stale session artifacts
  dataStore.initializeMockData()

  signatureStore.setCurrentPageId(to.name)
  next()
})

export default router
