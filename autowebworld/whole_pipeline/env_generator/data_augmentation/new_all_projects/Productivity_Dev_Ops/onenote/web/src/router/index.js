import { createRouter, createWebHistory } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

const routes = [
  { path: '/', name: 'HOME', component: () => import('../pages/HOME.vue') },
  { path: '/notebooks', name: 'NOTEBOOK_LIST', component: () => import('../pages/NOTEBOOK_LIST.vue') },
  { path: '/notebooks/create', name: 'NOTEBOOK_CREATE', component: () => import('../pages/NOTEBOOK_CREATE.vue') },
  { path: '/sections', name: 'SECTION_LIST', component: () => import('../pages/SECTION_LIST.vue') },
  { path: '/sections/create', name: 'SECTION_CREATE', component: () => import('../pages/SECTION_CREATE.vue') },
  { path: '/pages', name: 'PAGE_LIST', component: () => import('../pages/PAGE_LIST.vue') },
  { path: '/editor', name: 'NOTE_EDITOR', component: () => import('../pages/NOTE_EDITOR.vue') },
  { path: '/review', name: 'NOTE_REVIEW', component: () => import('../pages/NOTE_REVIEW.vue') },
  { path: '/share', name: 'NOTE_SHARE', component: () => import('../pages/NOTE_SHARE.vue') },
  { path: '/delete-confirm', name: 'NOTE_DELETE_CONFIRM', component: () => import('../pages/NOTE_DELETE_CONFIRM.vue') },
  { path: '/recent', name: 'RECENT_NOTES', component: () => import('../pages/RECENT_NOTES.vue') },
  { path: '/quick', name: 'QUICK_NOTES', component: () => import('../pages/QUICK_NOTES.vue') },
  { path: '/settings', name: 'SETTINGS', component: () => import('../pages/SETTINGS.vue') },
  { path: '/success/note-create', name: 'NOTE_CREATE_SUCCESS', component: () => import('../pages/NOTE_CREATE_SUCCESS.vue') },
  { path: '/success/note-update', name: 'NOTE_UPDATE_SUCCESS', component: () => import('../pages/NOTE_UPDATE_SUCCESS.vue') },
  { path: '/success/section-create', name: 'SECTION_CREATE_SUCCESS', component: () => import('../pages/SECTION_CREATE_SUCCESS.vue') },
  { path: '/success/note-share', name: 'NOTE_SHARE_SUCCESS', component: () => import('../pages/NOTE_SHARE_SUCCESS.vue') },
  { path: '/success/note-delete', name: 'NOTE_DELETE_SUCCESS', component: () => import('../pages/NOTE_DELETE_SUCCESS.vue') },
  { path: '/success/sign-up', name: 'sign_up_new_account_success', component: () => import('../pages/sign_up_new_account_success.vue') },
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach((to, from, next) => {
  const store = useSignatureStore()
  store.setCurrentPageId(to.name)
  next()
})

export default router