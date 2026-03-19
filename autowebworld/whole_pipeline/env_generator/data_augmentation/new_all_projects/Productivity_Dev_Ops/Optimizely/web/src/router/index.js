import { createRouter, createWebHistory } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

const routes = [
  {
    path: '/',
    name: 'HOME',
    component: () => import('../pages/HOME.vue')
  },
  {
    path: '/dashboard',
    name: 'DASHBOARD',
    component: () => import('../pages/DASHBOARD.vue')
  },
  {
    path: '/experiments',
    name: 'EXPERIMENTS_LIST',
    component: () => import('../pages/EXPERIMENTS_LIST.vue')
  },
  {
    path: '/experiments/:id',
    name: 'EXPERIMENT_DETAIL',
    component: () => import('../pages/EXPERIMENT_DETAIL.vue')
  },
  {
    path: '/experiments/create',
    name: 'EXPERIMENT_CREATE_TYPE',
    component: () => import('../pages/EXPERIMENT_CREATE_TYPE.vue')
  },
  {
    path: '/experiments/edit/variations',
    name: 'EXPERIMENT_EDIT_VARIATIONS',
    component: () => import('../pages/EXPERIMENT_EDIT_VARIATIONS.vue')
  },
  {
    path: '/experiments/edit/targeting',
    name: 'EXPERIMENT_EDIT_TARGETING',
    component: () => import('../pages/EXPERIMENT_EDIT_TARGETING.vue')
  },
  {
    path: '/experiments/schedule',
    name: 'EXPERIMENT_SCHEDULE',
    component: () => import('../pages/EXPERIMENT_SCHEDULE.vue')
  },
  {
    path: '/experiments/success/launched',
    name: 'EXPERIMENT_LAUNCHED_SUCCESS',
    component: () => import('../pages/EXPERIMENT_LAUNCHED_SUCCESS.vue')
  },
  {
    path: '/experiments/success/scheduled',
    name: 'EXPERIMENT_SCHEDULED_SUCCESS',
    component: () => import('../pages/EXPERIMENT_SCHEDULED_SUCCESS.vue')
  },
  {
    path: '/experiments/archive',
    name: 'EXPERIMENT_ARCHIVE_CONFIRM',
    component: () => import('../pages/EXPERIMENT_ARCHIVE_CONFIRM.vue')
  },
  {
    path: '/experiments/success/archived',
    name: 'EXPERIMENT_ARCHIVED_SUCCESS',
    component: () => import('../pages/EXPERIMENT_ARCHIVED_SUCCESS.vue')
  },
  {
    path: '/audiences',
    name: 'AUDIENCES_LIST',
    component: () => import('../pages/AUDIENCES_LIST.vue')
  },
  {
    path: '/audiences/:id',
    name: 'AUDIENCE_DETAIL',
    component: () => import('../pages/AUDIENCE_DETAIL.vue')
  },
  {
    path: '/audiences/create',
    name: 'AUDIENCE_CREATE',
    component: () => import('../pages/AUDIENCE_CREATE.vue')
  },
  {
    path: '/audiences/success',
    name: 'AUDIENCE_SAVED_SUCCESS',
    component: () => import('../pages/AUDIENCE_SAVED_SUCCESS.vue')
  },
  {
    path: '/feature-flags',
    name: 'FEATURE_FLAGS_LIST',
    component: () => import('../pages/FEATURE_FLAGS_LIST.vue')
  },
  {
    path: '/feature-flags/:id',
    name: 'FEATURE_FLAG_DETAIL',
    component: () => import('../pages/FEATURE_FLAG_DETAIL.vue')
  },
  {
    path: '/results',
    name: 'RESULTS_OVERVIEW',
    component: () => import('../pages/RESULTS_OVERVIEW.vue')
  },
  {
    path: '/account',
    name: 'ACCOUNT_SETTINGS',
    component: () => import('../pages/ACCOUNT_SETTINGS.vue')
  },
  {
    path: '/billing',
    name: 'BILLING_SETTINGS',
    component: () => import('../pages/BILLING_SETTINGS.vue')
  },
  {
    path: '/account/billing-success',
    name: 'ACCOUNT_BILLING_UPDATED_SUCCESS',
    component: () => import('../pages/ACCOUNT_BILLING_UPDATED_SUCCESS.vue')
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach((to, from, next) => {
  const signatureStore = useSignatureStore()
  signatureStore.setCurrentPageId(to.name)
  next()
})

export default router