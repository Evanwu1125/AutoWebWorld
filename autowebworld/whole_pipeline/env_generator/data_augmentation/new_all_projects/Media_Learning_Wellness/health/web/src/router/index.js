import { createRouter, createWebHistory } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

// Lazy load components
const pages = {
  HOME: () => import('../pages/HOME.vue'),
  LOGIN: () => import('../pages/LOGIN.vue'),
  DASHBOARD: () => import('../pages/DASHBOARD.vue'),
  VISIT_TYPE_SELECTION: () => import('../pages/VISIT_TYPE_SELECTION.vue'),
  PROVIDER_LIST: () => import('../pages/PROVIDER_LIST.vue'),
  PROVIDER_DETAIL: () => import('../pages/PROVIDER_DETAIL.vue'),
  SCHEDULE_APPOINTMENT: () => import('../pages/SCHEDULE_APPOINTMENT.vue'),
  SCHEDULE_REVIEW: () => import('../pages/SCHEDULE_REVIEW.vue'),
  SCHEDULE_VISIT_SUCCESS: () => import('../pages/SCHEDULE_VISIT_SUCCESS.vue'),
  INSTANT_VISIT_TRIAGE: () => import('../pages/INSTANT_VISIT_TRIAGE.vue'),
  INSTANT_VISIT_QUEUE: () => import('../pages/INSTANT_VISIT_QUEUE.vue'),
  INSTANT_VISIT_SUCCESS: () => import('../pages/INSTANT_VISIT_SUCCESS.vue'),
  PRESCRIPTION_LIST: () => import('../pages/PRESCRIPTION_LIST.vue'),
  PRESCRIPTION_DETAIL: () => import('../pages/PRESCRIPTION_DETAIL.vue'),
  PRESCRIPTION_RENEWAL_REVIEW: () => import('../pages/PRESCRIPTION_RENEWAL_REVIEW.vue'),
  PRESCRIPTION_RENEWAL_SUCCESS: () => import('../pages/PRESCRIPTION_RENEWAL_SUCCESS.vue'),
  MENTAL_HEALTH_LIST: () => import('../pages/MENTAL_HEALTH_LIST.vue'),
  MENTAL_HEALTH_DETAIL: () => import('../pages/MENTAL_HEALTH_DETAIL.vue'),
  MENTAL_HEALTH_SCHEDULE: () => import('../pages/MENTAL_HEALTH_SCHEDULE.vue'),
  MENTAL_HEALTH_REVIEW: () => import('../pages/MENTAL_HEALTH_REVIEW.vue'),
  MENTAL_HEALTH_BOOKING_SUCCESS: () => import('../pages/MENTAL_HEALTH_BOOKING_SUCCESS.vue'),
  APPOINTMENTS_LIST: () => import('../pages/APPOINTMENTS_LIST.vue'),
  APPOINTMENT_DETAIL: () => import('../pages/APPOINTMENT_DETAIL.vue'),
  BILLING_OVERVIEW: () => import('../pages/BILLING_OVERVIEW.vue'),
  BILL_DETAIL: () => import('../pages/BILL_DETAIL.vue'),
  BILL_PAYMENT: () => import('../pages/BILL_PAYMENT.vue'),
  BILL_PAYMENT_SUCCESS: () => import('../pages/BILL_PAYMENT_SUCCESS.vue'),
  BENEFITS_OVERVIEW: () => import('../pages/BENEFITS_OVERVIEW.vue'),
  SETTINGS_ACCOUNT: () => import('../pages/SETTINGS_ACCOUNT.vue'),
  SETTINGS_INSURANCE: () => import('../pages/SETTINGS_INSURANCE.vue')
}

const routes = Object.keys(pages).map(pageId => ({
  path: pageId === 'HOME' ? '/' : `/${pageId.toLowerCase().replace(/_/g, '-')}`,
  name: pageId,
  component: pages[pageId]
}))

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach((to, from, next) => {
  const store = useSignatureStore()
  if (to.name) {
    store.setCurrentPageId(to.name)
  }
  next()
})

export default router