import { createRouter, createWebHistory } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

// Import all pages
import HOME from '../pages/HOME.vue'
import LOGIN from '../pages/LOGIN.vue'
import COURSE_DISCOVERY from '../pages/COURSE_DISCOVERY.vue'
import COURSE_DETAIL from '../pages/COURSE_DETAIL.vue'
import COURSE_SYLLABUS from '../pages/COURSE_SYLLABUS.vue'
import AUDIT_CONFIRM from '../pages/AUDIT_CONFIRM.vue'
import ENROLLMENT_OPTIONS from '../pages/ENROLLMENT_OPTIONS.vue'
import PAYMENT_DETAILS from '../pages/PAYMENT_DETAILS.vue'
import ORDER_REVIEW from '../pages/ORDER_REVIEW.vue'
import ENROLL_COURSE_SUCCESS from '../pages/ENROLL_COURSE_SUCCESS.vue'
import AUDIT_COURSE_SUCCESS from '../pages/AUDIT_COURSE_SUCCESS.vue'
import COURSE_HOME from '../pages/COURSE_HOME.vue'
import COURSE_RATING_FORM from '../pages/COURSE_RATING_FORM.vue'
import COURSE_RATING_SUBMITTED_SUCCESS from '../pages/COURSE_RATING_SUBMITTED_SUCCESS.vue'
import SPECIALIZATION_LIST from '../pages/SPECIALIZATION_LIST.vue'
import SPECIALIZATION_DETAIL from '../pages/SPECIALIZATION_DETAIL.vue'
import SPECIALIZATION_SUBSCRIBE_PAYMENT from '../pages/SPECIALIZATION_SUBSCRIBE_PAYMENT.vue'
import SPECIALIZATION_SUBSCRIBE_SUCCESS from '../pages/SPECIALIZATION_SUBSCRIBE_SUCCESS.vue'
import PROFESSIONAL_CERT_LIST from '../pages/PROFESSIONAL_CERT_LIST.vue'
import PROFESSIONAL_CERT_DETAIL from '../pages/PROFESSIONAL_CERT_DETAIL.vue'
import PROFESSIONAL_CERT_ENROLL_PAYMENT from '../pages/PROFESSIONAL_CERT_ENROLL_PAYMENT.vue'
import ENROLL_PROFESSIONAL_CERT_SUCCESS from '../pages/ENROLL_PROFESSIONAL_CERT_SUCCESS.vue'
import LEARNER_DASHBOARD from '../pages/LEARNER_DASHBOARD.vue'

const routes = [
  { path: '/', name: 'HOME', component: HOME },
  { path: '/login', name: 'LOGIN', component: LOGIN },
  { path: '/courses', name: 'COURSE_DISCOVERY', component: COURSE_DISCOVERY },
  { path: '/course/:id', name: 'COURSE_DETAIL', component: COURSE_DETAIL },
  { path: '/course/:id/syllabus', name: 'COURSE_SYLLABUS', component: COURSE_SYLLABUS },
  { path: '/course/:id/audit', name: 'AUDIT_CONFIRM', component: AUDIT_CONFIRM },
  { path: '/course/:id/enroll', name: 'ENROLLMENT_OPTIONS', component: ENROLLMENT_OPTIONS },
  { path: '/payment', name: 'PAYMENT_DETAILS', component: PAYMENT_DETAILS },
  { path: '/order/review', name: 'ORDER_REVIEW', component: ORDER_REVIEW },
  { path: '/enroll/success', name: 'ENROLL_COURSE_SUCCESS', component: ENROLL_COURSE_SUCCESS },
  { path: '/audit/success', name: 'AUDIT_COURSE_SUCCESS', component: AUDIT_COURSE_SUCCESS },
  { path: '/course/:id/home', name: 'COURSE_HOME', component: COURSE_HOME },
  { path: '/course/:id/rate', name: 'COURSE_RATING_FORM', component: COURSE_RATING_FORM },
  { path: '/course/:id/rate/success', name: 'COURSE_RATING_SUBMITTED_SUCCESS', component: COURSE_RATING_SUBMITTED_SUCCESS },
  { path: '/specializations', name: 'SPECIALIZATION_LIST', component: SPECIALIZATION_LIST },
  { path: '/specialization/:id', name: 'SPECIALIZATION_DETAIL', component: SPECIALIZATION_DETAIL },
  { path: '/specialization/:id/subscribe', name: 'SPECIALIZATION_SUBSCRIBE_PAYMENT', component: SPECIALIZATION_SUBSCRIBE_PAYMENT },
  { path: '/specialization/success', name: 'SPECIALIZATION_SUBSCRIBE_SUCCESS', component: SPECIALIZATION_SUBSCRIBE_SUCCESS },
  { path: '/professional-certs', name: 'PROFESSIONAL_CERT_LIST', component: PROFESSIONAL_CERT_LIST },
  { path: '/professional-cert/:id', name: 'PROFESSIONAL_CERT_DETAIL', component: PROFESSIONAL_CERT_DETAIL },
  { path: '/professional-cert/:id/enroll', name: 'PROFESSIONAL_CERT_ENROLL_PAYMENT', component: PROFESSIONAL_CERT_ENROLL_PAYMENT },
  { path: '/professional-cert/success', name: 'ENROLL_PROFESSIONAL_CERT_SUCCESS', component: ENROLL_PROFESSIONAL_CERT_SUCCESS },
  { path: '/dashboard', name: 'LEARNER_DASHBOARD', component: LEARNER_DASHBOARD }
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