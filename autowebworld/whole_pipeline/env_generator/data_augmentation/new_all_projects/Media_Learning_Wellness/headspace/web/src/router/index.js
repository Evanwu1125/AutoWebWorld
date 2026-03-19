import { createRouter, createWebHistory } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

// Import Pages
import HOME from '../pages/HOME.vue';
import BROWSE from '../pages/BROWSE.vue';
import SESSION_DETAIL from '../pages/SESSION_DETAIL.vue';
import SESSION_START_FORM from '../pages/SESSION_START_FORM.vue';
import SESSION_REVIEW from '../pages/SESSION_REVIEW.vue';
import SESSION_COMPLETED_SUCCESS from '../pages/SESSION_COMPLETED_SUCCESS.vue';
import COURSES_LIST from '../pages/COURSES_LIST.vue';
import COURSE_DETAIL from '../pages/COURSE_DETAIL.vue';
import COURSE_ENROLL_FORM from '../pages/COURSE_ENROLL_FORM.vue';
import COURSE_ENROLL_REVIEW from '../pages/COURSE_ENROLL_REVIEW.vue';
import COURSE_ENROLLED_SUCCESS from '../pages/COURSE_ENROLLED_SUCCESS.vue';
import SLEEP_LIST from '../pages/SLEEP_LIST.vue';
import SLEEP_DETAIL from '../pages/SLEEP_DETAIL.vue';
import SLEEP_START_FORM from '../pages/SLEEP_START_FORM.vue';
import SLEEP_REVIEW from '../pages/SLEEP_REVIEW.vue';
import SLEEP_SESSION_COMPLETED_SUCCESS from '../pages/SLEEP_SESSION_COMPLETED_SUCCESS.vue';
import FOCUS_LIST from '../pages/FOCUS_LIST.vue';
import FOCUS_DETAIL from '../pages/FOCUS_DETAIL.vue';
import FOCUS_START_FORM from '../pages/FOCUS_START_FORM.vue';
import FOCUS_REVIEW from '../pages/FOCUS_REVIEW.vue';
import FOCUS_SESSION_COMPLETED_SUCCESS from '../pages/FOCUS_SESSION_COMPLETED_SUCCESS.vue';
import REMINDER_FORM from '../pages/REMINDER_FORM.vue';
import REMINDER_REVIEW from '../pages/REMINDER_REVIEW.vue';
import REMINDER_SET_SUCCESS from '../pages/REMINDER_SET_SUCCESS.vue';

const routes = [
  { path: '/', name: 'HOME', component: HOME },
  { path: '/browse', name: 'BROWSE', component: BROWSE },
  { path: '/session/:id?', name: 'SESSION_DETAIL', component: SESSION_DETAIL },
  { path: '/session-start', name: 'SESSION_START_FORM', component: SESSION_START_FORM },
  { path: '/session-review', name: 'SESSION_REVIEW', component: SESSION_REVIEW },
  { path: '/session-success', name: 'SESSION_COMPLETED_SUCCESS', component: SESSION_COMPLETED_SUCCESS },
  { path: '/courses', name: 'COURSES_LIST', component: COURSES_LIST },
  { path: '/course/:id?', name: 'COURSE_DETAIL', component: COURSE_DETAIL },
  { path: '/course-enroll', name: 'COURSE_ENROLL_FORM', component: COURSE_ENROLL_FORM },
  { path: '/course-enroll-review', name: 'COURSE_ENROLL_REVIEW', component: COURSE_ENROLL_REVIEW },
  { path: '/course-success', name: 'COURSE_ENROLLED_SUCCESS', component: COURSE_ENROLLED_SUCCESS },
  { path: '/sleep', name: 'SLEEP_LIST', component: SLEEP_LIST },
  { path: '/sleep/:id?', name: 'SLEEP_DETAIL', component: SLEEP_DETAIL },
  { path: '/sleep-start', name: 'SLEEP_START_FORM', component: SLEEP_START_FORM },
  { path: '/sleep-review', name: 'SLEEP_REVIEW', component: SLEEP_REVIEW },
  { path: '/sleep-success', name: 'SLEEP_SESSION_COMPLETED_SUCCESS', component: SLEEP_SESSION_COMPLETED_SUCCESS },
  { path: '/focus', name: 'FOCUS_LIST', component: FOCUS_LIST },
  { path: '/focus/:id?', name: 'FOCUS_DETAIL', component: FOCUS_DETAIL },
  { path: '/focus-start', name: 'FOCUS_START_FORM', component: FOCUS_START_FORM },
  { path: '/focus-review', name: 'FOCUS_REVIEW', component: FOCUS_REVIEW },
  { path: '/focus-success', name: 'FOCUS_SESSION_COMPLETED_SUCCESS', component: FOCUS_SESSION_COMPLETED_SUCCESS },
  { path: '/reminder', name: 'REMINDER_FORM', component: REMINDER_FORM },
  { path: '/reminder-review', name: 'REMINDER_REVIEW', component: REMINDER_REVIEW },
  { path: '/reminder-success', name: 'REMINDER_SET_SUCCESS', component: REMINDER_SET_SUCCESS },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

router.beforeEach((to, from, next) => {
  const store = useSignatureStore();
  store.setCurrentPageId(to.name);
  next();
});

export default router;