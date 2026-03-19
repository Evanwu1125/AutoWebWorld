import { createRouter, createWebHistory } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

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
    path: '/schedule',
    name: 'SCHEDULE_MEETING_FORM',
    component: () => import('../pages/SCHEDULE_MEETING_FORM.vue')
  },
  {
    path: '/schedule/review',
    name: 'SCHEDULE_MEETING_REVIEW',
    component: () => import('../pages/SCHEDULE_MEETING_REVIEW.vue')
  },
  {
    path: '/schedule/success',
    name: 'SCHEDULE_MEETING_SUCCESS',
    component: () => import('../pages/SCHEDULE_MEETING_SUCCESS.vue')
  },
  {
    path: '/join',
    name: 'JOIN_MEETING_FORM',
    component: () => import('../pages/JOIN_MEETING_FORM.vue')
  },
  {
    path: '/join/preview',
    name: 'JOIN_MEETING_PREVIEW',
    component: () => import('../pages/JOIN_MEETING_PREVIEW.vue')
  },
  {
    path: '/join/success',
    name: 'JOIN_MEETING_SUCCESS',
    component: () => import('../pages/JOIN_MEETING_SUCCESS.vue')
  },
  {
    path: '/instant',
    name: 'INSTANT_MEETING_LOBBY',
    component: () => import('../pages/INSTANT_MEETING_LOBBY.vue')
  },
  {
    path: '/instant/success',
    name: 'START_INSTANT_MEETING_SUCCESS',
    component: () => import('../pages/START_INSTANT_MEETING_SUCCESS.vue')
  },
  {
    path: '/profile',
    name: 'PROFILE_OVERVIEW',
    component: () => import('../pages/PROFILE_OVERVIEW.vue')
  },
  {
    path: '/profile/rename',
    name: 'PROFILE_RENAME_FORM',
    component: () => import('../pages/PROFILE_RENAME_FORM.vue')
  },
  {
    path: '/profile/rename/success',
    name: 'RENAME_PROFILE_SUCCESS',
    component: () => import('../pages/RENAME_PROFILE_SUCCESS.vue')
  },
  {
    path: '/profile/password',
    name: 'PROFILE_CHANGE_PASSWORD_FORM',
    component: () => import('../pages/PROFILE_CHANGE_PASSWORD_FORM.vue')
  },
  {
    path: '/profile/password/success',
    name: 'CHANGE_PASSWORD_SUCCESS',
    component: () => import('../pages/CHANGE_PASSWORD_SUCCESS.vue')
  },
  {
    path: '/settings',
    name: 'SETTINGS_GENERAL',
    component: () => import('../pages/SETTINGS_GENERAL.vue')
  },
  {
    path: '/settings/video',
    name: 'SETTINGS_VIDEO',
    component: () => import('../pages/SETTINGS_VIDEO.vue')
  },
  {
    path: '/meetings',
    name: 'MEETINGS_LIST',
    component: () => import('../pages/MEETINGS_LIST.vue')
  },
  {
    path: '/meetings/detail', // Could use /meetings/:id but FSM implies single selected ID in store
    name: 'MEETING_DETAIL',
    component: () => import('../pages/MEETING_DETAIL.vue')
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

router.beforeEach((to, from, next) => {
  const store = useSignatureStore();
  store.setCurrentPageId(to.name);
  next();
});

export default router;