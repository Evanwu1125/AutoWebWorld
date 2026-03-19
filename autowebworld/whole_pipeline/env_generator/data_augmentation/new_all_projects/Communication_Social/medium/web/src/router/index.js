import { createRouter, createWebHistory } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

const routes = [
  {
    path: '/',
    name: 'HOME',
    component: () => import('../pages/HOME.vue')
  },
  {
    path: '/posts',
    name: 'POST_LIST',
    component: () => import('../pages/POST_LIST.vue')
  },
  {
    path: '/post/:id',
    name: 'POST_DETAIL',
    component: () => import('../pages/POST_DETAIL.vue')
  },
  {
    path: '/post/:id/comment',
    name: 'COMMENT_FORM',
    component: () => import('../pages/COMMENT_FORM.vue')
  },
  {
    path: '/comment/success',
    name: 'COMMENT_SUBMIT_SUCCESS',
    component: () => import('../pages/COMMENT_SUBMIT_SUCCESS.vue')
  },
  {
    path: '/new-story',
    name: 'NEW_STORY_EDITOR',
    component: () => import('../pages/NEW_STORY_EDITOR.vue')
  },
  {
    path: '/new-story/publish',
    name: 'PUBLISH_OPTIONS',
    component: () => import('../pages/PUBLISH_OPTIONS.vue')
  },
  {
    path: '/new-story/confirm',
    name: 'PUBLISH_CONFIRM',
    component: () => import('../pages/PUBLISH_CONFIRM.vue')
  },
  {
    path: '/new-story/schedule',
    name: 'SCHEDULE_PICKER',
    component: () => import('../pages/SCHEDULE_PICKER.vue')
  },
  {
    path: '/publish/success',
    name: 'PUBLISH_POST_SUCCESS',
    component: () => import('../pages/PUBLISH_POST_SUCCESS.vue')
  },
  {
    path: '/schedule/success',
    name: 'SCHEDULE_POST_SUCCESS',
    component: () => import('../pages/SCHEDULE_POST_SUCCESS.vue')
  },
  {
    path: '/profile',
    name: 'PROFILE_OVERVIEW',
    component: () => import('../pages/PROFILE_OVERVIEW.vue')
  },
  {
    path: '/profile/edit',
    name: 'PROFILE_EDIT',
    component: () => import('../pages/PROFILE_EDIT.vue')
  },
  {
    path: '/profile/updated',
    name: 'PROFILE_UPDATE_SUCCESS',
    component: () => import('../pages/PROFILE_UPDATE_SUCCESS.vue')
  },
  {
    path: '/stories',
    name: 'STORIES_DRAFTS',
    component: () => import('../pages/STORIES_DRAFTS.vue')
  },
  {
    path: '/publications',
    name: 'PUBLICATION_LIST',
    component: () => import('../pages/PUBLICATION_LIST.vue')
  },
  {
    path: '/publication/:id',
    name: 'PUBLICATION_DETAIL',
    component: () => import('../pages/PUBLICATION_DETAIL.vue')
  },
  {
    path: '/settings',
    name: 'SETTINGS_PREFERENCES',
    component: () => import('../pages/SETTINGS_PREFERENCES.vue')
  },
  {
    path: '/membership',
    name: 'MEMBERSHIP_LANDING',
    component: () => import('../pages/MEMBERSHIP_LANDING.vue')
  },
  {
    path: '/payment',
    name: 'PAYMENT_DETAILS',
    component: () => import('../pages/PAYMENT_DETAILS.vue')
  },
  {
    path: '/subscription/success',
    name: 'SUBSCRIPTION_SUCCESS',
    component: () => import('../pages/SUBSCRIPTION_SUCCESS.vue')
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