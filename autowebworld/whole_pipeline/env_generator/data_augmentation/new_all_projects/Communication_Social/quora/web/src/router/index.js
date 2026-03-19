import { createRouter, createWebHistory } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'HOME',
      component: () => import('../pages/HOME.vue')
    },
    {
      path: '/feed',
      name: 'FEED',
      component: () => import('../pages/FEED.vue')
    },
    {
      path: '/topics',
      name: 'TOPIC_LIST',
      component: () => import('../pages/TOPIC_LIST.vue')
    },
    {
      path: '/topic/:id',
      name: 'TOPIC_DETAIL',
      component: () => import('../pages/TOPIC_DETAIL.vue')
    },
    {
      path: '/ask',
      name: 'ASK_QUESTION_FORM',
      component: () => import('../pages/ASK_QUESTION_FORM.vue')
    },
    {
      path: '/ask/review',
      name: 'ASK_QUESTION_REVIEW',
      component: () => import('../pages/ASK_QUESTION_REVIEW.vue')
    },
    {
      path: '/ask/success',
      name: 'ASK_QUESTION_SUCCESS',
      component: () => import('../pages/ASK_QUESTION_SUCCESS.vue')
    },
    {
      path: '/question/:id',
      name: 'QUESTION_DETAIL',
      component: () => import('../pages/QUESTION_DETAIL.vue')
    },
    {
      path: '/answer',
      name: 'ANSWER_FORM',
      component: () => import('../pages/ANSWER_FORM.vue')
    },
    {
      path: '/answer/success',
      name: 'ANSWER_QUESTION_SUCCESS',
      component: () => import('../pages/ANSWER_QUESTION_SUCCESS.vue')
    },
    {
      path: '/topic/follow-success',
      name: 'FOLLOW_TOPIC_SUCCESS',
      component: () => import('../pages/FOLLOW_TOPIC_SUCCESS.vue')
    },
    {
      path: '/upvote-success',
      name: 'UPVOTE_SUCCESS',
      component: () => import('../pages/UPVOTE_SUCCESS.vue')
    },
    {
      path: '/profile',
      name: 'PROFILE',
      component: () => import('../pages/PROFILE.vue')
    },
    {
      path: '/profile/edit',
      name: 'PROFILE_EDIT',
      component: () => import('../pages/PROFILE_EDIT.vue')
    },
    {
      path: '/profile/edit-success',
      name: 'EDIT_PROFILE_SUCCESS',
      component: () => import('../pages/EDIT_PROFILE_SUCCESS.vue')
    },
    {
      path: '/notifications',
      name: 'NOTIFICATIONS',
      component: () => import('../pages/NOTIFICATIONS.vue')
    },
    {
      path: '/bookmarks',
      name: 'BOOKMARKS',
      component: () => import('../pages/BOOKMARKS.vue')
    }
  ]
})

router.beforeEach((to, from, next) => {
  const signatureStore = useSignatureStore()
  signatureStore.setCurrentPageId(to.name)
  next()
})

export default router