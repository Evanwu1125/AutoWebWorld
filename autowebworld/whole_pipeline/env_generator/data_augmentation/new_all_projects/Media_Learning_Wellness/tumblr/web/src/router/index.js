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
      path: '/signup',
      name: 'SIGNUP',
      component: () => import('../pages/SIGNUP.vue')
    },
    {
      path: '/dashboard',
      name: 'DASHBOARD_FEED',
      component: () => import('../pages/DASHBOARD_FEED.vue')
    },
    {
      path: '/explore',
      name: 'EXPLORE',
      component: () => import('../pages/EXPLORE.vue')
    },
    {
      path: '/blog/:id', // Using param for specific blog overview
      name: 'BLOG_OVERVIEW',
      component: () => import('../pages/BLOG_OVERVIEW.vue')
    },
    {
      path: '/blog/:id/posts',
      name: 'BLOG_POSTS_LIST',
      component: () => import('../pages/BLOG_POSTS_LIST.vue')
    },
    {
      path: '/blog/:id/info',
      name: 'BLOG_INFO',
      component: () => import('../pages/BLOG_INFO.vue')
    },
    {
      path: '/blog/:id/follow',
      name: 'FOLLOW_BLOG_CONFIRM',
      component: () => import('../pages/FOLLOW_BLOG_CONFIRM.vue')
    },
    {
      path: '/blog/follow/success',
      name: 'FOLLOW_BLOG_SUCCESS',
      component: () => import('../pages/FOLLOW_BLOG_SUCCESS.vue')
    },
    {
      path: '/post/:id',
      name: 'POST_DETAIL',
      component: () => import('../pages/POST_DETAIL.vue')
    },
    {
      path: '/post/:id/reblog',
      name: 'REBLOG_FORM',
      component: () => import('../pages/REBLOG_FORM.vue')
    },
    {
      path: '/compose',
      name: 'COMPOSE_TEXT_POST',
      component: () => import('../pages/COMPOSE_TEXT_POST.vue')
    },
    {
      path: '/schedule',
      name: 'SCHEDULE_POST',
      component: () => import('../pages/SCHEDULE_POST.vue')
    },
    {
      path: '/published',
      name: 'POST_PUBLISH_SUCCESS',
      component: () => import('../pages/POST_PUBLISH_SUCCESS.vue')
    },
    {
      path: '/scheduled',
      name: 'POST_SCHEDULE_SUCCESS',
      component: () => import('../pages/POST_SCHEDULE_SUCCESS.vue')
    },
    {
      path: '/messages',
      name: 'MESSAGES_INBOX',
      component: () => import('../pages/MESSAGES_INBOX.vue')
    },
    {
      path: '/messages/:id',
      name: 'MESSAGE_THREAD',
      component: () => import('../pages/MESSAGE_THREAD.vue')
    },
    {
      path: '/messages/compose/new',
      name: 'MESSAGE_COMPOSE',
      component: () => import('../pages/MESSAGE_COMPOSE.vue')
    },
    {
      path: '/messages/sent/success',
      name: 'MESSAGE_SEND_SUCCESS',
      component: () => import('../pages/MESSAGE_SEND_SUCCESS.vue')
    },
    {
      path: '/settings',
      name: 'ACCOUNT_SETTINGS',
      component: () => import('../pages/ACCOUNT_SETTINGS.vue')
    },
    {
      path: '/settings/saved',
      name: 'ACCOUNT_SETTINGS_SAVE_SUCCESS',
      component: () => import('../pages/ACCOUNT_SETTINGS_SAVE_SUCCESS.vue')
    }
  ]
})

router.beforeEach((to, from, next) => {
  const signatureStore = useSignatureStore()
  // Update current page ID in store
  if (to.name) {
    signatureStore.setCurrentPageId(to.name)
  }
  next()
})

export default router