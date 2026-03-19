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
      path: '/trending',
      name: 'TRENDING',
      component: () => import('../pages/TRENDING.vue')
    },
    {
      path: '/results',
      name: 'SEARCH_RESULTS',
      component: () => import('../pages/SEARCH_RESULTS.vue')
    },
    {
      path: '/watch/:id?',
      name: 'WATCH_VIDEO',
      component: () => import('../pages/WATCH_VIDEO.vue')
    },
    {
      path: '/watch/like-success',
      name: 'WATCH_LIKE_SUCCESS',
      component: () => import('../pages/WATCH_LIKE_SUCCESS.vue')
    },
    {
      path: '/watch/comment-success',
      name: 'WATCH_COMMENT_SUCCESS',
      component: () => import('../pages/WATCH_COMMENT_SUCCESS.vue')
    },
    {
      path: '/subscriptions',
      name: 'SUBSCRIPTIONS',
      component: () => import('../pages/SUBSCRIPTIONS.vue')
    },
    {
      path: '/channel/:id?',
      name: 'CHANNEL_PAGE',
      component: () => import('../pages/CHANNEL_PAGE.vue')
    },
    {
      path: '/channel/subscribe-confirm',
      name: 'CHANNEL_SUBSCRIBE_CONFIRM',
      component: () => import('../pages/CHANNEL_SUBSCRIBE_CONFIRM.vue')
    },
    {
      path: '/channel/subscribe-success',
      name: 'SUBSCRIBE_SUCCESS',
      component: () => import('../pages/SUBSCRIBE_SUCCESS.vue')
    },
    {
      path: '/library',
      name: 'LIBRARY',
      component: () => import('../pages/LIBRARY.vue')
    },
    {
      path: '/playlist/:id?',
      name: 'PLAYLIST_DETAIL',
      component: () => import('../pages/PLAYLIST_DETAIL.vue')
    },
    {
      path: '/playlist/create',
      name: 'PLAYLIST_CREATE_FORM',
      component: () => import('../pages/PLAYLIST_CREATE_FORM.vue')
    },
    {
      path: '/playlist/create-success',
      name: 'PLAYLIST_CREATE_SUCCESS',
      component: () => import('../pages/PLAYLIST_CREATE_SUCCESS.vue')
    },
    {
      path: '/upload',
      name: 'UPLOAD_VIDEO',
      component: () => import('../pages/UPLOAD_VIDEO.vue')
    },
    {
      path: '/upload/details',
      name: 'UPLOAD_DETAILS',
      component: () => import('../pages/UPLOAD_DETAILS.vue')
    },
    {
      path: '/upload/visibility',
      name: 'UPLOAD_VISIBILITY',
      component: () => import('../pages/UPLOAD_VISIBILITY.vue')
    },
    {
      path: '/upload/success',
      name: 'UPLOAD_PUBLISH_SUCCESS',
      component: () => import('../pages/UPLOAD_PUBLISH_SUCCESS.vue')
    }
  ]
})

router.beforeEach((to, from, next) => {
  const signatureStore = useSignatureStore()
  signatureStore.setCurrentPageId(to.name)
  next()
})

export default router