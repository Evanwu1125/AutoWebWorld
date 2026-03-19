import { createRouter, createWebHistory } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

// Lazy load all pages
const HOME = () => import('../pages/HOME.vue')
const BROWSE = () => import('../pages/BROWSE.vue')
const GENRE_CATEGORY = () => import('../pages/GENRE_CATEGORY.vue')
const YOUR_LIBRARY = () => import('../pages/YOUR_LIBRARY.vue')
const PLAYLIST_DETAIL = () => import('../pages/PLAYLIST_DETAIL.vue')
const TRACK_DETAIL = () => import('../pages/TRACK_DETAIL.vue')
const ALBUM_DETAIL = () => import('../pages/ALBUM_DETAIL.vue')
const ALBUM_DOWNLOAD_CONFIRM = () => import('../pages/ALBUM_DOWNLOAD_CONFIRM.vue')
const ALBUM_DOWNLOAD_SUCCESS = () => import('../pages/ALBUM_DOWNLOAD_SUCCESS.vue')
const ARTIST_DETAIL = () => import('../pages/ARTIST_DETAIL.vue')
const SEARCH_PAGE = () => import('../pages/SEARCH_PAGE.vue')
const SIGNUP = () => import('../pages/SIGNUP.vue')
const SIGNUP_SUCCESS = () => import('../pages/SIGNUP_SUCCESS.vue')
const ACCOUNT_OVERVIEW = () => import('../pages/ACCOUNT_OVERVIEW.vue')
const PREMIUM_UPSELL = () => import('../pages/PREMIUM_UPSELL.vue')
const PREMIUM_PAYMENT = () => import('../pages/PREMIUM_PAYMENT.vue')
const PREMIUM_UPGRADE_SUCCESS = () => import('../pages/PREMIUM_UPGRADE_SUCCESS.vue')
const PLAYLIST_CREATE = () => import('../pages/PLAYLIST_CREATE.vue')
const PLAYLIST_CREATED_SUCCESS = () => import('../pages/PLAYLIST_CREATED_SUCCESS.vue')
const PLAYLIST_SHARE = () => import('../pages/PLAYLIST_SHARE.vue')
const PLAYLIST_SHARED_SUCCESS = () => import('../pages/PLAYLIST_SHARED_SUCCESS.vue')
const PAYMENT_METHODS = () => import('../pages/PAYMENT_METHODS.vue')
const PAYMENT_METHOD_DETAIL = () => import('../pages/PAYMENT_METHOD_DETAIL.vue')
const SETTINGS = () => import('../pages/SETTINGS.vue')

const routes = [
  { path: '/', name: 'HOME', component: HOME },
  { path: '/browse', name: 'BROWSE', component: BROWSE },
  { path: '/genre', name: 'GENRE_CATEGORY', component: GENRE_CATEGORY },
  { path: '/library', name: 'YOUR_LIBRARY', component: YOUR_LIBRARY },
  { path: '/playlist/:id?', name: 'PLAYLIST_DETAIL', component: PLAYLIST_DETAIL },
  { path: '/track/:id?', name: 'TRACK_DETAIL', component: TRACK_DETAIL },
  { path: '/album/:id?', name: 'ALBUM_DETAIL', component: ALBUM_DETAIL },
  { path: '/album-download-confirm', name: 'ALBUM_DOWNLOAD_CONFIRM', component: ALBUM_DOWNLOAD_CONFIRM },
  { path: '/album-download-success', name: 'ALBUM_DOWNLOAD_SUCCESS', component: ALBUM_DOWNLOAD_SUCCESS },
  { path: '/artist/:id?', name: 'ARTIST_DETAIL', component: ARTIST_DETAIL },
  { path: '/search', name: 'SEARCH_PAGE', component: SEARCH_PAGE },
  { path: '/signup', name: 'SIGNUP', component: SIGNUP },
  { path: '/signup-success', name: 'SIGNUP_SUCCESS', component: SIGNUP_SUCCESS },
  { path: '/account', name: 'ACCOUNT_OVERVIEW', component: ACCOUNT_OVERVIEW },
  { path: '/premium', name: 'PREMIUM_UPSELL', component: PREMIUM_UPSELL },
  { path: '/premium-payment', name: 'PREMIUM_PAYMENT', component: PREMIUM_PAYMENT },
  { path: '/premium-success', name: 'PREMIUM_UPGRADE_SUCCESS', component: PREMIUM_UPGRADE_SUCCESS },
  { path: '/playlist/create', name: 'PLAYLIST_CREATE', component: PLAYLIST_CREATE },
  { path: '/playlist/create-success', name: 'PLAYLIST_CREATED_SUCCESS', component: PLAYLIST_CREATED_SUCCESS },
  { path: '/playlist/share', name: 'PLAYLIST_SHARE', component: PLAYLIST_SHARE },
  { path: '/playlist/share-success', name: 'PLAYLIST_SHARED_SUCCESS', component: PLAYLIST_SHARED_SUCCESS },
  { path: '/payment-methods', name: 'PAYMENT_METHODS', component: PAYMENT_METHODS },
  { path: '/payment-method/:id?', name: 'PAYMENT_METHOD_DETAIL', component: PAYMENT_METHOD_DETAIL },
  { path: '/settings', name: 'SETTINGS', component: SETTINGS },
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