import { createRouter, createWebHistory } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

// Lazy load all pages
const HOME = () => import('../pages/HOME.vue')
const CHATS_LIST = () => import('../pages/CHATS_LIST.vue')
const CHAT_THREAD = () => import('../pages/CHAT_THREAD.vue')
const SEND_MESSAGE_CONFIRM = () => import('../pages/SEND_MESSAGE_CONFIRM.vue')
const SEND_MESSAGE_SUCCESS = () => import('../pages/SEND_MESSAGE_SUCCESS.vue')
const NEW_CHAT_CHOOSE_CONTACT = () => import('../pages/NEW_CHAT_CHOOSE_CONTACT.vue')
const NEW_CHAT_COMPOSE = () => import('../pages/NEW_CHAT_COMPOSE.vue')
const CONTACTS_LIST = () => import('../pages/CONTACTS_LIST.vue')
const CONTACT_DETAIL = () => import('../pages/CONTACT_DETAIL.vue')
const BLOCK_USER_CONFIRM = () => import('../pages/BLOCK_USER_CONFIRM.vue')
const BLOCK_USER_SUCCESS = () => import('../pages/BLOCK_USER_SUCCESS.vue')
const GROUPS_LIST = () => import('../pages/GROUPS_LIST.vue')
const GROUP_DETAIL = () => import('../pages/GROUP_DETAIL.vue')
const GROUP_CREATE_DETAILS = () => import('../pages/GROUP_CREATE_DETAILS.vue')
const GROUP_CREATE_ADD_MEMBERS = () => import('../pages/GROUP_CREATE_ADD_MEMBERS.vue')
const GROUP_CREATE_REVIEW = () => import('../pages/GROUP_CREATE_REVIEW.vue')
const CREATE_GROUP_SUCCESS = () => import('../pages/CREATE_GROUP_SUCCESS.vue')
const CALL_HISTORY = () => import('../pages/CALL_HISTORY.vue')
const START_CALL_SETUP = () => import('../pages/START_CALL_SETUP.vue')
const START_CALL_SUCCESS = () => import('../pages/START_CALL_SUCCESS.vue')
const CHAT_INFO = () => import('../pages/CHAT_INFO.vue')
const DISAPPEARING_MESSAGES_SETTINGS = () => import('../pages/DISAPPEARING_MESSAGES_SETTINGS.vue')
const SETTINGS_PRIVACY = () => import('../pages/SETTINGS_PRIVACY.vue')
const SETTINGS_NOTIFICATIONS = () => import('../pages/SETTINGS_NOTIFICATIONS.vue')
const UPDATE_SETTINGS_SUCCESS = () => import('../pages/UPDATE_SETTINGS_SUCCESS.vue')

const routes = [
  { path: '/', name: 'HOME', component: HOME },
  { path: '/chats', name: 'CHATS_LIST', component: CHATS_LIST },
  { path: '/chats/thread', name: 'CHAT_THREAD', component: CHAT_THREAD },
  { path: '/chats/send/confirm', name: 'SEND_MESSAGE_CONFIRM', component: SEND_MESSAGE_CONFIRM },
  { path: '/chats/send/success', name: 'SEND_MESSAGE_SUCCESS', component: SEND_MESSAGE_SUCCESS },
  { path: '/new-chat/contacts', name: 'NEW_CHAT_CHOOSE_CONTACT', component: NEW_CHAT_CHOOSE_CONTACT },
  { path: '/new-chat/compose', name: 'NEW_CHAT_COMPOSE', component: NEW_CHAT_COMPOSE },
  { path: '/contacts', name: 'CONTACTS_LIST', component: CONTACTS_LIST },
  { path: '/contacts/detail', name: 'CONTACT_DETAIL', component: CONTACT_DETAIL },
  { path: '/contacts/block/confirm', name: 'BLOCK_USER_CONFIRM', component: BLOCK_USER_CONFIRM },
  { path: '/contacts/block/success', name: 'BLOCK_USER_SUCCESS', component: BLOCK_USER_SUCCESS },
  { path: '/groups', name: 'GROUPS_LIST', component: GROUPS_LIST },
  { path: '/groups/detail', name: 'GROUP_DETAIL', component: GROUP_DETAIL },
  { path: '/groups/create/details', name: 'GROUP_CREATE_DETAILS', component: GROUP_CREATE_DETAILS },
  { path: '/groups/create/members', name: 'GROUP_CREATE_ADD_MEMBERS', component: GROUP_CREATE_ADD_MEMBERS },
  { path: '/groups/create/review', name: 'GROUP_CREATE_REVIEW', component: GROUP_CREATE_REVIEW },
  { path: '/groups/create/success', name: 'CREATE_GROUP_SUCCESS', component: CREATE_GROUP_SUCCESS },
  { path: '/calls', name: 'CALL_HISTORY', component: CALL_HISTORY },
  { path: '/calls/start', name: 'START_CALL_SETUP', component: START_CALL_SETUP },
  { path: '/calls/success', name: 'START_CALL_SUCCESS', component: START_CALL_SUCCESS },
  { path: '/chats/info', name: 'CHAT_INFO', component: CHAT_INFO },
  { path: '/chats/settings/disappearing', name: 'DISAPPEARING_MESSAGES_SETTINGS', component: DISAPPEARING_MESSAGES_SETTINGS },
  { path: '/settings/privacy', name: 'SETTINGS_PRIVACY', component: SETTINGS_PRIVACY },
  { path: '/settings/notifications', name: 'SETTINGS_NOTIFICATIONS', component: SETTINGS_NOTIFICATIONS },
  { path: '/settings/success', name: 'UPDATE_SETTINGS_SUCCESS', component: UPDATE_SETTINGS_SUCCESS },
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach((to, from, next) => {
  const signatureStore = useSignatureStore()
  if (to.name) {
    signatureStore.setCurrentPageId(to.name)
  }
  next()
})

export default router