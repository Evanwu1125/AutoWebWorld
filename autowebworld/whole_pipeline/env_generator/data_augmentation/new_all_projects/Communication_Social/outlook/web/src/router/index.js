import { createRouter, createWebHistory } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

// Import all pages (Note: These files will be created in the next step)
import HOME from '../pages/HOME.vue'
import MAIL_INBOX from '../pages/MAIL_INBOX.vue'
import MAIL_MESSAGE_READ from '../pages/MAIL_MESSAGE_READ.vue'
import MAIL_COMPOSE from '../pages/MAIL_COMPOSE.vue'
import MAIL_REPLY from '../pages/MAIL_REPLY.vue'
import MAIL_FORWARD from '../pages/MAIL_FORWARD.vue'
import MAIL_MOVE from '../pages/MAIL_MOVE.vue'
import MAIL_SENT from '../pages/MAIL_SENT.vue'
import MAIL_DRAFTS from '../pages/MAIL_DRAFTS.vue'
import MAIL_TRASH from '../pages/MAIL_TRASH.vue'
import CALENDAR_MONTH from '../pages/CALENDAR_MONTH.vue'
import CALENDAR_DAY from '../pages/CALENDAR_DAY.vue'
import CALENDAR_NEW_EVENT from '../pages/CALENDAR_NEW_EVENT.vue'
import CALENDAR_EVENT_DETAIL from '../pages/CALENDAR_EVENT_DETAIL.vue'
import MAIL_SETTINGS_GENERAL from '../pages/MAIL_SETTINGS_GENERAL.vue'
import PEOPLE_LIST from '../pages/PEOPLE_LIST.vue'
import PEOPLE_DETAIL from '../pages/PEOPLE_DETAIL.vue'
import SEND_EMAIL_SUCCESS from '../pages/SEND_EMAIL_SUCCESS.vue'
import REPLY_EMAIL_SUCCESS from '../pages/REPLY_EMAIL_SUCCESS.vue'
import FORWARD_EMAIL_SUCCESS from '../pages/FORWARD_EMAIL_SUCCESS.vue'
import SCHEDULE_MEETING_SUCCESS from '../pages/SCHEDULE_MEETING_SUCCESS.vue'
import MOVE_EMAIL_SUCCESS from '../pages/MOVE_EMAIL_SUCCESS.vue'

const routes = [
  { path: '/', redirect: '/home' },
  { path: '/home', name: 'HOME', component: HOME },
  { path: '/inbox', name: 'MAIL_INBOX', component: MAIL_INBOX },
  { path: '/read/:id?', name: 'MAIL_MESSAGE_READ', component: MAIL_MESSAGE_READ },
  { path: '/compose', name: 'MAIL_COMPOSE', component: MAIL_COMPOSE },
  { path: '/reply', name: 'MAIL_REPLY', component: MAIL_REPLY },
  { path: '/forward', name: 'MAIL_FORWARD', component: MAIL_FORWARD },
  { path: '/move', name: 'MAIL_MOVE', component: MAIL_MOVE },
  { path: '/sent', name: 'MAIL_SENT', component: MAIL_SENT },
  { path: '/drafts', name: 'MAIL_DRAFTS', component: MAIL_DRAFTS },
  { path: '/trash', name: 'MAIL_TRASH', component: MAIL_TRASH },
  { path: '/calendar', name: 'CALENDAR_MONTH', component: CALENDAR_MONTH },
  { path: '/calendar/day', name: 'CALENDAR_DAY', component: CALENDAR_DAY },
  { path: '/calendar/new', name: 'CALENDAR_NEW_EVENT', component: CALENDAR_NEW_EVENT },
  { path: '/calendar/event/:id?', name: 'CALENDAR_EVENT_DETAIL', component: CALENDAR_EVENT_DETAIL },
  { path: '/settings', name: 'MAIL_SETTINGS_GENERAL', component: MAIL_SETTINGS_GENERAL },
  { path: '/people', name: 'PEOPLE_LIST', component: PEOPLE_LIST },
  { path: '/people/detail/:id?', name: 'PEOPLE_DETAIL', component: PEOPLE_DETAIL },
  { path: '/success/send', name: 'SEND_EMAIL_SUCCESS', component: SEND_EMAIL_SUCCESS },
  { path: '/success/reply', name: 'REPLY_EMAIL_SUCCESS', component: REPLY_EMAIL_SUCCESS },
  { path: '/success/forward', name: 'FORWARD_EMAIL_SUCCESS', component: FORWARD_EMAIL_SUCCESS },
  { path: '/success/schedule', name: 'SCHEDULE_MEETING_SUCCESS', component: SCHEDULE_MEETING_SUCCESS },
  { path: '/success/move', name: 'MOVE_EMAIL_SUCCESS', component: MOVE_EMAIL_SUCCESS },
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