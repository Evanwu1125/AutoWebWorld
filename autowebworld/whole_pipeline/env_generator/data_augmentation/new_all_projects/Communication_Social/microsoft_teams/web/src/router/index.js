import { createRouter, createWebHistory } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

// Import all pages
import HOME from '../pages/HOME.vue'
import TEAMS_LIST from '../pages/TEAMS_LIST.vue'
import CHANNELS_LIST from '../pages/CHANNELS_LIST.vue'
import CHANNEL_POST_COMPOSE from '../pages/CHANNEL_POST_COMPOSE.vue'
import CHANNEL_POST_SENT_SUCCESS from '../pages/CHANNEL_POST_SENT_SUCCESS.vue'
import CREATE_TEAM from '../pages/CREATE_TEAM.vue'
import TEAM_CREATED_SUCCESS from '../pages/TEAM_CREATED_SUCCESS.vue'
import CALENDAR_VIEW from '../pages/CALENDAR_VIEW.vue'
import MEETING_DETAILS from '../pages/MEETING_DETAILS.vue'
import MEETING_REVIEW from '../pages/MEETING_REVIEW.vue'
import MEETING_SCHEDULED_SUCCESS from '../pages/MEETING_SCHEDULED_SUCCESS.vue'
import MEET_NOW_SETUP from '../pages/MEET_NOW_SETUP.vue'
import MEET_NOW_STARTED_SUCCESS from '../pages/MEET_NOW_STARTED_SUCCESS.vue'
import CHAT_LIST from '../pages/CHAT_LIST.vue'
import CHAT_THREAD from '../pages/CHAT_THREAD.vue'
import CHAT_MESSAGE_SENT_SUCCESS from '../pages/CHAT_MESSAGE_SENT_SUCCESS.vue'
import CALLS_HUB from '../pages/CALLS_HUB.vue'

const routes = [
  { path: '/', name: 'HOME', component: HOME },
  { path: '/teams', name: 'TEAMS_LIST', component: TEAMS_LIST },
  { path: '/teams/:teamId/channels', name: 'CHANNELS_LIST', component: CHANNELS_LIST },
  { path: '/teams/:teamId/channels/:channelId/compose', name: 'CHANNEL_POST_COMPOSE', component: CHANNEL_POST_COMPOSE },
  { path: '/teams/:teamId/channels/:channelId/sent', name: 'CHANNEL_POST_SENT_SUCCESS', component: CHANNEL_POST_SENT_SUCCESS },
  { path: '/teams/create', name: 'CREATE_TEAM', component: CREATE_TEAM },
  { path: '/teams/created', name: 'TEAM_CREATED_SUCCESS', component: TEAM_CREATED_SUCCESS },
  { path: '/calendar', name: 'CALENDAR_VIEW', component: CALENDAR_VIEW },
  { path: '/calendar/new', name: 'MEETING_DETAILS', component: MEETING_DETAILS },
  { path: '/calendar/review', name: 'MEETING_REVIEW', component: MEETING_REVIEW },
  { path: '/calendar/scheduled', name: 'MEETING_SCHEDULED_SUCCESS', component: MEETING_SCHEDULED_SUCCESS },
  { path: '/calendar/meet-now', name: 'MEET_NOW_SETUP', component: MEET_NOW_SETUP },
  { path: '/calendar/meet-now/started', name: 'MEET_NOW_STARTED_SUCCESS', component: MEET_NOW_STARTED_SUCCESS },
  { path: '/chat', name: 'CHAT_LIST', component: CHAT_LIST },
  { path: '/chat/:chatId', name: 'CHAT_THREAD', component: CHAT_THREAD },
  { path: '/chat/:chatId/sent', name: 'CHAT_MESSAGE_SENT_SUCCESS', component: CHAT_MESSAGE_SENT_SUCCESS },
  { path: '/calls', name: 'CALLS_HUB', component: CALLS_HUB }
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