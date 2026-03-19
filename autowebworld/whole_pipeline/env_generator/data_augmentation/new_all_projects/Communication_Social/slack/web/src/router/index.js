import { createRouter, createWebHistory } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

// Import Pages
import HOME from '../pages/HOME.vue'
import WORKSPACE_OVERVIEW from '../pages/WORKSPACE_OVERVIEW.vue'
import CHANNEL_LIST from '../pages/CHANNEL_LIST.vue'
import DM_LIST from '../pages/DM_LIST.vue'
import CHANNEL_DETAIL from '../pages/CHANNEL_DETAIL.vue'
import DM_DETAIL from '../pages/DM_DETAIL.vue'
import MESSAGE_COMPOSE from '../pages/MESSAGE_COMPOSE.vue'
import DM_COMPOSE from '../pages/DM_COMPOSE.vue'
import CHANNEL_SETTINGS from '../pages/CHANNEL_SETTINGS.vue'
import MESSAGE_SCHEDULE from '../pages/MESSAGE_SCHEDULE.vue'
import PROFILE_VIEW from '../pages/PROFILE_VIEW.vue'
import PROFILE_EDIT from '../pages/PROFILE_EDIT.vue'
import SEND_MESSAGE_SUCCESS from '../pages/SEND_MESSAGE_SUCCESS.vue'
import CREATE_CHANNEL_SUCCESS from '../pages/CREATE_CHANNEL_SUCCESS.vue'
import START_DM_SUCCESS from '../pages/START_DM_SUCCESS.vue'
import SCHEDULE_MESSAGE_SUCCESS from '../pages/SCHEDULE_MESSAGE_SUCCESS.vue'
import UPDATE_PROFILE_SUCCESS from '../pages/UPDATE_PROFILE_SUCCESS.vue'

const routes = [
  { path: '/', name: 'HOME', component: HOME },
  { path: '/workspace', name: 'WORKSPACE_OVERVIEW', component: WORKSPACE_OVERVIEW },
  { path: '/channels', name: 'CHANNEL_LIST', component: CHANNEL_LIST },
  { path: '/dms', name: 'DM_LIST', component: DM_LIST },
  { path: '/channel/:id', name: 'CHANNEL_DETAIL', component: CHANNEL_DETAIL },
  { path: '/dm/:id', name: 'DM_DETAIL', component: DM_DETAIL },
  { path: '/compose', name: 'MESSAGE_COMPOSE', component: MESSAGE_COMPOSE },
  { path: '/dm-compose', name: 'DM_COMPOSE', component: DM_COMPOSE },
  { path: '/channel/:id/settings', name: 'CHANNEL_SETTINGS', component: CHANNEL_SETTINGS },
  { path: '/schedule', name: 'MESSAGE_SCHEDULE', component: MESSAGE_SCHEDULE },
  { path: '/profile', name: 'PROFILE_VIEW', component: PROFILE_VIEW },
  { path: '/profile/edit', name: 'PROFILE_EDIT', component: PROFILE_EDIT },
  { path: '/success/sent', name: 'SEND_MESSAGE_SUCCESS', component: SEND_MESSAGE_SUCCESS },
  { path: '/success/channel', name: 'CREATE_CHANNEL_SUCCESS', component: CREATE_CHANNEL_SUCCESS },
  { path: '/success/dm-sent', name: 'START_DM_SUCCESS', component: START_DM_SUCCESS },
  { path: '/success/scheduled', name: 'SCHEDULE_MESSAGE_SUCCESS', component: SCHEDULE_MESSAGE_SUCCESS },
  { path: '/success/profile', name: 'UPDATE_PROFILE_SUCCESS', component: UPDATE_PROFILE_SUCCESS },
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