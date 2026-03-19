import { createRouter, createWebHistory } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

// Import all pages
import HOME from '../pages/HOME.vue';
import NEWS_FEED from '../pages/NEWS_FEED.vue';
import CREATE_POST from '../pages/CREATE_POST.vue';
import CREATE_POST_REVIEW from '../pages/CREATE_POST_REVIEW.vue';
import POST_DETAIL from '../pages/POST_DETAIL.vue';
import POST_PUBLISH_SUCCESS from '../pages/POST_PUBLISH_SUCCESS.vue';
import FRIENDS_LIST from '../pages/FRIENDS_LIST.vue';
import FRIEND_SUGGESTIONS from '../pages/FRIEND_SUGGESTIONS.vue';
import PROFILE_TIMELINE from '../pages/PROFILE_TIMELINE.vue';
import PROFILE_ABOUT from '../pages/PROFILE_ABOUT.vue';
import FRIEND_REQUEST_CONFIRM from '../pages/FRIEND_REQUEST_CONFIRM.vue';
import FRIEND_REQUEST_SENT_SUCCESS from '../pages/FRIEND_REQUEST_SENT_SUCCESS.vue';
import MESSENGER_INBOX from '../pages/MESSENGER_INBOX.vue';
import MESSAGE_THREAD from '../pages/MESSAGE_THREAD.vue';
import MESSAGE_COMPOSE from '../pages/MESSAGE_COMPOSE.vue';
import MESSAGE_REVIEW from '../pages/MESSAGE_REVIEW.vue';
import MESSAGE_SEND_SUCCESS from '../pages/MESSAGE_SEND_SUCCESS.vue';
import EVENTS_LIST from '../pages/EVENTS_LIST.vue';
import EVENT_DETAIL from '../pages/EVENT_DETAIL.vue';
import CREATE_EVENT_DETAILS from '../pages/CREATE_EVENT_DETAILS.vue';
import CREATE_EVENT_DATE from '../pages/CREATE_EVENT_DATE.vue';
import CREATE_EVENT_REVIEW from '../pages/CREATE_EVENT_REVIEW.vue';
import EVENT_CREATE_SUCCESS from '../pages/EVENT_CREATE_SUCCESS.vue';
import SETTINGS_ACCOUNT from '../pages/SETTINGS_ACCOUNT.vue';
import SETTINGS_ACCOUNT_REVIEW from '../pages/SETTINGS_ACCOUNT_REVIEW.vue';
import ACCOUNT_SETTINGS_SAVED_SUCCESS from '../pages/ACCOUNT_SETTINGS_SAVED_SUCCESS.vue';

const routes = [
  { path: '/', name: 'HOME', component: HOME },
  { path: '/feed', name: 'NEWS_FEED', component: NEWS_FEED },
  { path: '/post/create', name: 'CREATE_POST', component: CREATE_POST },
  { path: '/post/review', name: 'CREATE_POST_REVIEW', component: CREATE_POST_REVIEW },
  { path: '/post/:id', name: 'POST_DETAIL', component: POST_DETAIL },
  { path: '/post/success', name: 'POST_PUBLISH_SUCCESS', component: POST_PUBLISH_SUCCESS },
  { path: '/friends', name: 'FRIENDS_LIST', component: FRIENDS_LIST },
  { path: '/friends/suggestions', name: 'FRIEND_SUGGESTIONS', component: FRIEND_SUGGESTIONS },
  { path: '/profile/:id', name: 'PROFILE_TIMELINE', component: PROFILE_TIMELINE },
  { path: '/profile/:id/about', name: 'PROFILE_ABOUT', component: PROFILE_ABOUT },
  { path: '/friend/request/:id', name: 'FRIEND_REQUEST_CONFIRM', component: FRIEND_REQUEST_CONFIRM },
  { path: '/friend/success', name: 'FRIEND_REQUEST_SENT_SUCCESS', component: FRIEND_REQUEST_SENT_SUCCESS },
  { path: '/messages', name: 'MESSENGER_INBOX', component: MESSENGER_INBOX },
  { path: '/messages/:id', name: 'MESSAGE_THREAD', component: MESSAGE_THREAD },
  { path: '/messages/compose', name: 'MESSAGE_COMPOSE', component: MESSAGE_COMPOSE },
  { path: '/messages/review', name: 'MESSAGE_REVIEW', component: MESSAGE_REVIEW },
  { path: '/messages/success', name: 'MESSAGE_SEND_SUCCESS', component: MESSAGE_SEND_SUCCESS },
  { path: '/events', name: 'EVENTS_LIST', component: EVENTS_LIST },
  { path: '/events/:id', name: 'EVENT_DETAIL', component: EVENT_DETAIL },
  { path: '/events/create', name: 'CREATE_EVENT_DETAILS', component: CREATE_EVENT_DETAILS },
  { path: '/events/create/date', name: 'CREATE_EVENT_DATE', component: CREATE_EVENT_DATE },
  { path: '/events/create/review', name: 'CREATE_EVENT_REVIEW', component: CREATE_EVENT_REVIEW },
  { path: '/events/success', name: 'EVENT_CREATE_SUCCESS', component: EVENT_CREATE_SUCCESS },
  { path: '/settings', name: 'SETTINGS_ACCOUNT', component: SETTINGS_ACCOUNT },
  { path: '/settings/review', name: 'SETTINGS_ACCOUNT_REVIEW', component: SETTINGS_ACCOUNT_REVIEW },
  { path: '/settings/success', name: 'ACCOUNT_SETTINGS_SAVED_SUCCESS', component: ACCOUNT_SETTINGS_SAVED_SUCCESS },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

router.beforeEach((to, from, next) => {
  const signatureStore = useSignatureStore();
  signatureStore.setCurrentPageId(to.name);
  next();
});

export default router;