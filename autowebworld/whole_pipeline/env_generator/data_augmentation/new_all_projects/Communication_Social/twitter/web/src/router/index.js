import { createRouter, createWebHistory } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'HOME',
      component: () => import('../pages/HOME.vue')
    },
    {
      path: '/home',
      name: 'HOME_TIMELINE',
      component: () => import('../pages/HOME_TIMELINE.vue')
    },
    {
      path: '/tweet/:tweet_id?',
      name: 'TWEET_DETAIL',
      component: () => import('../pages/TWEET_DETAIL.vue')
    },
    {
      path: '/compose/tweet',
      name: 'COMPOSE_TWEET',
      component: () => import('../pages/COMPOSE_TWEET.vue')
    },
    {
      path: '/tweet/success',
      name: 'TWEET_POST_SUCCESS',
      component: () => import('../pages/TWEET_POST_SUCCESS.vue')
    },
    {
      path: '/tweet/schedule-success',
      name: 'TWEET_SCHEDULE_SUCCESS',
      component: () => import('../pages/TWEET_SCHEDULE_SUCCESS.vue')
    },
    {
      path: '/profile',
      name: 'PROFILE_OVERVIEW',
      component: () => import('../pages/PROFILE_OVERVIEW.vue')
    },
    {
      path: '/profile/tweets',
      name: 'PROFILE_TWEETS',
      component: () => import('../pages/PROFILE_TWEETS.vue')
    },
    {
      path: '/profile/following',
      name: 'PROFILE_FOLLOWING_LIST',
      component: () => import('../pages/PROFILE_FOLLOWING_LIST.vue')
    },
    {
      path: '/user/:user_id?',
      name: 'USER_PROFILE_OVERVIEW',
      component: () => import('../pages/USER_PROFILE_OVERVIEW.vue')
    },
    {
      path: '/user/:user_id?/follow',
      name: 'FOLLOW_USER_CONFIRM',
      component: () => import('../pages/FOLLOW_USER_CONFIRM.vue')
    },
    {
      path: '/user/follow-success',
      name: 'FOLLOW_USER_SUCCESS',
      component: () => import('../pages/FOLLOW_USER_SUCCESS.vue')
    },
    {
      path: '/messages',
      name: 'MESSAGES_INBOX',
      component: () => import('../pages/MESSAGES_INBOX.vue')
    },
    {
      path: '/messages/thread/:thread_id?',
      name: 'MESSAGES_THREAD',
      component: () => import('../pages/MESSAGES_THREAD.vue')
    },
    {
      path: '/messages/compose',
      name: 'MESSAGES_COMPOSE',
      component: () => import('../pages/MESSAGES_COMPOSE.vue')
    },
    {
      path: '/messages/success',
      name: 'MESSAGE_SEND_SUCCESS',
      component: () => import('../pages/MESSAGE_SEND_SUCCESS.vue')
    },
    {
      path: '/notifications',
      name: 'NOTIFICATIONS',
      component: () => import('../pages/NOTIFICATIONS.vue')
    },
    {
      path: '/settings/profile',
      name: 'SETTINGS_PROFILE_EDIT',
      component: () => import('../pages/SETTINGS_PROFILE_EDIT.vue')
    },
    {
      path: '/settings/profile/success',
      name: 'PROFILE_UPDATE_SUCCESS',
      component: () => import('../pages/PROFILE_UPDATE_SUCCESS.vue')
    },
    {
      path: '/explore',
      name: 'TRENDS_EXPLORE',
      component: () => import('../pages/TRENDS_EXPLORE.vue')
    },
    {
      path: '/topic/:topic_id?',
      name: 'TOPIC_TWEET_LIST',
      component: () => import('../pages/TOPIC_TWEET_LIST.vue')
    },
    {
      path: '/bookmarks',
      name: 'BOOKMARKS',
      component: () => import('../pages/BOOKMARKS.vue')
    }
  ]
});

router.beforeEach((to, from, next) => {
  const signatureStore = useSignatureStore();
  if (to.name) {
    signatureStore.setCurrentPageId(to.name);
    
    // Handle URL parameters to update store
    if (to.params.tweet_id) signatureStore.selected_tweet_id = to.params.tweet_id;
    if (to.params.user_id) signatureStore.user_id = to.params.user_id; // For USER_PROFILE
    if (to.params.thread_id) signatureStore.thread_id = to.params.thread_id;
    if (to.params.topic_id) signatureStore.topic_id = to.params.topic_id; // If topic_id existed in signature
  }
  next();
});

export default router;