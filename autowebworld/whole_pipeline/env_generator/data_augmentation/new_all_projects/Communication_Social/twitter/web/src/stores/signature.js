import { defineStore } from 'pinia';
import fsmData from '../../fsm.json';

export const useSignatureStore = defineStore('signature', {
  state: () => ({
    // Global State defined in FSM
    currentPageId: fsmData.meta.initial_page_id,
    
    // Schema fields
    current_user_id: null,
    cookie_consent_given: null,
    
    // HOME_TIMELINE
    tweets: null,
    home_timeline_filters_applied: null,
    home_timeline_viewport_anchor_tweet_id: null,
    matched_tweet_id: null,
    home_timeline_has_searched: null,
    location_permission_granted: null,

    // TWEET_DETAIL
    selected_tweet_id: null,

    // COMPOSE_TWEET
    draft_tweet_text: null,
    draft_tweet_visibility: null,
    draft_tweet_allow_replies: null,

    // TWEET_POST_SUCCESS
    posted_tweet_id: null,

    // TWEET_SCHEDULE_SUCCESS
    scheduled_tweet_id: null,

    // PROFILE_OVERVIEW
    profile_user_id: null,

    // PROFILE_TWEETS
    profile_tweets_filters_applied: null,
    profile_tweets_viewport_anchor_tweet_id: null,
    profile_tweets_has_searched: null,

    // PROFILE_FOLLOWING_LIST
    following_users: null,
    profile_following_filters_applied: null,
    profile_following_viewport_anchor_user_id: null,

    // USER_PROFILE_OVERVIEW
    user_id: null,

    // FOLLOW_USER_CONFIRM
    target_user_id: null,
    confirm_checked: null,

    // FOLLOW_USER_SUCCESS
    followed_user_id: null,

    // MESSAGES_INBOX
    threads: null,
    messages_inbox_filters_applied: null,
    messages_inbox_viewport_anchor_thread_id: null,
    matched_thread_id: null,
    messages_inbox_has_searched: null,

    // MESSAGES_THREAD
    thread_id: null,

    // MESSAGES_COMPOSE
    recipient_user_id: null,
    message_text: null,

    // MESSAGE_SEND_SUCCESS
    sent_thread_id: null,

    // NOTIFICATIONS
    notifications: null,
    notifications_filters_applied: null,

    // SETTINGS_PROFILE_EDIT
    display_name: null,
    bio: null,
    location: null,

    // PROFILE_UPDATE_SUCCESS
    updated_profile_id: null,

    // TRENDS_EXPLORE
    trends: null,
    trends_filters_applied: null,
    trends_viewport_anchor_tweet_id: null,

    // TOPIC_TWEET_LIST
    topic_tweets_filters_applied: null,
    topic_tweets_viewport_anchor_tweet_id: null,

    // BOOKMARKS
    bookmarked_tweets: null,
    bookmarks_filters_applied: null,
    bookmarks_viewport_anchor_tweet_id: null,
  }),
  actions: {
    setCurrentPageId(pageId) {
      this.currentPageId = pageId;
    },
    // Generic setter for any field
    setField(field, value) {
      if (Object.prototype.hasOwnProperty.call(this.$state, field)) {
        this.$state[field] = value;
      } else {
        console.warn(`Field ${field} does not exist in signature store.`);
      }
    }
  },
  persist: {
    storage: sessionStorage,
  },
});