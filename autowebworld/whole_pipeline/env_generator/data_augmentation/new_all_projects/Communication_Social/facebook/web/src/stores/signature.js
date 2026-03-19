import { defineStore } from 'pinia';
import { JSONPath } from 'jsonpath-plus';
import { get } from 'lodash-es';

export const useSignatureStore = defineStore('signature', {
  state: () => ({
    // Global State
    currentPageId: 'HOME',
    
    // Schema Fields from FSM
    current_user_id: 'user_1', // Default logged in user
    cookie_consent_given: null,
    
    // News Feed
    posts: [],
    location_permission_granted: null,
    news_feed_viewport_anchor_id: null,
    news_feed_filters_applied: null,
    news_feed_has_searched: null,
    matched_post_id: null,
    selected_post_id: null,
    
    // Create Post
    post_text: null,
    post_audience: null,
    
    // Post Publish Success
    success_message: null,
    
    // Friends List
    friends: [],
    friend_requests_list_filters_applied: null,
    friend_requests_list_viewport_anchor_id: null,
    friend_requests_list_has_searched: null,
    matched_user_id: null,
    selected_user_id: null,
    
    // Friend Suggestions
    suggested_friends: [],
    friend_suggestions_viewport_anchor_id: null,
    friend_suggestions_has_searched: null,
    
    // Messenger Inbox
    threads: [],
    messenger_inbox_viewport_anchor_id: null,
    messenger_inbox_has_searched: null,
    matched_thread_id: null,
    selected_thread_id: null,
    messenger_inbox_filters_applied: null,
    
    // Message Compose/Review
    message_text: null,
    recipient_selected: null,
    
    // Events List
    events: [],
    events_list_viewport_anchor_id: null,
    events_list_filters_applied: null,
    selected_event_id: null,
    
    // Create Event
    event_name: null,
    event_location: null,
    event_date: null,
    
    // Settings
    name_input_filled: null,
    privacy_option: null
  }),
  
  actions: {
    setCurrentPageId(pageId) {
      this.currentPageId = pageId;
    },
    
    // Helper to safely set value by path
    setByPath(path, value) {
      // Simple implementation for flat state or known structures
      // For complex JSONPath updates, we might need a more robust utility
      // Here we map JSONPath like $.field to state.field
      if (path.startsWith('$.')) {
        const field = path.substring(2);
        if (field in this) {
          this[field] = value;
        }
      }
    },
    
    getFromPath(path) {
      if (path.startsWith('$.')) {
        const field = path.substring(2);
        return this[field];
      }
      return undefined;
    }
  },
  
  persist: {
    storage: sessionStorage
  }
});