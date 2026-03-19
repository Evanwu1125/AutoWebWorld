import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useSignatureStore = defineStore('signature', () => {
  // Global Signature Schema
  const posts = ref([])
  const blogs = ref([])
  const messages = ref([])

  // Page Specific Signatures
  // HOME
  const current_user_id = ref(null)
  const cookie_consent_given = ref(null)

  // SIGNUP
  const signup_email = ref(null)
  const signup_password = ref(null)
  const signup_blogname = ref(null)

  // DASHBOARD_FEED
  const location_permission_granted = ref(null)
  const dashboard_feed_filters_applied = ref(null)
  const dashboard_feed_has_searched = ref(null)
  const dashboard_feed_viewport_anchor_id = ref(null)
  const matched_post_id = ref(null)
  const selected_post_id = ref(null)

  // EXPLORE
  const explore_filters_applied = ref(null)
  const explore_has_searched = ref(null)
  const explore_viewport_anchor_id = ref(null)
  const matched_blog_id = ref(null)
  const selected_blog_id = ref(null)

  // BLOG_OVERVIEW / INFO / FOLLOW
  const confirm_follow_notes = ref(null)
  
  // BLOG_POSTS_LIST
  const blog_posts_list_filters_applied = ref(null)
  const blog_posts_list_has_searched = ref(null)
  const blog_posts_list_viewport_anchor_id = ref(null)

  // FOLLOW_BLOG_SUCCESS / POST_PUBLISH_SUCCESS / ACCOUNT_SETTINGS_SAVE_SUCCESS
  const success_message = ref(null)

  // REBLOG_FORM
  const reblog_text = ref(null)
  const reblog_tags = ref(null)

  // COMPOSE_TEXT_POST
  const compose_title = ref(null)
  const compose_body = ref(null)
  const compose_tags = ref(null)
  const compose_visibility = ref(null)

  // SCHEDULE_POST
  const schedule_datetime = ref(null)

  // MESSAGES_INBOX
  const messages_inbox_filters_applied = ref(null)
  const messages_inbox_has_searched = ref(null)
  const messages_inbox_viewport_anchor_id = ref(null)
  const matched_message_id = ref(null)
  const selected_message_id = ref(null)

  // MESSAGE_COMPOSE
  const message_recipient = ref(null)
  const message_body = ref(null)

  // ACCOUNT_SETTINGS
  const display_name = ref(null)
  const bio = ref(null)
  const theme_color = ref(null)

  // Navigation State
  const currentPageId = ref('HOME')

  function setCurrentPageId(id) {
    currentPageId.value = id
  }

  return {
    // State
    posts,
    blogs,
    messages,
    current_user_id,
    cookie_consent_given,
    signup_email,
    signup_password,
    signup_blogname,
    location_permission_granted,
    dashboard_feed_filters_applied,
    dashboard_feed_has_searched,
    dashboard_feed_viewport_anchor_id,
    matched_post_id,
    selected_post_id,
    explore_filters_applied,
    explore_has_searched,
    explore_viewport_anchor_id,
    matched_blog_id,
    selected_blog_id,
    confirm_follow_notes,
    blog_posts_list_filters_applied,
    blog_posts_list_has_searched,
    blog_posts_list_viewport_anchor_id,
    success_message,
    reblog_text,
    reblog_tags,
    compose_title,
    compose_body,
    compose_tags,
    compose_visibility,
    schedule_datetime,
    messages_inbox_filters_applied,
    messages_inbox_has_searched,
    messages_inbox_viewport_anchor_id,
    matched_message_id,
    selected_message_id,
    message_recipient,
    message_body,
    display_name,
    bio,
    theme_color,
    currentPageId,
    
    // Actions
    setCurrentPageId
  }
}, {
  persist: {
    storage: sessionStorage
  }
})