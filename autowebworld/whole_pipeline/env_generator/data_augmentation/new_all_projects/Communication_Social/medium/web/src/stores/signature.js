import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useSignatureStore = defineStore('signature', () => {
  // Global FSM State
  const currentPageId = ref('HOME')
  
  // Signature Fields from FSM
  
  // HOME
  const current_user_id = ref('user_1') // Default logged in user
  const location_permission_granted = ref(null)
  const home_feed_posts = ref([])
  const home_selected_post_id = ref(null)
  const home_viewport_anchor_id = ref(null)
  const home_has_searched = ref(null)
  const home_matched_post_id = ref(null)
  const home_filters_applied = ref(null)
  const cookie_consent_given = ref(null)

  // POST_LIST
  const posts = ref([])
  const post_list_selected_post_id = ref(null)
  const post_list_viewport_anchor_id = ref(null)
  const post_list_has_searched = ref(null)
  const post_list_matched_post_id = ref(null)
  const post_list_filters_applied = ref(null)

  // POST_DETAIL
  const post_detail_post_id = ref(null)
  const post_is_bookmarked = ref(null)
  const post_clapped = ref(null)

  // COMMENT_FORM
  const comment_text = ref(null)
  const comment_preview_shown = ref(null)

  // SUCCESS Pages
  const success_message = ref(null)

  // NEW_STORY_EDITOR
  const draft_title = ref(null)
  const draft_subtitle = ref(null)
  const draft_body = ref(null)
  const draft_tag = ref(null)
  const draft_can_publish = ref(null)

  // PUBLISH_OPTIONS
  const publish_to_publication = ref(null)
  const allow_responses = ref(null)
  const selected_publish_option = ref(null)

  // PUBLISH_CONFIRM
  const ready_to_publish = ref(null)

  // SCHEDULE_PICKER
  const scheduled_datetime = ref(null)

  // PROFILE
  const profile_username = ref(null)
  const profile_bio = ref(null)
  const edit_name = ref(null)
  const edit_bio = ref(null)
  const edit_location = ref(null)
  const profile_can_save = ref(null)

  // STORIES_DRAFTS
  const drafts = ref([])
  const stories_viewport_anchor_id = ref(null)
  const stories_selected_draft_id = ref(null)
  const stories_filters_applied = ref(null)

  // PUBLICATION_LIST
  const publications = ref([])
  const publication_viewport_anchor_id = ref(null)
  const publication_selected_id = ref(null)
  const publication_filters_applied = ref(null)

  // PUBLICATION_DETAIL
  const publication_id = ref(null)

  // SETTINGS_PREFERENCES
  const dark_mode_enabled = ref(null)
  const email_notifications_enabled = ref(null)

  // MEMBERSHIP
  const membership_plan_selected = ref(null)

  // PAYMENT
  const card_number_entered = ref(null)
  const card_name_entered = ref(null)
  const card_cvv_entered = ref(null)
  const payment_ready = ref(null)

  function setCurrentPageId(id) {
    currentPageId.value = id
  }

  return {
    currentPageId,
    setCurrentPageId,
    
    current_user_id,
    location_permission_granted,
    home_feed_posts,
    home_selected_post_id,
    home_viewport_anchor_id,
    home_has_searched,
    home_matched_post_id,
    home_filters_applied,
    cookie_consent_given,
    
    posts,
    post_list_selected_post_id,
    post_list_viewport_anchor_id,
    post_list_has_searched,
    post_list_matched_post_id,
    post_list_filters_applied,
    
    post_detail_post_id,
    post_is_bookmarked,
    post_clapped,
    
    comment_text,
    comment_preview_shown,
    success_message,
    
    draft_title,
    draft_subtitle,
    draft_body,
    draft_tag,
    draft_can_publish,
    
    publish_to_publication,
    allow_responses,
    selected_publish_option,
    
    ready_to_publish,
    scheduled_datetime,
    
    profile_username,
    profile_bio,
    edit_name,
    edit_bio,
    edit_location,
    profile_can_save,
    
    drafts,
    stories_viewport_anchor_id,
    stories_selected_draft_id,
    stories_filters_applied,
    
    publications,
    publication_viewport_anchor_id,
    publication_selected_id,
    publication_filters_applied,
    publication_id,
    
    dark_mode_enabled,
    email_notifications_enabled,
    
    membership_plan_selected,
    
    card_number_entered,
    card_name_entered,
    card_cvv_entered,
    payment_ready
  }
}, {
  persist: {
    storage: sessionStorage
  }
})