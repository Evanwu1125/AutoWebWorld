import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useSignatureStore = defineStore('signature', () => {
  // Global State matching FSM signature_schema
  const current_user_id = ref('user_001')
  const cookie_consent_given = ref(null)
  
  // Data Collections
  const questions = ref([])
  const answers = ref([])
  const topics = ref([])
  const notifications = ref([])
  const bookmarked_questions = ref([])
  
  // Permissions
  const location_permission_granted = ref(null)
  
  // Filter & Search States
  const feed_filters_applied = ref(null)
  const matched_question_id = ref(null)
  const selected_question_id = ref(null)
  const feed_has_searched = ref(null)
  const feed_viewport_anchor_id = ref(null)
  
  const topics_filters_applied = ref(null)
  const matched_topic_id = ref(null)
  const selected_topic_id = ref(null)
  const topics_has_searched = ref(null)
  const topics_viewport_anchor_id = ref(null)
  
  const notifications_filters_applied = ref(null)
  const notifications_viewport_anchor_id = ref(null)
  
  const bookmarks_filters_applied = ref(null)
  const bookmarks_viewport_anchor_id = ref(null)
  
  // Drafts
  const draft_question_title = ref(null)
  const draft_question_details = ref(null)
  const draft_question_topic_id = ref(null)
  const answer_body_draft = ref(null)
  
  // Profile
  const profile_name = ref("Alex Designer")
  const profile_bio = ref("Product Designer & Tech Enthusiast")
  
  // System Messages
  const success_message = ref(null)
  
  // Navigation Tracking
  const currentPageId = ref('HOME')
  
  function setCurrentPageId(id) {
    currentPageId.value = id
  }

  // Reset/Clear functions for effects
  function clearFeedFilters() {
    feed_filters_applied.value = null
  }
  
  function clearTopicFilters() {
    topics_filters_applied.value = null
  }

  return {
    current_user_id,
    cookie_consent_given,
    questions,
    answers,
    topics,
    notifications,
    bookmarked_questions,
    location_permission_granted,
    feed_filters_applied,
    matched_question_id,
    selected_question_id,
    feed_has_searched,
    feed_viewport_anchor_id,
    topics_filters_applied,
    matched_topic_id,
    selected_topic_id,
    topics_has_searched,
    topics_viewport_anchor_id,
    notifications_filters_applied,
    notifications_viewport_anchor_id,
    bookmarks_filters_applied,
    bookmarks_viewport_anchor_id,
    draft_question_title,
    draft_question_details,
    draft_question_topic_id,
    answer_body_draft,
    profile_name,
    profile_bio,
    success_message,
    currentPageId,
    setCurrentPageId,
    clearFeedFilters,
    clearTopicFilters
  }
}, {
  persist: {
    storage: sessionStorage
  }
})