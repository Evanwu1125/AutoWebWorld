import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export const useSignatureStore = defineStore('signature', () => {
  // Global State
  const currentPageId = ref('HOME')

  // FSM Signature Fields
  // User & Consent
  const current_user_id = ref('user_123') // Default logged in
  const cookie_consent_given = ref(null)
  
  // Permissions
  const location_permission_granted = ref(null)

  // Navigation & Selection State
  const selected_video_id = ref(null)
  const selected_playlist_id = ref(null)
  const selected_channel_id = ref(null)
  const matched_video_id = ref(null)
  const matched_playlist_id = ref(null)
  const matched_channel_id = ref(null)

  // Filters & Search
  const trending_filters_applied = ref(null)
  const trending_viewport_anchor_id = ref(null)
  const trending_has_searched = ref(null)

  const search_results_filters_applied = ref(null)
  const search_results_viewport_anchor_id = ref(null)
  const search_results_has_searched = ref(null)
  
  const library_filters_applied = ref(null)
  const library_viewport_anchor_id = ref(null)
  const library_has_searched = ref(null)

  const subscriptions_filters_applied = ref(null)
  const subscriptions_viewport_anchor_id = ref(null)
  const subscriptions_has_searched = ref(null)

  // Interaction State
  const is_liked = ref(null)
  const is_subscribed = ref(null)
  const comment_text_entered = ref(null)
  const confirm_checked = ref(null) // For subscription confirm

  // Upload Flow
  const file_selected = ref(null)
  const title_entered = ref(null)
  const description_entered = ref(null)
  const audience_selected = ref(null)
  const tags_entered = ref(null)
  const thumbnail_selected = ref(null)
  const visibility_selected = ref(null)
  const uploaded_video_id = ref(null)

  // Playlist Creation
  const playlist_title_entered = ref(null)
  const playlist_description_entered = ref(null)
  const playlist_privacy_selected = ref(null)
  const created_playlist_id = ref(null)

  // Lists (Mocked in data store, but refs kept here for FSM consistency if needed)
  const videos = ref([])
  const playlists = ref([])
  const channels = ref([])

  function setCurrentPageId(pageId) {
    currentPageId.value = pageId
  }

  return {
    currentPageId,
    current_user_id,
    cookie_consent_given,
    location_permission_granted,
    selected_video_id,
    selected_playlist_id,
    selected_channel_id,
    matched_video_id,
    matched_playlist_id,
    matched_channel_id,
    trending_filters_applied,
    trending_viewport_anchor_id,
    trending_has_searched,
    search_results_filters_applied,
    search_results_viewport_anchor_id,
    search_results_has_searched,
    library_filters_applied,
    library_viewport_anchor_id,
    library_has_searched,
    subscriptions_filters_applied,
    subscriptions_viewport_anchor_id,
    subscriptions_has_searched,
    is_liked,
    is_subscribed,
    comment_text_entered,
    confirm_checked,
    file_selected,
    title_entered,
    description_entered,
    audience_selected,
    tags_entered,
    thumbnail_selected,
    visibility_selected,
    uploaded_video_id,
    playlist_title_entered,
    playlist_description_entered,
    playlist_privacy_selected,
    created_playlist_id,
    videos,
    playlists,
    channels,
    setCurrentPageId
  }
}, {
  persist: {
    storage: sessionStorage
  }
})